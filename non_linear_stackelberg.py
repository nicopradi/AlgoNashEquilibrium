# Modelisation of the sequential game with continuous price in a non linear model

# General
import sys
import time
import copy
# Ipopt
# Documentation/Example about ipopt:
# https://pythonhosted.org/ipopt/tutorial.html
import ipopt
# numpy
import numpy as np
# data
import Data.Non_linear_Stackelberg.ProbLogit_n10 as data_file

class Stackelberg(object):
    def __init__(self, **kwargs):
        ''' Construct a non linear Stackelberg game.
            KeywordArgs:
                I                 Number of alternatives
                N                 Number of customers
                endo_coef         Beta coefficient of the endogene variables
                exo_utility       Value of the utility for the exogene variables
                #### Optional kwargs ####
                optimizer         Index of the current operator
                operator          Mapping between alternative and operators
                p_fixed           Fixed price of the alternatives managed by other operators
                y_fixed           Fixed availability of the alternatives managed by other operators
                previous_revenue  Revenue of the current optimizer before its best reponse strategy
                capacity          Capacity value for each alternative [list]
                priority_list     Priority list for each alternative
        '''
        # Keyword arguments
        self.I = kwargs.get('I')
        self.N = kwargs.get('N')
        self.endo_coef = kwargs.get('endo_coef')
        self.exo_utility = kwargs.get('exo_utility')
        # Optional keyword arguments
        self.optimizer = kwargs.get('optimizer', None)
        self.operator = kwargs.get('operator', None)
        self.p_fixed = kwargs.get('p_fixed', None)
        self.y_fixed = kwargs.get('y_fixed', None)
        self.previous_revenue = kwargs.get('previous_revenue', 0.0)
        # Optional capacity constraints
        self.capacity = kwargs.get('capacity', None)
        self.priority_list = kwargs.get('priority_list', None)

        ### Mapping
        # Price variables
        current_index = 0
        self.p = np.empty(self.I + 1, dtype = int)
        for i in range(len(self.p)):
            self.p[i] = current_index
            current_index += 1

        # Utility variables
        self.U = np.empty([self.I + 1, self.N], dtype = int)
        for i in range(len(self.U)):
            for j in range(len(self.U[0])):
                self.U[i, j] = current_index
                current_index += 1

        # Choice variables
        self.w = np.empty([self.I + 1, self.N], dtype = int)
        for i in range(len(self.w)):
            for j in range(len(self.w[0])):
                self.w[i, j] = current_index
                current_index += 1

        # Capacity variables
        if self.capacity is not None:
            # Availability variables
            self.y = np.empty([self.I + 1, self.N], dtype = int)
            for i in range(len(self.y)):
                for j in range(len(self.y[0])):
                    self.y[i, j] = current_index
                    current_index += 1

    def objective(self, x):
        ''' The callback for calculating the objective
            This is a minimization problem.
        '''
        expression = 0.0
        for i in range(self.I + 1):
            # self.optimizer if None if and only if there is one operator
            if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                for n in range(self.N):
                    # Note that this is a minimization problem.
                    expression += -(x[self.p[i]] * x[self.w[i, n]])

        return expression

    def gradient(self, x):
        ''' The callback for calculating the gradient.
        '''
        gradient = []
        # Price variables
        for i in range(len(self.p)):
            expression = 0.0
            if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                for n in range(len(self.w[i])):
                    expression += -x[self.w[i, n]]
            gradient.append(expression)
        # Utility variables
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                gradient.append(0.0)
        # Choice variables
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                    gradient.append(-x[self.p[i]])
                else:
                    gradient.append(0.0)
        # Availability variables
        if self.capacity is not None:
            for i in range(len(self.y)):
                for n in range(len(self.y[i])):
                    gradient.append(0.0)

        return np.asarray(gradient)

    def constraints(self, x):
        ''' The callback for calculating the constraints.
        '''
        constraints = []
        # Probabilistic choice
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                expression = 0.0
                if self.capacity is not None:
                    numerator = float(np.exp(x[self.U[i, n]]) * x[self.y[i, n]])
                    denominator = 0.0
                    for j in range(len(self.w)):
                        denominator += (np.exp(x[self.U[j, n]]) * x[self.y[j, n]])
                    expression = x[self.w[i, n]] - (numerator/denominator)
                else:
                    numerator = float(np.exp(x[self.U[i, n]]))
                    denominator = 0.0
                    for j in range(len(self.w)):
                        denominator += np.exp(x[self.U[j, n]])
                    expression = x[self.w[i, n]] - numerator/denominator
                constraints.append(expression)

        # Utility value
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                expression = x[self.U[i, n]] - self.endo_coef[i, n] * x[self.p[i]] - self.exo_utility[i, n]
                constraints.append(expression)

        # Fixed prices
        if (self.optimizer is not None):
            for i in range(len(self.p)):
                if self.operator[i] != self.optimizer:
                    expression = x[self.p[i]] - self.p_fixed[i]
                    constraints.append(expression)

        # Capacity constraints
        if self.capacity is not None:
            # Ensure that y is a binary variable
            for i in range(len(self.y)):
                for n in range(len(self.y[0])):
                    expression = x[self.y[i, n]] * (1.0 - x[self.y[i, n]])
                    constraints.append(expression)
            # opt-out is always an available option
            for n in range(self.N):
                expression = x[self.y[0, n]]
                constraints.append(expression)
            # Capacity is not exceeded
            for i in range(self.I + 1):
                sum = 0
                for n in range(self. N):
                    sum += x[self.w[i, n]]
                expression = sum - self.capacity[i]
                constraints.append(expression)
            # Priority list, if y is 0 then the max capacity is reached
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    expression = self.capacity[i]*(1.0 - x[self.y[i, n]])
                    # Compute the number of customers with a higher priority which chose alternative i
                    sum = np.sum([x[self.w[i, m]] for m in range(self.N) if self.priority_list[i, m] < self.priority_list[i, n]])
                    expression += -sum
                    constraints.append(expression)
            # Priority list, if y is 1 then there is free room
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    # This type of constraint is revelant only if the capacity could be exceeded
                    if self.priority_list[i, n] > self.capacity[i]:
                        # Compute the number of customers with a higher priority which chose alternative i
                        sum = np.sum([x[self.w[i, m]] for m in range(self.N) if self.priority_list[i, m] < self.priority_list[i, n]])
                        expression = sum - (self.capacity[i] - 1)*x[self.y[i, n]] - (self.priority_list[i, n] - 1)*(1 - x[self.y[i, n]])
                        constraints.append(expression)

        return constraints

    def jacobian(self, x):
        ''' The callback for calculating the Jacobian.
        '''
        jacobian = []
        # For each constraint
            # For each variable
                # Append value to jacobian

        ### Probabilistic choice constraints
        # For each constraint
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                # For each variable
                # Price variables
                for j in range(len(self.p)):
                    jacobian.append(0.0)
                # Utility variables
                for j in range(len(self.U)):
                    for m in range(len(self.U[j])):
                        if m != n:
                            expression = 0.0
                        elif i == j:
                            if self.capacity is None:
                                sum = 0
                                # TODO: Put the sum in the outer loop to reduce running time, replace by np.sum
                                for k in range(len(self.U)):
                                    sum += np.exp(x[self.U[k, n]])
                                expression = -(np.exp(x[self.U[j, m]])*sum - np.exp(x[self.U[j, m]])**2)/(sum**2)
                            else:
                                sum = 0
                                for k in range(len(self.U)):
                                    #print('TERM: %r' %(np.exp(x[self.U[k, m]])*x[self.y[k, m]]))
                                    sum += (np.exp(x[self.U[k, m]])*x[self.y[k, m]])
                                #print('SUM: %r' %sum)
                                expression = -(np.exp(x[self.U[j, m]])*x[self.y[j, m]]*sum - (np.exp(x[self.U[j, m]])*x[self.y[j, m]])**2)/(sum**2)
                        else:
                            expression = 0.0
                            if self.capacity is None:
                                sum = 0
                                for k in range(len(self.U)):
                                    sum += np.exp(x[self.U[k, m]])
                                expression = (np.exp(x[self.U[i, m]])*np.exp(x[self.U[j, m]]))/(sum*sum)
                            else:
                                sum = 0
                                for k in range(len(self.U)):
                                    sum += (np.exp(x[self.U[k, m]])*x[self.y[k, m]])
                                expression = (np.exp(x[self.U[i, m]])*np.exp(x[self.U[j, m]])*x[self.y[i, m]]*x[self.y[j, m]])/(sum**2)
                        jacobian.append(expression)
                # Choice variables
                for j in range(len(self.w)):
                    for m in range(len(self.w[j])):
                        if (j == i) and (n == m):
                            jacobian.append(1.0)
                        else:
                            jacobian.append(0.0)
                # Availability variables
                if self.capacity is not None:
                    for j in range(len(self.y)):
                        for m in range(len(self.y[j])):
                            if (j == i) and (n == m):
                                sum = 0
                                for k in range(len(self.y)):
                                    sum += (np.exp(x[self.U[k, m]])*x[self.y[k, m]])
                                expression = -(np.exp(x[self.U[i, n]])*sum - x[self.y[i, n]]*(np.exp(x[self.U[i, n]])**2))/(sum**2)
                            elif (n == m):
                                sum = 0
                                for k in range(len(self.y)):
                                    sum += (np.exp(x[self.U[k, m]])*x[self.y[k, m]])
                                expression = np.exp(x[self.U[i, n]])*x[self.y[i, n]]*np.exp(x[self.U[j, n]])/(sum**2)
                            else:
                                expression = 0.0
                            jacobian.append(expression)

        #### Utility value constraints
        # For each constraint
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                # For each variable
                # Price variables
                for j in range(len(self.p)):
                    if j == i:
                        jacobian.append(-self.endo_coef[i, n])
                    else:
                        jacobian.append(0.0)
                # Utility variables
                for j in range(len(self.U)):
                    for m in range(len(self.U[j])):
                        if (j == i) and (n == m):
                            jacobian.append(1.0)
                        else:
                            jacobian.append(0.0)
                # Choice variables
                for j in range(len(self.w)):
                    for m in range(len(self.w[j])):
                        jacobian.append(0.0)
                # Availability variables
                if self.capacity is not None:
                    for j in range(len(self.y)):
                        for m in range(len(self.y[j])):
                            jacobian.append(0.0)

        #### Fixed prices constraints
        if (self.optimizer is not None):
            for i in range(len(self.p)):
                if self.operator[i] != self.optimizer:
                    # For each variable
                    # Price variables
                    for j in range(len(self.p)):
                        if i == j:
                            jacobian.append(1.0)
                        else:
                            jacobian.append(0.0)
                    # Utility variables
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            jacobian.append(0.0)
                    # Choice variables
                    for j in range(len(self.w)):
                        for m in range(len(self.w[j])):
                            jacobian.append(0.0)
                    # Availability variables
                    if self.capacity is not None:
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                jacobian.append(0.0)

        #### Capacity constraints
        if self.capacity is not None:
            #### Ensure that y is a binary variable
            for i in range(len(self.y)):
                for n in range(len(self.y[i])):
                    # For each variable
                    # Price variables
                    for j in range(len(self.p)):
                        jacobian.append(0.0)
                    # Utility variables
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            jacobian.append(0.0)
                    # Choice variables
                    for j in range(len(self.w)):
                        for m in range(len(self.w[j])):
                            jacobian.append(0.0)
                    # Availability variables
                    for j in range(len(self.y)):
                        for m in range(len(self.y[j])):
                            if (j == i) and (n == m):
                                jacobian.append(1.0 - 2*x[self.y[j, m]])
                            else:
                                jacobian.append(0.0)
            #### opt-out is always an available option
            for n in range(self.N):
                # For each variable
                # Price variables
                for j in range(len(self.p)):
                    jacobian.append(0.0)
                # Utility variables
                for j in range(len(self.U)):
                    for m in range(len(self.U[j])):
                        jacobian.append(0.0)
                # Choice variables
                for j in range(len(self.w)):
                    for m in range(len(self.w[j])):
                        jacobian.append(0.0)
                # Availability variables
                for j in range(len(self.y)):
                    for m in range(len(self.y[j])):
                        if (j == 0) and (n == m):
                            jacobian.append(1.0)
                        else:
                            jacobian.append(0.0)
            #### capacity is not exceeded
            for i in range(self.I + 1):
                # For each variable
                # Price variables
                for j in range(len(self.p)):
                    jacobian.append(0.0)
                # Utility variables
                for j in range(len(self.U)):
                    for m in range(len(self.U[j])):
                        jacobian.append(0.0)
                # Choice variables
                for j in range(len(self.w)):
                    for m in range(len(self.w[j])):
                        if j == i:
                            jacobian.append(1.0)
                        else:
                            jacobian.append(0.0)
                # Availability variables
                for j in range(len(self.y)):
                    for m in range(len(self.y[j])):
                        jacobian.append(0.0)
            #### Priority list, if y[i, n] is 0 then the max capacity of alternative i is reached
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    # For each variable
                    # Price variables
                    for j in range(len(self.p)):
                        jacobian.append(0.0)
                    # Utility variables
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            jacobian.append(0.0)
                    # Choice variables
                    for j in range(len(self.w)):
                        for m in range(len(self.w[j])):
                            if (j == i) and (self.priority_list[i, m] < self.priority_list[i, n]):
                                jacobian.append(-1.0)
                            else:
                                jacobian.append(0.0)
                    # Availability variables
                    for j in range(len(self.y)):
                        for m in range(len(self.y[j])):
                            if (j == i) and (m == n):
                                jacobian.append(-self.capacity[i])
                            else:
                                jacobian.append(0.0)
            #### Priority list, if y[i, n] is 1 then there is free room for n to choose alternative i
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    # This type of constraint is revelant only if the capacity could be exceeded
                    if self.priority_list[i, n] > self.capacity[i]:
                        # For each variable
                        # Price variables
                        for j in range(len(self.p)):
                            jacobian.append(0.0)
                        # Utility variables
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                jacobian.append(0.0)
                        # Choice variables
                        for j in range(len(self.w)):
                            for m in range(len(self.w[j])):
                                if (j == i) and (self.priority_list[i, m] < self.priority_list[i, n]):
                                    jacobian.append(1.0)
                                else:
                                    jacobian.append(0.0)
                        # Availability variables
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                if (j == i) and (m == n):
                                    jacobian.append(-self.capacity[i] + self.priority_list[i, n])
                                else:
                                    jacobian.append(0.0)

        return jacobian

def main(data):
    ''' Define the problem.
    '''

    x0 = []
    # Lower bound and upper bound on the decision variables
    lb = []
    ub = []
    # Price variables
    for i in range(data['I'] + 1):
        x0.append((data['lb_p'][i]+data['ub_p'][i])/2.0)
        lb.append(data['lb_p'][i])
        ub.append(data['ub_p'][i])
    # Utility variables
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            x0.append((data['lb_U'][i, n]+data['ub_U'][i, n])/2.0)
            lb.append(data['lb_U'][i, n])
            ub.append(data['ub_U'][i, n])
    # Choice variables
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            if i == 0:
                x0.append(1.0)
            else:
                x0.append(0.0)
            lb.append(0.0)
            ub.append(1.0)
    # Capacity variables
    if 'capacity' in data.keys():
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                x0.append(1.0)
                lb.append(0.0)
                ub.append(1.0)

    # x0 is the starting point of the interior point method
    # Start from the previous operator's strategies if given
    if 'x0' in data.keys():
        x0 = copy.deepcopy(data['x0'])
    else:
        x0 = getInitialPoint(data)

    # Lower bound and upper bound on the constraints
    cl = []
    cu = []
    #TODO: Adjust tolerance value, make it a keyword argument
    tol = 1e-3
    # Probabilistic choice constraints
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            cl.append(-tol)
            cu.append(tol)
    # Utility value constraints
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            cl.append(-tol)
            cu.append(tol)
    # Fixed prices constraints
    if 'optimizer' in data.keys():
        for i in range(data['I'] + 1):
            if data['operator'][i] != data['optimizer']:
                cl.append(-tol)
                cu.append(tol)
    # capacity constraints
    if 'capacity' in data.keys():
        # Ensure that y is a binary variable
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                cl.append(-tol)
                cu.append(tol)
        # opt-out is always an available option
        for n in range(data['N']):
            cl.append(1.0 - tol)
            cu.append(1.0 + tol)
        # Capacity is not exceeded
        for i in range(data['I'] + 1):
            cl.append(-data['capacity'][i] - tol)
            cu.append(tol)
        # Priority list, if y is 0 then the max capacity is reached
        for i in range(1, data['I'] + 1):
            for n in range(data['N']):
                cl.append(-data['N'] - 1.0 - tol)
                cu.append(1.0 + tol)
        # Priority list, if y is 1 then there is free room
        for i in range(1, data['I'] + 1):
            for n in range(data['N']):
                # This type of constraint is revelant only if the capacity could be exceeded
                if data['priority_list'][i, n] > data['capacity'][i]:
                    cl.append(-data['N'] - 1.0 - tol)
                    cu.append(tol)

    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=Stackelberg(**data),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    # Set the parameters
    nlp.addOption('print_level', 0)
    nlp.addOption('max_iter', 1500)
    nlp.addOption('warm_start_init_point', 'yes')
    nlp.addOption('warm_start_bound_push', 1e-19)
    nlp.addOption('warm_start_bound_frac', 1e-19)
    nlp.addOption('warm_start_slack_bound_frac', 1e-19)
    nlp.addOption('warm_start_slack_bound_push', 1e-19)
    nlp.addOption('warm_start_mult_bound_push', 1e-19)
    # Solve the problem
    x, info = nlp.solve(x0)

    # Change the sign of the optimal objective function value
    # (conversion of a minimization problem to a maximization)
    info['obj_val'] = -info['obj_val']
    # Print the solution
    printSolution(data, x, info)
    choice_start = data['I'] + 1 + data['N']*(data['I'] + 1)
    choice_end = data['I'] + 1 + 2*data['N']*(data['I'] + 1)

    return x[:data['I'] + 1], x0[choice_start:choice_end], x, info['status'], info['status_msg']

def getInitialPoint(data):
    ''' Compute an initial feasible solution to the best response game.
        The initial solution of the interior point algorithm will be set at this
        solution.
    '''

    print('\n--- Compute an initial x0 ----')

    x0 = []
    count = 0
    # Price variables
    p_index = np.empty([data['I'] + 1], dtype = int)
    for i in range(data['I'] + 1):
        if data['operator'][i] != data['optimizer']:
            x0.append(data['p_fixed'][i])
        else:
            x0.append((data['ub_p'][i]+data['lb_p'][i])/2.0)
        p_index[i] = count
        count += 1

    # Utility variables
    u_index = np.empty([data['I'] + 1, data['N']], dtype = int)
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            x0.append(data['endo_coef'][i, n]*x0[p_index[i]] + data['exo_utility'][i, n])
            u_index[i, n] = count
            count += 1

    # Compute choice variables and availability variables
    # according to whether capacities are given
    if 'capacity' in data.keys():
        # Initialize the value of the variables in x0
        # Choice variables
        w_index = np.empty([data['I'] + 1, data['N']], dtype = int)
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                x0.append(1.0)
                w_index[i, n] = count
                count += 1
        # Availability variables
        y_index = np.empty([data['I'] + 1, data['N']], dtype = int)
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                x0.append(1.0)
                y_index[i, n] = count
                count += 1

        # Compute the choice and availability variables until convergence to a feasible solution
        feasible = False
        while feasible is False:
            # Choice variables
            for i in range(data['I'] + 1):
                for n in range(data['N']):
                    denominator = np.sum([np.exp(x0[u_index[j, n]])*x0[y_index[j, n]] for j in range(data['I'] + 1)])
                    numerator = np.exp(x0[u_index[i, n]])*x0[y_index[i, n]]
                    x0[w_index[i, n]] = float(numerator)/denominator
            # Check if the solution is feasible
            feasible = True
            capa = copy.deepcopy(data['capacity'])
            for i in range(data['I'] + 1):
                occupancy = 0.0
                for n in range(data['N']):
                    occupancy += x0[w_index[i, n]]
                    if capa[i] >= 1.0:
                        capa[i] -= x0[w_index[i, n]]
                        if x0[y_index[i, n]] == 0:
                            feasible = False
                            x0[y_index[i, n]] = 1.0
                    else:
                        if x0[y_index[i, n]] == 1:
                            feasible = False
                            x0[y_index[i, n]] = 0.0
                    if occupancy > data['capacity'][i]:
                        feasible = False
                        x0[y_index[i, n]] = 0.0

    else:
        # Choice variables
        w_index = np.empty([data['I'] + 1, data['N']], dtype = int)
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                denominator = np.sum([np.exp(x0[u_index[j, n]]) for j in range(data['I'] + 1)])
                numerator = np.exp(x0[u_index[i, n]])
                x0.append(float(numerator)/denominator)
                w_index[i, n] = count
                count += 1

    return x0

def printSolution(data, x, info):

    print('\nResults:')
    print('Decision variables: \n')
    # Price variables
    counter = 0
    for i in range(data['I'] + 1):
        print('Price of alternative %r: %r'%(i, x[counter]))
        counter += 1
    print('\n')
    '''
    # Utility variables
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            print('Utility of alternative %r for user %r : %r'%(i, n, x[counter]))
            counter += 1
    print('\n')
    #TODO: Compute the revenue/market share
    # Choice variables
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            print('Choice of alternative %r for user %r : %r'%(i, n, x[counter]))
            counter += 1
    print('\n')
    # Availability variables
    if 'capacity' in data.keys():
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                print('Availability of alternative %r for user %r : %r'%(i, n, x[counter]))
                counter += 1
        print('\n')
    '''
    print("Objective function(revenue) = %r\n" % info['obj_val'])


if __name__ == '__main__':
    # Get the data and preprocess
    data = data_file.getData()
    data_file.preprocess(data)
    # Solve the non linear model
    t0 = time.time()
    main(data)
    print('Total running time: %r ' %(time.time() - t0))
