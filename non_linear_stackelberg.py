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
                I               Number of alternatives
                N               Number of customers
                EndoCoef        Beta coefficient of the endogene variables
                ExoUtility      Value of the utility for the exogene variables
                #### Optional kwargs ####
                Optimizer       Index of the current operator
                Operator        Mapping between alternative and operators
                p_fixed         Fixed price of the alternatives managed by other operators
                y_fixed         Fixed availability of the alternatives managed by other operators
                Capacity        Capacity value for each alternative [list]
                PriorityList    Priority list for each alternative
        '''
        # Keyword arguments
        self.I = kwargs.get('I')
        self.N = kwargs.get('N')
        self.EndoCoef = kwargs.get('EndoCoef')
        self.ExoUtility = kwargs.get('ExoUtility')
        # Optional keyword arguments
        self.Optimizer = kwargs.get('Optimizer', None)
        self.Operator = kwargs.get('Operator', None)
        self.p_fixed = kwargs.get('p_fixed', None)
        self.y_fixed = kwargs.get('y_fixed', None)
        # Optinal capacity constraints
        self.Capacity = kwargs.get('Capacity', None)
        self.PriorityList = kwargs.get('PriorityList', None)

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
        if self.Capacity is not None:
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
            if (self.Optimizer is None) or (self.Operator[i] == self.Optimizer):
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
            if (self.Optimizer is None) or (self.Operator[i] == self.Optimizer):
                for n in range(len(self.w[i])):
                    #print ('index value : %r and type : %r '%(self.w[i, n], type(self.w[i, n])))
                    expression += -x[self.w[i, n]]
            gradient.append(expression)
        # Utility variables
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                gradient.append(0.0)
        # Choice variables
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                if (self.Optimizer is None) or (self.Operator[i] == self.Optimizer):
                    gradient.append(-x[self.p[i]])
                else:
                    gradient.append(0.0)
        # Availability variables
        if self.Capacity is not None:
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
                if self.Capacity is not None:
                    numerator = float(np.exp(x[self.U[i, n]]) * x[self.y[i, n]])
                    denominator = 0.0
                    for j in range(len(self.w)):
                        denominator += (np.exp(x[self.U[j, n]]) * x[self.y[j, n]])
                    expression = x[self.w[i, n]] - numerator/denominator
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
                expression = x[self.U[i, n]] - self.EndoCoef[i, n] * x[self.p[i]] - self.ExoUtility[i, n]
                constraints.append(expression)

        # Fixed prices
        if (self.Optimizer is not None):
            for i in range(len(self.p)):
                if self.Operator[i] != self.Optimizer:
                    expression = x[self.p[i]] - self.p_fixed[i]
                    constraints.append(expression)

        # Capacity constraints
        if self.Capacity is not None:
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
                expression = sum - self.Capacity[i]
                constraints.append(expression)
            # Priority list, if y is 0 then the max capacity is reached
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    expression = self.Capacity[i]*(1.0 - x[self.y[i, n]])
                    # Compute the number of customers with a higher priority which chose alternative i
                    sum = np.sum([x[self.w[i, m]] for m in range(self.N) if self.PriorityList[i, m] < self.PriorityList[i, n]])
                    expression += -sum
                    constraints.append(expression)
            # Priority list, if y is 1 then there is free room
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    # This type of constraint is revelant only if the capacity could be exceeded
                    if self.PriorityList[i, n] > self.Capacity[i]:
                        # Compute the number of customers with a higher priority which chose alternative i
                        sum = np.sum([x[self.w[i, m]] for m in range(self.N) if self.PriorityList[i, m] < self.PriorityList[i, n]])
                        expression = sum - (self.Capacity[i] - 1)*x[self.y[i, n]] - (self.PriorityList[i, n] - 1)*(1 - x[self.y[i, n]])
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
                            if self.Capacity is None:
                                sum = 0
                                # TODO: Put the sum in the outer loop to reduce running time, replace by np.sum
                                for k in range(len(self.U)):
                                    sum += np.exp(x[self.U[k, n]])
                                expression = -(np.exp(x[self.U[j, m]])*sum - np.exp(x[self.U[j, m]])**2)/(sum**2)
                            else:
                                sum = 0
                                for k in range(len(self.U)):
                                    sum += (np.exp(x[self.U[k, m]])*x[self.y[k, m]])
                                expression = -(np.exp(x[self.U[j, m]])*x[self.y[j, m]]*sum - (np.exp(x[self.U[j, m]])*x[self.y[j, m]])**2)/(sum**2)
                        else:
                            expression = 0.0
                            if self.Capacity is None:
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
                if self.Capacity is not None:
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
                        jacobian.append(-self.EndoCoef[i, n])
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
                if self.Capacity is not None:
                    for j in range(len(self.y)):
                        for m in range(len(self.y[j])):
                            jacobian.append(0.0)

        #### Fixed prices constraints
        if (self.Optimizer is not None):
            for i in range(len(self.p)):
                if self.Operator[i] != self.Optimizer:
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
                    if self.Capacity is not None:
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                jacobian.append(0.0)

        #### Capacity constraints
        if self.Capacity is not None:
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
            #### Capacity is not exceeded
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
                            if (j == i) and (self.PriorityList[i, m] < self.PriorityList[i, n]):
                                jacobian.append(-1.0)
                            else:
                                jacobian.append(0.0)
                    # Availability variables
                    for j in range(len(self.y)):
                        for m in range(len(self.y[j])):
                            if (j == i) and (m == n):
                                jacobian.append(-self.Capacity[i])
                            else:
                                jacobian.append(0.0)
            #### Priority list, if y[i, n] is 1 then there is free room for n to choose alternative i
            for i in range(1, self.I + 1):
                for n in range(self.N):
                    # This type of constraint is revelant only if the capacity could be exceeded
                    if self.PriorityList[i, n] > self.Capacity[i]:
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
                                if (j == i) and (self.PriorityList[i, m] < self.PriorityList[i, n]):
                                    jacobian.append(1.0)
                                else:
                                    jacobian.append(0.0)
                        # Availability variables
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                if (j == i) and (m == n):
                                    jacobian.append(-self.Capacity[i] + self.PriorityList[i, n])
                                else:
                                    jacobian.append(0.0)

        return jacobian

def main(data):
    ''' Define the problem.
    '''
    # x0 is the starting point of the interior point method
    x0 = []
    # Lower bound and upper bound on the decision variables
    lb = []
    ub = []
    # Price variables
    for i in range(data['I'] + 1):
        x0.append(0.0)
        lb.append(data['lb_p'][i])
        ub.append(data['ub_p'][i])
    # Utility variables
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            x0.append(data['lb_U'][i, n])
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
    if 'Capacity' in data.keys():
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                x0.append(1.0)
                lb.append(0.0)
                ub.append(1.0)

    # Lower bound and upper bound on the constraints
    cl = []
    cu = []
    #TODO: Adjust tolerance value, make it a keyword argument
    # Probabilistic choice constraints
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            cl.append(-1e-6)
            cu.append(1e-6)
    # Utility value constraints
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            cl.append(-1e-6)
            cu.append(1e-6)
    # Fixed prices constraints
    if 'Optimizer' in data.keys():
        for i in range(data['I'] + 1):
            if data['Operator'][i] != data['Optimizer']:
                cl.append(-1e-6)
                cu.append(1e-6)
    # Capacity constraints
    if 'Capacity' in data.keys():
        # Ensure that y is a binary variable
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                cl.append(0.0)
                cu.append(0.0)
        # opt-out is always an available option
        for n in range(data['N']):
            cl.append(1.0 - 1e-6)
            cu.append(1.0 + 1e-6)
        # Capacity is not exceeded
        for i in range(data['I'] + 1):
            cl.append(-data['Capacity'][i] - 1e-6)
            cu.append(1e-6)
        # Priority list, if y is 0 then the max capacity is reached
        for i in range(1, data['I'] + 1):
            for n in range(data['N']):
                cl.append(-data['N'])
                cu.append(1e-6)
        # Priority list, if y is 1 then there is free room
        for i in range(1, data['I'] + 1):
            for n in range(data['N']):
                # This type of constraint is revelant only if the capacity could be exceeded
                if data['PriorityList'][i, n] > data['Capacity'][i]:
                    cl.append(-data['N'])
                    cu.append(1e-6)

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
    nlp.addOption('max_iter', 3000)
    # Solve the problem
    x, info = nlp.solve(x0)
    # Change the sign of the optimal objective function value
    # (conversion of a minimization problem to a maximization)
    info['obj_val'] = -info['obj_val']
    # Print the solution
    printSolution(data, x, info)

    return x[:data['I'] + 1]

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
    # Choice variables
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            print('Choice of alternative %r for user %r : %r'%(i, n, x[counter]))
            counter += 1
    print('\n')
    # Availability variables
    if 'Capacity' in data.keys():
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
