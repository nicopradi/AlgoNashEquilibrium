# Modelisation of the sequential game with continuous price in a non linear n^mdoel

# General
import sys
import time
import copy
# Ipopt
import ipopt
# numpy
import numpy as np
# data
import Data.Non_linear_Stackelberg.ProbLogit_n10 as data_file

class Stackelberg(object):
    def __init__(self, **kwargs):
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

        ### Mapping
        # Price
        current_index = 0
        self.p = np.empty(self.I + 1, dtype = int)
        for i in range(len(self.p)):
            self.p[i] = current_index
            current_index += 1

        # Utility
        self.U = np.empty([self.I + 1, self.N], dtype = int)
        for i in range(len(self.U)):
            for j in range(len(self.U[0])):
                self.U[i, j] = current_index
                current_index += 1

        # Choice
        self.w = np.empty([self.I + 1, self.N], dtype = int)
        for i in range(len(self.w)):
            for j in range(len(self.w[0])):
                self.w[i, j] = current_index
                current_index += 1

    def objective(self, x):
        #
        # The callback for calculating the objective
        # This is a minimization problem
        #
        expression = 0.0
        for i in range(self.I + 1):
            if (self.Optimizer is None) or (self.Operator[i] == self.Optimizer):
                for n in range(self.N):
                    expression += -(x[self.p[i]] * x[self.w[i, n]])

        return expression

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        gradient = []
        # Price
        for i in range(len(self.p)):
            expression = 0.0
            if (self.Optimizer is None) or (self.Operator[i] == self.Optimizer):
                for n in range(len(self.w[i])):
                    #print ('index value : %r and type : %r '%(self.w[i, n], type(self.w[i, n])))
                    expression += -x[self.w[i, n]]
            gradient.append(expression)
        # Utility
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                gradient.append(0.0)
        # Choice
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                if (self.Optimizer is None) or (self.Operator[i] == self.Optimizer):
                    gradient.append(-x[self.p[i]])
                else:
                    gradient.append(0.0)

        return np.asarray(gradient)

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        constraints = []
        # Probabilistic choice
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                expression = 0.0
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

        return constraints

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
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
                            expression = 0.0
                            sum = 0
                            for k in range(len(self.U)):
                                sum += np.exp(x[self.U[k, m]])
                            expression = -(np.exp(x[self.U[j, m]])*sum - np.exp(x[self.U[j, m]])*np.exp(x[self.U[j, m]]))/(sum*sum)
                        else:
                            expression = 0.0
                            sum = 0
                            for k in range(len(self.U)):
                                sum += np.exp(x[self.U[k, m]])
                            expression = (np.exp(x[self.U[i, m]])*np.exp(x[self.U[j, m]]))/(sum*sum)
                        jacobian.append(expression)
                # Choice variables
                for j in range(len(self.w)):
                    for m in range(len(self.w[j])):
                        if (j == i) and (n == m):
                            jacobian.append(1.0)
                        else:
                            jacobian.append(0.0)

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
        return jacobian

def main(data):
    #
    # Define the problem
    #
    x0 = []
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

    cl = []
    cu = []
    #TODO: Adjust tolerance
    # Probabilistic choice
    for i in range(data['I'] + 1):
        for n in range(data['N']):
            cl.append(-1e-6)
            cu.append(1e-6)
    # Utility value
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
    # Solve the problem
    x, info = nlp.solve(x0)
    # Change the sign of the optimal objective function value
    # (conversion of a maximimazion problem to a minimization)
    info['obj_val'] = -info['obj_val']
    # Print the solution
    printSolution(data, x, info)

    return x[:data['I'] + 1]

def printSolution(data, x, info):

    print('Decision variables: \n')
    # Price variables
    counter = 0
    for i in range(data['I'] + 1):
        print('Price of alternative %r: %r'%(i, x[counter]))
        counter += 1
    print('\n')
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
    print("Objective function(revenue) = %r\n" % info['obj_val'])

if __name__ == '__main__':
    # Get the data and preprocess
    data = data_file.getData()
    data_file.preprocess(data)
    # Solve the non linear model
    main(data)
