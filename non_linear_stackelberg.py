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
import Data.Non_linear_Stackelberg.Parking_Stackelberg_i2n10r50_Cap as data_file

class Stackelberg(object):
    def __init__(self):
        ### Mapping
        # Price
        current_index = 0
        self.p = np.empty(dict['I'] + 1, dtype = int)
        for i in range(len(self.p)):
            self.p[i] = current_index
            current_index += 1

        # Utility
        self.U = np.empty([dict['I'] + 1, dict['N']], dtype = int)
        for i in range(len(self.U)):
            for j in range(len(self.U[0])):
                self.U[i, j] = current_index
                current_index += 1

        # Choice
        self.w = np.empty([dict['I'] + 1, dict['N']], dtype = int)
        for i in range(len(self.w)):
            for j in range(len(self.w[0])):
                self.w[i, j] = current_index
                current_index += 1

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        expression = 0.0
        for i in range(dict['I'] + 1):
            for n in range(dict['N']):
                expression += x[self.p[i]] * x[self.w[i, n]]

        return expression

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        gradient = []
        # Price
        for i in range(len(self.p)):
            expression = 0.0
            for n in range(len(self.w[i])):
                #print ('index value : %r and type : %r '%(self.w[i, n], type(self.w[i, n])))
                expression += x[self.w[i, n]]
            gradient.append(expression)
        # Utility
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                gradient.append(0.0)
        # Choice
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                gradient.append(x[self.p[i]])

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
                print ('w[%r, %r] = %r' %(i, n, x[self.w[i, n]]))
                print ('minus = %r ' %(numerator/denominator))
                expression = x[self.w[i, n]] - numerator/denominator
                constraints.append(expression)

        # Utility value
        for i in range(len(self.U)):
            for n in range(len(self.U[i])):
                expression = x[self.U[i, n]] - dict['EndoCoef'][i, n] * x[self.p[i]] - dict['ExoUtility'][i, n]
                constraints.append(expression)

        print('Constraints : %r' %constraints)
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
                        jacobian.append(-dict['EndoCoef'][i, n])
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

        return jacobian

def main():
    #
    # Define the problem
    #
    x0 = []
    lb = []
    ub = []
    # Price variables
    for i in range(dict['I'] + 1):
        x0.append(0.0)
        lb.append(dict['lb_p'][i])
        ub.append(dict['ub_p'][i])
    # Utility variables
    for i in range(dict['I'] + 1):
        for n in range(dict['N']):
            x0.append(dict['lb_U'][i, n])
            lb.append(dict['lb_U'][i, n])
            ub.append(dict['ub_U'][i, n])
    # Choice variables
    for i in range(dict['I'] + 1):
        for n in range(dict['N']):
            if i == 0:
                x0.append(1.0)
            else:
                x0.append(0.0)
            lb.append(0.0)
            ub.append(1.0)

    cl = []
    cu = []
    # Probabilistic choice
    for i in range(dict['I'] + 1):
        for n in range(dict['N']):
            cl.append(0.0)
            cu.append(0.0)
    # Utility value
    for i in range(dict['I'] + 1):
        for n in range(dict['N']):
            cl.append(0.0)
            cu.append(0.0)

    x0 = [0.0,
        0.777975,
        0.639209,
        -14.0,
        -19.762,
        -19.762,
        -14.0,
        -14.0,
        -24.6043,
        -24.6043,
        -24.6043,
        -15.7042,
        -16.0504,
        -12.3191,
        -12.3191,
        -12.3191,
        -1.46306,
        -3.54344,
        0.156983,
        0.00058527,
        0.00058527,
        3.5915e-06,
        2.87582e-05,
        2.30935e-10,
        2.92469e-10,
        2.92469e-10,
        3.01066e-07,
        1.65236e-06,
        0.843017,
        0.999415,
        0.999415,
        0.999996,
        0.99997]

    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=Stackelberg(),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))

if __name__ == '__main__':
    # Get the data and preprocess
    dict = data_file.getData()
    data_file.preprocess(dict)
    # Solve the non linear model
    main()
