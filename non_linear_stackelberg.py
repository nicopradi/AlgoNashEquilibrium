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
import Data.Parking_lot.Non_linear_Stackelberg.ProbLogit_n10 as data_file

class Stackelberg(object):
    def __init__(self, **kwargs):
        ''' Construct a non linear Stackelberg game.
            KeywordArgs:
                I                 Number of alternatives (without opt-out) [int]
                I_opt_out         Number of opt-out alternatives [int]
                N                 Number of customers [int]
                endo_coef         Beta coefficient of the endogene variables [list]
                exo_utility       Value of the utility for the exogene variables [list]
                #### Optional kwargs ####
                optimizer         Index of the operator playing its best reponse strategy [int]
                operator          Mapping between alternative and operators [list]
                p_fixed           Fixed price of the alternatives managed by other operators [list]
                y_fixed           Fixed availability of the alternatives managed by other operators [list]
                capacity          Capacity value for each alternative [list]
                choice_set        Individual choice sets [list]
                priority_list     Priority list for each alternative [list]
                R_coef            Number of draws for the beta coefficient [int]
                fixed_cost        Initial cost of an alternative [list]
                customer_cost     Additional cost of an alternative for each addition customer [list]
        '''
        # Keyword arguments
        self.I = kwargs.get('I')
        self.I_opt_out = kwargs.get('I_opt_out', 1)
        self.N = kwargs.get('N')
        self.endo_coef = kwargs.get('endo_coef')
        self.exo_utility = kwargs.get('exo_utility')
        # Optional keyword arguments
        self.optimizer = kwargs.get('optimizer', None)
        self.operator = kwargs.get('operator', None)
        self.p_fixed = kwargs.get('p_fixed', None)
        self.y_fixed = kwargs.get('y_fixed', None)
        self.capacity = kwargs.get('capacity', None)
        self.choice_set = kwargs.get('choice_set', None)
        self.priority_list = kwargs.get('priority_list', None)
        # R_coef determines the DCM: Mixed logit or Logit
        self.R_coef = kwargs.get('R_coef', None)
        self.fixed_cost = kwargs.get('fixed_cost', None)
        self.customer_cost = kwargs.get('customer_cost', None)

        ### Mapping between decision variable and an index value
        # Variable used to assign an index to a variable
        current_index = 0

        # Price variables
        self.p = np.empty(self.I + self.I_opt_out, dtype = int)
        for i in range(len(self.p)):
            self.p[i] = current_index
            current_index += 1

        # Utility variables
        if self.R_coef is None:
            # Logit
            self.U = np.empty([self.I + self.I_opt_out, self.N], dtype = int)
            for i in range(len(self.U)):
                for j in range(len(self.U[0])):
                    self.U[i, j] = current_index
                    current_index += 1
        else:
            # Mixed logit
            self.U = np.empty([self.I + self.I_opt_out, self.N, self.R_coef], dtype = int)
            for i in range(len(self.U)):
                for j in range(len(self.U[i])):
                    for r in range(len(self.U[i, j])):
                        self.U[i, j, r] = current_index
                        current_index += 1

        # Choice variables
        self.w = np.empty([self.I + self.I_opt_out, self.N], dtype = int)
        for i in range(len(self.w)):
            for j in range(len(self.w[0])):
                self.w[i, j] = current_index
                current_index += 1

        # Capacity variables
        if self.capacity is not None:
            # Availability variables
            self.y = np.empty([self.I + self.I_opt_out, self.N], dtype = int)
            for i in range(len(self.y)):
                for j in range(len(self.y[0])):
                    self.y[i, j] = current_index
                    current_index += 1

    def objective(self, x):
        ''' The callback for calculating the objective function.
            Note that this is a minimization problem.
        '''
        expression = 0.0
        # Add revenue to the objective function
        for i in range(self.I_opt_out, self.I + self.I_opt_out):
            # Consider the alternatives managed by the optimizer only
            if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                for n in range(self.N):
                    # Note that this is a minimization problem.
                    expression += -(x[self.p[i]] * x[self.w[i, n]])

        # Add the initial cost and customer cost
        if self.fixed_cost is not None:
            for i in range(self.I_opt_out, self.I + self.I_opt_out):
                if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                    # Initial cost
                    expression += self.fixed_cost[i]
                    # Customer cost
                    if self.customer_cost is not None:
                        for n in range(self.N):
                            expression += self.customer_cost[i] * x[self.w[i, n]]

        return expression

    def gradient(self, x):
        ''' The callback for calculating the gradient of the objective function.
        '''
        gradient = []

        ### Price variables
        for i in range(len(self.p)):
            expression = 0.0
            if (i >= self.I_opt_out) and ((self.optimizer is None) or (self.operator[i] == self.optimizer)):
                for n in range(len(self.w[i])):
                    expression += -x[self.w[i, n]]
            gradient.append(expression)

        ### Utility variables
        if self.R_coef is None:
            for i in range(len(self.U)):
                for n in range(len(self.U[i])):
                    gradient.append(0.0)
        else:
            for i in range(len(self.U)):
                for n in range(len(self.U[i])):
                    for r in range(len(self.U[i, n])):
                        gradient.append(0.0)

        ### Choice variables
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                if (i >= self.I_opt_out) and ((self.optimizer is None) or (self.operator[i] == self.optimizer)):
                    if self.customer_cost is not None:
                        gradient.append(-x[self.p[i]] + self.customer_cost[i])
                    else:
                        gradient.append(-x[self.p[i]])
                else:
                    gradient.append(0.0)

        ### Availability variables
        if self.capacity is not None:
            for i in range(len(self.y)):
                for n in range(len(self.y[i])):
                    gradient.append(0.0)

        return np.asarray(gradient)

    def constraints(self, x):
        ''' The callback for defining the constraints.
        '''
        constraints = []

        ### Probabilistic choice constraints
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                expression = 0.0
                # Capacitated model
                if self.capacity is not None:
                    # Check if the DCM is the Logit or Mixed logit model
                    if self.R_coef is None:
                        # Logit
                        numerator = float(np.exp(x[self.U[i, n]]) * x[self.y[i, n]])
                        denominator = 0.0
                        for j in range(len(self.w)):
                            denominator += (np.exp(x[self.U[j, n]]) * x[self.y[j, n]])
                        expression = x[self.w[i, n]] - (numerator/denominator)
                    else:
                        # Mixed logit
                        sum = 0
                        for r in range(self.R_coef):
                            numerator = float(np.exp(x[self.U[i, n, r]]) * x[self.y[i, n]])
                            denominator = 0.0
                            for j in range(len(self.w)):
                                denominator += (np.exp(x[self.U[j, n, r]]) * x[self.y[j, n]])
                            sum += (numerator/denominator)
                        sum = sum/self.R_coef
                        expression = x[self.w[i, n]] - sum
                # Uncapacitated model
                else:
                    if self.R_coef is None:
                        if self.choice_set[i, n] == 0:
                            expression = x[self.w[i, n]]
                        else:
                            numerator = float(np.exp(x[self.U[i, n]]))
                            denominator = 0.0
                            for j in range(len(self.w)):
                                denominator += np.exp(x[self.U[j, n]])
                            expression = x[self.w[i, n]] - numerator/denominator
                    else:
                        if self.choice_set[i, n] == 0:
                            expression = x[self.w[i, n]]
                        else:
                            sum = 0
                            for r in range(self.R_coef):
                                numerator = float(np.exp(x[self.U[i, n, r]]))
                                denominator = 0.0
                                for j in range(len(self.w)):
                                    denominator += (np.exp(x[self.U[j, n, r]]))
                                sum += (numerator/denominator)
                            sum = sum/self.R_coef
                            expression = x[self.w[i, n]] - sum

                constraints.append(expression)

        ### Utility value constraints
        if self.R_coef is None:
            # Logit
            for i in range(len(self.U)):
                for n in range(len(self.U[i])):
                    expression = x[self.U[i, n]] - self.endo_coef[i, n] * x[self.p[i]] - self.exo_utility[i, n]
                    constraints.append(expression)
        else:
            # Mixed logit
            for i in range(len(self.U)):
                for n in range(len(self.U[i])):
                    for r in range(len(self.U[i, n])):
                        expression = x[self.U[i, n, r]] - self.endo_coef[i, n, r] * x[self.p[i]] - self.exo_utility[i, n, r]
                        constraints.append(expression)


        ### Fixed prices constraints
        if (self.optimizer is not None):
            for i in range(len(self.p)):
                if self.operator[i] != self.optimizer:
                    expression = x[self.p[i]] - self.p_fixed[i]
                    constraints.append(expression)

        ### Capacity constraints
        if self.capacity is not None:
            # Ensure that y is a binary variable
            for i in range(len(self.y)):
                for n in range(len(self.y[0])):
                    expression = x[self.y[i, n]] * (1.0 - x[self.y[i, n]])
                    constraints.append(expression)
            # Alternative availability is subject to the choice-set
            for i in range(self.I_opt_out):
                for n in range(self.N):
                    if self.choice_set[i, n] == 0:
                        expression = x[self.y[i, n]]
                        constraints.append(expression)
            # Capacity is not exceeded
            for i in range(self.I + self.I_opt_out):
                sum = 0
                for n in range(self. N):
                    sum += x[self.w[i, n]]
                expression = sum - self.capacity[i]
                constraints.append(expression)
            # Priority list, if y is 0 then the max capacity is reached, or the alternative is not available in the choice set
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    expression = self.choice_set[i, n]*self.capacity[i]*(1.0 - x[self.y[i, n]])
                    # Compute the number of customers with a higher priority which chose alternative i
                    sum = np.sum([x[self.w[i, m]] for m in range(self.N) if self.priority_list[i, m] < self.priority_list[i, n]])
                    expression += -sum
                    constraints.append(expression)
            # Priority list, if y is 1 then there is free room
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    # This type of constraint is revelant only if the capacity could be exceeded
                    if self.priority_list[i, n] > self.capacity[i]:
                        # Compute the number of customers with a higher priority which chose alternative i
                        sum = np.sum([x[self.w[i, m]] for m in range(self.N) if self.priority_list[i, m] < self.priority_list[i, n]])
                        expression = sum - (self.capacity[i] - 1)*x[self.y[i, n]] - (self.priority_list[i, n] - 1)*(1 - x[self.y[i, n]])
                        constraints.append(expression)

        return constraints

    def jacobian(self, x):
        ''' The callback for calculating the Jacobian. For each constraint,
            compute the partial derivative for each variable.
        '''
        jacobian = []
        # Schema:
        # For each constraint
            # For each variable
                # Append value to jacobian

        ### Probabilistic choice constraints
        # For each constraints
        for i in range(len(self.w)):
            for n in range(len(self.w[i])):
                # For each variable
                # Price variables
                for j in range(len(self.p)):
                    jacobian.append(0.0)
                # Utility variables
                if self.R_coef is None:
                    # Logit
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            if m != n:
                                expression = 0.0
                            elif i == j:
                                if self.capacity is None:
                                    if self.choice_set[i, n] == 0:
                                        expression = 0.0
                                    else:
                                        sum = np.sum([np.exp(x[self.U[k, m]]) for k in range(len(self.U))])
                                        expression = -(np.exp(x[self.U[j, m]])*sum - np.exp(x[self.U[j, m]])**2)/(sum**2)
                                else:
                                    sum = np.sum([np.exp(x[self.U[k, m]])*x[self.y[k, m]] for k in range(len(self.U))])
                                    expression = -(np.exp(x[self.U[j, m]])*x[self.y[j, m]]*sum - (np.exp(x[self.U[j, m]])*x[self.y[j, m]])**2)/(sum**2)
                            else:
                                expression = 0.0
                                if self.capacity is None:
                                    if self.choice_set[i, n] == 0:
                                        expression = 0.0
                                    else:
                                        sum = np.sum([np.exp(x[self.U[k, m]]) for k in range(len(self.U))])
                                        expression = (np.exp(x[self.U[i, m]])*np.exp(x[self.U[j, m]]))/(sum*sum)
                                else:
                                    sum = np.sum([np.exp(x[self.U[k, m]])*x[self.y[k, m]] for k in range(len(self.U))])
                                    expression = (np.exp(x[self.U[i, m]])*np.exp(x[self.U[j, m]])*x[self.y[i, m]]*x[self.y[j, m]])/(sum**2)
                            jacobian.append(expression)
                else:
                    # Mixed logit
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            for r in range(len(self.U[j, m])):
                                if m != n:
                                    expression = 0.0
                                elif i == j:
                                    expression = 0.0
                                    if self.capacity is None:
                                        if self.choice_set[i, n] == 0:
                                            expression = 0.0
                                        else:
                                            for s in range(len(self.U[j, m])):
                                                sum = np.sum([np.exp(x[self.U[k, m, s]]) for k in range(len(self.U))])
                                                expression += -(np.exp(x[self.U[j, m, s]])*sum - np.exp(x[self.U[j, m, s]])**2)/(sum**2)
                                    else:
                                        for s in range(len(self.U[j, m])):
                                            sum = np.sum([np.exp(x[self.U[k, m, s]])*x[self.y[k, m]] for k in range(len(self.U))])
                                            expression += -(np.exp(x[self.U[j, m, s]])*x[self.y[j, m]]*sum - (np.exp(x[self.U[j, m, s]])*x[self.y[j, m]])**2)/(sum**2)
                                    expression = expression/float(self.R_coef)
                                else:
                                    expression = 0.0
                                    if self.capacity is None:
                                        if self.choice_set[i, n] == 0:
                                            expression = 0.0
                                        else:
                                            for s in range(len(self.U[j, m])):
                                                sum = np.sum([np.exp(x[self.U[k, m, s]]) for k in range(len(self.U))])
                                                expression += (np.exp(x[self.U[i, m, s]])*np.exp(x[self.U[j, m, s]]))/(sum**2)
                                    else:
                                        for s in range(len(self.U[j, m])):
                                            sum = np.sum([np.exp(x[self.U[k, m, s]])*x[self.y[k, m]] for k in range(len(self.U))])
                                            expression += (np.exp(x[self.U[i, m, s]])*np.exp(x[self.U[j, m, s]])*x[self.y[i, m]]*x[self.y[j, m]])/(sum**2)
                                    expression = expression/float(self.R_coef)
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
                    if self.R_coef is None:
                        # Logit
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                if (j == i) and (n == m):
                                    sum = np.sum([np.exp(x[self.U[k, m]])*x[self.y[k, m]] for k in range(len(self.y))])
                                    expression = -(np.exp(x[self.U[i, n]])*sum - x[self.y[i, n]]*(np.exp(x[self.U[i, n]])**2))/(sum**2)
                                elif (n == m):
                                    sum = np.sum([np.exp(x[self.U[k, m]])*x[self.y[k, m]] for k in range(len(self.y))])
                                    expression = np.exp(x[self.U[i, n]])*x[self.y[i, n]]*np.exp(x[self.U[j, n]])/(sum**2)
                                else:
                                    expression = 0.0
                                jacobian.append(expression)
                    else:
                        # Mixed logit
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                if (j == i) and (n == m):
                                    for s in range(len(self.U[j, m])):
                                        sum = np.sum([np.exp(x[self.U[k, m, s]])*x[self.y[k, m]] for k in range(len(self.y))])
                                        expression += -(np.exp(x[self.U[i, n, s]])*sum - x[self.y[i, n]]*(np.exp(x[self.U[i, n, s]])**2))/(sum**2)
                                elif (n == m):
                                    for s in range(len(self.U[j, m])):
                                        sum = np.sum([np.exp(x[self.U[k, m, s]])*x[self.y[k, m]] for k in range(len(self.y))])
                                        expression += np.exp(x[self.U[i, n, s]])*x[self.y[i, n]]*np.exp(x[self.U[j, n, s]])/(sum**2)
                                else:
                                    expression = 0.0
                                expression = expression/float(self.R_coef)
                                jacobian.append(expression)

        #### Utility value constraints
        if self.R_coef is None:
            # Logit
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
                            if (i == j) and (n == m):
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
        else:
            # Mixed Logit
            # For each constraint
            for i in range(len(self.U)):
                for n in range(len(self.U[i])):
                    for r in range(len(self.U[i, n])):
                        # For each variable
                        # Price variables
                        for j in range(len(self.p)):
                            if j == i:
                                jacobian.append(-self.endo_coef[i, n, r])
                            else:
                                jacobian.append(0.0)
                        # Utility variables
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                for s in range(len(self.U[j, m])):
                                    if (i == j) and (n == m) and (r == s):
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
                    if self.R_coef is None:
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                jacobian.append(0.0)
                    else:
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                for r in range(len(self.U[j, m])):
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
                    if self.R_coef is None:
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                jacobian.append(0.0)
                    else:
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                for r in range(len(self.U[j, m])):
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
            #### Alternative availability is subject to the choice-set
            for i in range(self.I_opt_out):
                for n in range(self.N):
                    if self.choice_set[i, n] == 0:
                        # For each variable
                        # Price variables
                        for j in range(len(self.p)):
                            jacobian.append(0.0)
                        # Utility variables
                        if self.R_coef is None:
                            for j in range(len(self.U)):
                                for m in range(len(self.U[j])):
                                    jacobian.append(0.0)
                        else:
                            for j in range(len(self.U)):
                                for m in range(len(self.U[j])):
                                    for r in range(len(self.U[j, m])):
                                        jacobian.append(0.0)
                        # Choice variables
                        for j in range(len(self.w)):
                            for m in range(len(self.w[j])):
                                jacobian.append(0.0)
                        # Availability variables
                        for j in range(len(self.y)):
                            for m in range(len(self.y[j])):
                                if (j == i) and (n == m):
                                    jacobian.append(1.0)
                                else:
                                    jacobian.append(0.0)
            #### capacity is not exceeded
            for i in range(self.I + self.I_opt_out):
                # For each variable
                # Price variables
                for j in range(len(self.p)):
                    jacobian.append(0.0)
                # Utility variables
                if self.R_coef is None:
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            jacobian.append(0.0)
                else:
                    for j in range(len(self.U)):
                        for m in range(len(self.U[j])):
                            for r in range(len(self.U[j, m])):
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
            #### Priority list, if y is 0 then the max capacity is reached, or the alternative is not available in the choice set
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    # For each variable
                    # Price variables
                    for j in range(len(self.p)):
                        jacobian.append(0.0)
                    # Utility variables
                    if self.R_coef is None:
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                jacobian.append(0.0)
                    else:
                        for j in range(len(self.U)):
                            for m in range(len(self.U[j])):
                                for r in range(len(self.U[j, m])):
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
                                jacobian.append(-self.capacity[i]*self.choice_set[i, n])
                            else:
                                jacobian.append(0.0)
            #### Priority list, if y[i, n] is 1 then there is free room for n to choose alternative i
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    # This type of constraint is revelant only if the capacity can be exceeded
                    if self.priority_list[i, n] > self.capacity[i]:
                        # For each variable
                        # Price variables
                        for j in range(len(self.p)):
                            jacobian.append(0.0)
                        # Utility variables
                        if self.R_coef is None:
                            for j in range(len(self.U)):
                                for m in range(len(self.U[j])):
                                    jacobian.append(0.0)
                        else:
                            for j in range(len(self.U)):
                                for m in range(len(self.U[j])):
                                    for r in range(len(self.U[j, m])):
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

def main(data, tol=1e-3):
    ''' Define the problem.
        Args:
            data     data for the MINLP Stackelberg formulation [dict]
            tol      tolerance used for IPOPT solver [float]
    '''

    ### Define the lower and upper bounds on the decision variables
    lb = []
    ub = []
    # Price variables
    for i in range(data['I'] + data['I_opt_out']):
        lb.append(data['lb_p'][i])
        ub.append(data['ub_p'][i])
    # Utility variables
    if 'R_coef' in data.keys():
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                for r in range(data['R_coef']):
                    lb.append(data['lb_U'][i, n, r])
                    ub.append(data['ub_U'][i, n, r])
    else:
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                lb.append(data['lb_U'][i, n])
                ub.append(data['ub_U'][i, n])
    # Choice variables
    for i in range(data['I'] + data['I_opt_out']):
        for n in range(data['N']):
            lb.append(0.0)
            ub.append(1.0)
    # Capacity variables
    if 'capacity' in data.keys():
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                lb.append(0.0)
                ub.append(1.0)

    # x0 is the starting point of the interior point method
    # Start from the previous solution if given
    if 'x0' in data.keys():
        # Pass the previous solution to the getInitial point method to polish it
        # It avoids numerical error
        x0 = getInitialPoint(data, previous_solution=data['x0'])
    else:
        x0 = getInitialPoint(data)

    ### Define the lower and upper bound on the constraints
    cl = []
    cu = []
    # Probabilistic choice constraints
    for i in range(data['I'] + data['I_opt_out']):
        for n in range(data['N']):
            cl.append(-tol)
            cu.append(tol)
    # Utility value constraints
    if 'R_coef' in data.keys():
        # Mixed logit
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                for r in range(data['R_coef']):
                    cl.append(-tol)
                    cu.append(tol)
    else:
        # Logit
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                cl.append(-tol)
                cu.append(tol)
    # Fixed prices constraints
    if 'optimizer' in data.keys():
        for i in range(data['I'] + data['I_opt_out']):
            if data['operator'][i] != data['optimizer']:
                cl.append(-tol)
                cu.append(tol)
    # Capacity constraints
    if 'capacity' in data.keys():
        # Ensure that y is a binary variable
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                cl.append(-tol)
                cu.append(tol)
        # Alternative availability is subject to the choice-set
        for i in range(data['I_opt_out']):
            for n in range(data['N']):
                if data['choice_set'][i, n] == 0:
                    cl.append(0.0 - tol)
                    cu.append(0.0 + tol)
        # Capacity is not exceeded
        for i in range(data['I'] + data['I_opt_out']):
            cl.append(-data['capacity'][i] - tol)
            cu.append(tol)
        # Priority list, if y is 0 then the max capacity is reached
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                cl.append(-data['N'] - 1.0 - tol)
                cu.append(1.0 + tol)
        # Priority list, if y is 1 then there is free room
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                # This type of constraint is revelant only if the capacity could be exceeded
                if data['priority_list'][i, n] > data['capacity'][i]:
                    cl.append(-data['N'] - 1.0 - tol)
                    cu.append(tol)

    # Initialize IPOPT problem
    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=Stackelberg(**data),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    # Set the parameters of IPOPT solver
    # Documentation about the IPOPT parameters:
    # https://www.coin-or.org/Ipopt/documentation/node40.html

    # Do not print the solver's output
    nlp.addOption('print_level', 0)
    # Maximum number of iterations
    nlp.addOption('max_iter', 1500)
    # Set up a warm start to a x0
    # Explanation:
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.589.5002&rep=rep1&type=pdf
    nlp.addOption('warm_start_init_point', 'yes')
    nlp.addOption('warm_start_bound_push', 1e-19)
    nlp.addOption('warm_start_bound_frac', 1e-19)
    nlp.addOption('warm_start_slack_bound_frac', 1e-19)
    nlp.addOption('warm_start_slack_bound_push', 1e-19)
    nlp.addOption('warm_start_mult_bound_push', 1e-19)
    # Check for invalid derivative information
    #nlp.addOption('check_derivatives_for_naninf', 'yes')
    # Set the acceptable tolerance
    nlp.addOption('acceptable_tol', tol)
    nlp.addOption('tol', tol)
    nlp.addOption('constr_viol_tol', tol)

    ### Solve the IPOPT problem
    print('\n--- Solve the IPOPT problem ----')
    x, info = nlp.solve(x0)

    # Change the sign of the optimal objective function value
    # (conversion of a minimization problem to a maximization)
    info['obj_val'] = -info['obj_val']

    # Print the solution
    printSolution(data, x, info)
    # Get the index of the choice variables
    if 'R_coef' in data.keys():
        choice_start = data['I'] + data['I_opt_out'] + data['N']*(data['I'] + data['I_opt_out'])*data['R_coef']
        choice_end = data['I'] + data['I_opt_out'] + data['N']*(data['I'] + data['I_opt_out'])*data['R_coef'] + data['N']*(data['I'] + data['I_opt_out'])
    else:
        choice_start = data['I'] + data['I_opt_out'] + data['N']*(data['I'] + data['I_opt_out'])
        choice_end = data['I'] + data['I_opt_out'] + 2*data['N']*(data['I'] + data['I_opt_out'])

    # Return the price, choice, whole solution and info variables
    return x[:data['I'] + data['I_opt_out']], x0[choice_start:choice_end], x, info['status'], info['status_msg']

def getInitialPoint(data, previous_solution=None):
    ''' Compute an initial feasible solution to the best response game.
        The initial solution of the interior point algorithm will be set at this
        solution.
        If a previous solution is given, polish it.
        Args:
            data                data for the MINLP Stackelberg formulation [dict]
            previous_solution   solution of the previous iterative of the sequential game [list]
    '''

    print('\n--- Compute a feasible solution x0 ----')
    x0 = []
    # count is used to keep track of the variable index
    count = 0

    # Price variables
    p_index = np.empty([data['I'] + data['I_opt_out']], dtype = int)
    for i in range(data['I'] + data['I_opt_out']):
        if 'operator' in data.keys() and data['operator'][i] != data['optimizer']:
            x0.append(data['p_fixed'][i])
        else:
            if previous_solution is not None:
                x0.append(previous_solution[i])
            else:
                x0.append((data['ub_p'][i]+data['lb_p'][i])/2.0)
        p_index[i] = count
        count += 1

    # Utility variables
    if 'R_coef' in data.keys():
        # Mixed Logit
        u_index = np.empty([data['I'] + data['I_opt_out'], data['N'], data['R_coef']], dtype = int)
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                for r in range(data['R_coef']):
                    x0.append(data['endo_coef'][i, n, r]*x0[p_index[i]] + data['exo_utility'][i, n, r])
                    u_index[i, n, r] = count
                    count += 1
    else:
        # Logit
        u_index = np.empty([data['I'] + data['I_opt_out'], data['N']], dtype = int)
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                x0.append(data['endo_coef'][i, n]*x0[p_index[i]] + data['exo_utility'][i, n])
                u_index[i, n] = count
                count += 1

    # Choice variables and availability variables
    # TODO: Explain how they are computed
    if 'capacity' in data.keys():
        # Capacitated model
        # Initial Choice variables
        w_index = np.empty([data['I'] + data['I_opt_out'], data['N']], dtype = int)
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                x0.append(1.0)
                w_index[i, n] = count
                count += 1
        # Initial Availability variables
        y_index = np.empty([data['I'] + data['I_opt_out'], data['N']], dtype = int)
        for i in range(data['I'] + 1):
            for n in range(data['N']):
                if data['choice_set'] == 0:
                    x0.append(0.0)
                else:
                    x0.append(1.0)
                y_index[i, n] = count
                count += 1

        # Adjust the choice and availability variables until convergence to a feasible solution
        feasible = False
        while feasible is False:
            # Choice variables
            for i in range(data['I'] + data['I_opt_out']):
                for n in range(data['N']):
                    if 'R_coef' in data.keys():
                        # Mixed Logit
                        sum = 0.0
                        for r in range(data['R_coef']):
                            numerator = np.exp(x0[u_index[i, n, r]])*x0[y_index[i, n]]
                            denominator = np.sum([np.exp(x0[u_index[j, n, r]])*x0[y_index[j, n]] for j in range(data['I'] + data['I_opt_out'])])
                            sum += float(numerator)/denominator
                        sum = sum/data['R_coef']
                        x0[w_index[i, n]] = sum
                    else:
                        # Logit
                        denominator = np.sum([np.exp(x0[u_index[j, n]])*x0[y_index[j, n]] for j in range(data['I'] + data['I_opt_out'])])
                        numerator = np.exp(x0[u_index[i, n]])*x0[y_index[i, n]]
                        x0[w_index[i, n]] = float(numerator)/denominator
            # Check if the solution is feasible
            feasible = True
            capa = copy.deepcopy(data['capacity'])
            for i in range(data['I'] + data['I_opt_out']):
                occupancy = 0.0
                for n in range(data['N']):
                    occupancy += x0[w_index[i, n]]
                    if capa[i] >= 1.0:
                        capa[i] -= x0[w_index[i, n]]
                        if (x0[y_index[i, n]] == 0) and (data['choice_set'][i, n] == 1):
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
        # Uncapacitated
        # Choice variables
        w_index = np.empty([data['I'] + data['I_opt_out'], data['N']], dtype = int)
        for i in range(data['I'] + data['I_opt_out']):
            for n in range(data['N']):
                if data['choice_set'][i, n] == 0:
                    x0.append(0.0)
                    w_index[i, n] = count
                    count += 1
                elif 'R_coef' in data.keys():
                    # Mixed Logit
                    sum = 0.0
                    for r in range(data['R_coef']):
                        numerator = np.exp(x0[u_index[i, n, r]])
                        denominator = np.sum([np.exp(x0[u_index[j, n, r]]) for j in range(data['I'] + data['I_opt_out'])])
                        sum += float(numerator)/denominator
                    sum = sum/data['R_coef']
                    x0.append(sum)
                    w_index[i, n] = count
                    count += 1
                else:
                    # Logit
                    numerator = np.exp(x0[u_index[i, n]])
                    denominator = np.sum([np.exp(x0[u_index[j, n]]) for j in range(data['I'] + data['I_opt_out'])])
                    x0.append(float(numerator)/denominator)
                    w_index[i, n] = count
                    count += 1

    return x0

def printSolution(data, x, info):

    print('\nResults:')
    print('Decision variables: \n')
    counter = 0
    # Price variables
    for i in range(data['I'] + data['I_opt_out']):
        print('Price of alternative %r: %r'%(i, x[counter]))
        counter += 1
    print('\n')
    '''
    # Utility variables
    for i in range(data['I'] + data['I_opt_out']):
        for n in range(data['N']):
            print('Utility of alternative %r for user %r : %r'%(i, n, x[counter]))
            counter += 1
    print('\n')
    # Choice variables
    for i in range(data['I'] + data['I_opt_out']):
        for n in range(data['N']):
            print('Choice of alternative %r for user %r : %r'%(i, n, x[counter]))
            counter += 1
    print('\n')
    # Availability variables
    if 'capacity' in data.keys():
        for i in range(data['I'] + data['I_opt_out']):
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
