# Modelisation of the Stackelberg game with continuous price and capacity constraints

# General
import sys
import time
# CPLEX
import cplex
from cplex.exceptions import CplexSolverError
# numpy
import numpy as np
# data
import Data.Stackelberg.MILPLogit_n10r050 as data_file

class Stackelberg:

    def __init__(self, **kwargs):
        ''' Construct a Stackelberg game
            KeywordArgs:
                I               Number of alternatives
                N               Number of customers
                R               Number of draws
                choice_set       Individual choice sets
                capacity        Maximum capacity for each alternative
                priority_list    Priority list for each alternative
                lb_p            Lower bound on price for each alternatives
                ub_p            Upper bound on price for each alternatives
                lb_U            Lower bound on utility for each alternative and customer
                ub_U            Upper bound on utility for each alternative and customer
                lb_Umin         Lower bound on utility for each customer
                ub_Umax         Upper bound on utility for each customer
                M               Big M value for each customer
                exo_utility      Value of the utility for the exogene variables
                endo_coef        Beta coefficient of the endogene variables
                xi              Error term values
                #### Optional kwargs ####
                operator        Mapping between alternative and operators
                optimizer       Index of the current operator
                p_fixed         Fixed price of the alternatives managed by other operators
                y_fixed         Fixed availability of the alternatives managed by other operators

        '''
        ## TODO: Add kwargs for capacity/no capacity, continuous price/discrete price
        ## TODO: Check correctness of the attributes value
        self.I = kwargs.get('I', 2)
        self.N = kwargs.get('N', 10)
        self.R = kwargs.get('R', 50)
        self.choice_set = kwargs.get('choice_set', None)
        self.capacity = kwargs.get('capacity', None)
        self.priority_list = kwargs.get('priority_list', None)
        self.lb_p = kwargs.get('lb_p', np.zeros(self.I + 1))
        self.ub_p = kwargs.get('ub_p', np.zeros(self.I + 1))
        self.lb_U = kwargs.get('lb_U')
        self.ub_U = kwargs.get('ub_U')
        self.lb_Umin = kwargs.get('lb_Umin')
        self.ub_Umax = kwargs.get('ub_Umax')
        self.M = kwargs.get('M')
        self.exo_utility = kwargs.get('exo_utility')
        self.endo_coef = kwargs.get('endo_coef')
        self.xi = kwargs.get('xi')
        # Optinal keyword arguments
        self.operator = kwargs.get('operator', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.p_fixed = kwargs.get('p_fixed', None)
        self.y_fixed = kwargs.get('y_fixed', None)

    def getModel(self):
        ''' Construct a CPLEX model corresponding the a Stackelberg game (1 leader,
        1 follower).
            Returns:
                model          CPLEX model
        '''

        model = cplex.Cplex() # Initialize the model
        model.objective.set_sense(model.objective.sense.maximize) ## Set the objective function to maximization

        ##### Add the decision variables #####

        # Availability at scenario level, as a result of the choices of other customers
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Availability at operator level
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.binary],
                                names = ['y[' + str(i) + ']'])

        # Choice made by the customer
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Utility
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Discounted utility based on choice availabily (y_scen)
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Maximum discounted utility for each customer, each draw
        for n in range(self.N):
            for r in range(self.R):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['Umax[' + str(n) + ']' + '[' + str(r) + ']'])

        # Continuous price
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.continuous],
                               lb = [-cplex.infinity], ub = [cplex.infinity],
                               names = ['p[' + str(i) + ']'])

        # Linearized product choice-price
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    if (i > 0) and ((self.optimizer is None) or (self.operator[i] == self.optimizer)):
                        model.variables.add(obj = [1.0/self.R], types = [model.variables.type.continuous],
                                            lb = [-cplex.infinity], ub = [cplex.infinity],
                                            names = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    else:
                        model.variables.add(types = [model.variables.type.continuous],
                                            lb = [-cplex.infinity], ub = [cplex.infinity],
                                            names = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Auxiliary variable to calculate the demand
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.continuous],
                               lb = [-cplex.infinity], ub = [cplex.infinity],
                               names = ['d[' + str(i) + ']'])

        ##### Add the constraints #####

        ##### Fixed price/alternatives

        # The price/availability of the alternatives not managed by the current optimizer are fixed
        if self.p_fixed is not None:
            for i in range(self.I + 1):
                if (i > 0) and (self.operator[i] != self.optimizer):
                    indices = ['p[' + str(i) + ']']
                    coefs = [1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [self.p_fixed[i]])
                    indices = ['y[' + str(i) + ']']
                    coefs = [1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [self.y_fixed[i]])
        ##### Choice Availability

        # Each customer chooses exactly one alternative:
        for n in range(self.N):
            for r in range(self.R):
                indices = []
                coefs = []
                for i in range(self.I + 1):
                    indices.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    coefs.append(1.0)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [1.0])

        # Customer can only choose options that are available at scenario level
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        # Availability at scenario level is subject to availability at operator level
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y[' + str(i) + ']']
                    coefs = [1.0, -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        # Opt out is always available at scenario level
        for n in range(self.N):
            for r in range(self.R):
                indices = ['y_scen[' + str(0) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                coefs = [1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [1.0])

        # Alternative not available at scenario level if not included in the ChoiceSet
        if self.ChoiceSet is not None:
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        if self.choice_set[i, n] == 0:
                            indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                            coefs = [1.0]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'E',
                                                         rhs = [0.0])

        ##### Capacity constraints
        if self.capacity is not None:
            for i in range(1, self.I + 1): # Do not consider opt-out
                for r in range(self.R):
                    indices = []
                    coefs = []
                    for n in range(self.N):
                        indices.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        coefs.append(1.0)
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.capacity[i]])

            # Priority list: if alternative not available at scenario level,
            # then the capacity is reached, or the alternative is not available in the choice set
            for i in range(1, self.I + 1): # Do not consider opt-out
                for n in range(self.N):
                    for r in range(self.R):
                        indices = []
                        coefs = []
                        # Sum of the customers which have priority
                        for m in range(self.N):
                            if self.priority_list[i, m] < self.priority_list[i, n]:
                                indices.append('w[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']')
                                coefs.append(-1.0)
                        indices.append('y[' + str(i) + ']')
                        coefs.append(self.capacity[i]*self.choice_set[i, n])
                        indices.append('y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        coefs.append(-self.capacity[i]*self.choice_set[i, n])
                        # Add the constraint
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [0.0])

            # Priority list: if alternative is available at scenario level,
            # then there is still some free capacity for the customer
            for i in range(1, self.I + 1): # Do not consider opt-out
                for r in range(self.R):
                    for n in range(self.N):
                        if (self.priority_list[i, n] > self.capacity[i]) and \
                           (self.choice_set[i, n] == 1):
                           indices = []
                           coefs = []
                           # Sum of the customers which have priority
                           for m in range(self.N):
                               if self.priority_list[i, m] < self.priority_list[i, n]:
                                   indices.append('w[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']')
                                   coefs.append(1.0)
                           indices.append('y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                           coefs.append(-self.capacity[i] + self.priority_list[i, n])
                           # Add the constraint
                           model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                        senses = 'L',
                                                        rhs = [self.priority_list[i, n] - 1.0])
        #### Price-choice constraints

        # Bound on price for each alternatives
        for i in range(self.I + 1):
            indices = ['p[' + str(i) + ']']
            coefs = [1.0]
            # Lower bound constraint
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'G',
                                         rhs = [self.lb_p[i]])
            # Upper bound constraint
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'L',
                                         rhs = [self.ub_p[i]])

        # Linearized price-choice: alpha is 0 if alternative is not choosen
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [self.lb_p[i], -1.0]
                    # Lower bound constraint
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
                    indices = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [self.ub_p[i], -1.0]
                    # Upper bound constraint
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'G',
                                                 rhs = [0.0])

        # Linearized price-choice: alpha equals the price if alternative is choosen
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['p[' + str(i) + ']',
                               'w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, self.ub_p[i], -1.0]
                    # alpha is greater than the price for the choosen alternative
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.ub_p[i]])
                    indices = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'p[' + str(i) + ']']
                    coefs = [1.0, -1.0]
                    # alpha is smaller than the price
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        #### Utility function
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'p[' + str(i) + ']']
                    coefs = [1.0, -self.endo_coef[i, n]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [self.exo_utility[i, n] + self.xi[i, n, r]])

        #### Discounted utility function
        if self.Capacity is not None:           # QUESTION: Why do you need the capacity here ?
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        # Discounted utility greater than utility lower bound
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'G',
                                                     rhs = [self.lb_Umin[n, r]])
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, -self.M[n, r]]
                        # Discounted utility equal to utility lower bound if alternative not available
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.lb_Umin[n, r]])
                        indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, self.M[n, r], -1.0]
                        # Discounted utility equal to utility if alternative available
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.M[n, r]])
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, -1.0]
                        # Discounted utility greater than utility lower bound
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [0.0])
        else:
            # Assume y = 1 for each alternative
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, -1.0]
                        # Discounted utility greater than utility lower bound
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [0.0])

        #### Utility-choice constraints
        # The selected alternative is the one with the highest utility
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'Umax[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
                    indices = ['Umax[' + str(n) + ']' + '[' + str(r) + ']',
                               'z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0, self.M[n, r]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.M[n, r]])

        #### Auxiliary constraints to compute the average demands
        for i in range(self.I + 1):
            indices = []
            coefs = []
            for n in range(self.N):
                for r in range(self.R):
                    indices.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    coefs.append(-1.0/self.R)
            indices.append('d[' + str(i) + ']')
            coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [0.0])

        return model

    def solveModel(self, model):
        ''' Solve the given model, return the solved model.
            Args:
                model          cplex model to solve
            Returns:
                model          cplex model solved
        '''
        try:
            print("--SOLUTION : --")
            #model.set_results_stream(None)
            #model.set_warning_stream(None)
            model.solve()
            print(model.solution.get_objective_value())
            for i in range(self.I +1):
                print('Price of alt %r : %r' %(i, model.solution.get_values('p[' + str(i) + ']')))
            return model
        except CplexSolverError as e:
            print('Exception raised during dual of restricted problem')


        solution.get_values([0, 4, 5])

if __name__ == '__main__':
    # Get the data and preprocess
    dict = data_file.getData()
    data_file.preprocess(dict)
    # Instanciate a Stackelberg game and solve it
    game = Stackelberg(**dict)
    model = game.getModel()
    game.solveModel(model)
