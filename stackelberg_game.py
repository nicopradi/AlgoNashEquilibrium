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
import Data.Italian.Stackelberg.MILPLogit_n40r50 as data_file

class Stackelberg:

    def __init__(self, **kwargs):
        ''' Construct a Stackelberg game
            KeywordArgs:
                I               Number of alternatives (without opt-out) [int]
                I_opt_out       Number of opt-out alternatives [int]
                N               Number of customers [int]
                R               Number of draws [int]
                choice_set      Individual choice sets [list]
                capacity        Maximum capacity for each alternative [list]
                priority_list   Priority list for each alternative [list]
                lb_p            Lower bound on price for each alternatives [list]
                ub_p            Upper bound on price for each alternatives [list]
                lb_Umin         Lower bound on utility for each customer [list]
                ub_Umax         Upper bound on utility for each customer [list]
                M               Big M value for each customer [list]
                exo_utility     Value of the utility for the exogene variables [list]
                endo_coef       Beta coefficient of the endogene variables [list]
                xi              Error term values [list]
                #### Optional kwargs ####
                operator        Index of the operator playing its best reponse strategy [int]
                optimizer       Index of the current operator [list]
                p_fixed         Fixed price of the alternatives managed by other operators [list]
                y_fixed         Fixed availability of the alternatives managed by other operators [list]
                fixed_cost      Initial cost of an alternative [list]
                customer_cost   Additional cost of an alternative for each addition customer [list]

        '''
        # Keyword arguments
        self.I = kwargs.get('I', 2)
        self.I_opt_out = kwargs.get('I_opt_out', 1)
        self.N = kwargs.get('N', 10)
        self.R = kwargs.get('R', 50)
        self.choice_set = kwargs.get('choice_set', None)
        self.capacity = kwargs.get('capacity', None)
        self.priority_list = kwargs.get('priority_list', None)
        self.lb_p = kwargs.get('lb_p', np.zeros(self.I + self.I_opt_out))
        self.ub_p = kwargs.get('ub_p', np.zeros(self.I + self.I_opt_out))
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
        self.fixed_cost = kwargs.get('fixed_cost', None)
        self.customer_cost = kwargs.get('customer_cost', None)

    def getModel(self):
        ''' Construct a CPLEX model corresponding the a Stackelberg game (1 leader,
        1 follower).
            Returns:
                model          CPLEX model
        '''
        # Initialize the model
        model = cplex.Cplex()
        # Set the objective function sense
        model.objective.set_sense(model.objective.sense.maximize)
        # Add the fixed cost to the objective function
        if self.fixed_cost is not None:
            initial_cost = 0.0
            for i in range(self.I_opt_out, self.I + self.I_opt_out):
                if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                    # Alternative i is managed by the optimizer
                    initial_cost += self.fixed_cost[i]
            model.objective.set_offset(-initial_cost)

        ##### Add the decision variables #####
        # Availability at scenario level variables
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Availability at operator level variables
        for i in range(self.I + self.I_opt_out):
            model.variables.add(types = [model.variables.type.binary],
                                names = ['y[' + str(i) + ']'])

        # Customer choice variables
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Utility variables
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Discounted utility
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Maximum utility for each customer and draw
        for n in range(self.N):
            for r in range(self.R):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['Umax[' + str(n) + ']' + '[' + str(r) + ']'])

        # Price variables
        for i in range(self.I + self.I_opt_out):
            model.variables.add(types = [model.variables.type.continuous],
                               lb = [-cplex.infinity], ub = [cplex.infinity],
                               names = ['p[' + str(i) + ']'])

        # Linearized choice-price variables
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    if (i >= self.I_opt_out) and ((self.optimizer is None) or (self.operator[i] == self.optimizer)):
                        model.variables.add(obj = [1.0/self.R], types = [model.variables.type.continuous],
                                            lb = [-cplex.infinity], ub = [cplex.infinity],
                                            names = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    else:
                        model.variables.add(types = [model.variables.type.continuous],
                                            lb = [-cplex.infinity], ub = [cplex.infinity],
                                            names = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Auxiliary variable to calculate the demand
        for i in range(self.I + self.I_opt_out):
            if self.customer_cost is not None:
                # Add customer cost in the objective function
                if (self.optimizer is None) or (self.operator[i] == self.optimizer):
                    model.variables.add(obj = [-self.customer_cost[i]],
                                       types = [model.variables.type.continuous],
                                       lb = [-cplex.infinity], ub = [cplex.infinity],
                                       names = ['d[' + str(i) + ']'])
                else:
                    model.variables.add(types = [model.variables.type.continuous],
                                       lb = [-cplex.infinity], ub = [cplex.infinity],
                                       names = ['d[' + str(i) + ']'])
            else:
                model.variables.add(types = [model.variables.type.continuous],
                                   lb = [-cplex.infinity], ub = [cplex.infinity],
                                   names = ['d[' + str(i) + ']'])

        ##### Add the constraints #####

        ##### Fixed price and alternatives availability constraints
        # The price/availability of the alternatives not managed by the current optimizer are fixed
        if self.p_fixed is not None:
            for i in range(self.I + self.I_opt_out):
                if (i >= self.I_opt_out) and (self.operator[i] != self.optimizer):
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

        ##### Choice and availabilty constraints
        # Each customer choose one alternative
        for n in range(self.N):
            for r in range(self.R):
                indices = []
                coefs = []
                for i in range(self.I + self.I_opt_out):
                    indices.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    coefs.append(1.0)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [1.0])

        # Customer can only choose an option that is available at scenario level
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        # Availability at scenario level is subject to availability at operator level
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y[' + str(i) + ']']
                    coefs = [1.0, -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        # Opt out is always available at scenario level (if available in the choice set)
        # TODO: Remove this constraint ?
        if np.count_nonzero(self.choice_set[:self.I_opt_out, :] == 1) == np.prod(self.choice_set[:self.I_opt_out, :].shape):
            for n in range(self.N):
                for r in range(self.R):
                    for i in range(self.I_opt_out):
                        indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [1.0])

        # Alternative not available at scenerio level if not included in the choice_set
        if self.choice_set is not None:
            for i in range(self.I + self.I_opt_out):
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
            # Demand does not exceed capacity
            for i in range(self.I + self.I_opt_out):
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
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        indices = []
                        coefs = []
                        # Sum of the customers choice which priority over n
                        for m in range(self.N):
                            if self.priority_list[i, m] < self.priority_list[i, n]:
                                indices.append('w[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']')
                                coefs.append(-1.0)
                        indices.append('y[' + str(i) + ']')
                        coefs.append(self.capacity[i]*self.choice_set[i, n])
                        indices.append('y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        coefs.append(-self.capacity[i]*self.choice_set[i, n])
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [0.0])

            # Priority list: if alternative is available at scenario level,
            # then there is still some free capacity for the customer
            for i in range(self.I + self.I_opt_out):
                for r in range(self.R):
                    for n in range(self.N):
                        if (self.priority_list[i, n] > self.capacity[i]) and \
                           (self.choice_set[i, n] == 1):
                           indices = []
                           coefs = []
                           # Sum of the customers choice which have priority over n
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

        #### Price constraints
        # Bound on the price for each alternatives
        for i in range(self.I + self.I_opt_out):
            indices = ['p[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'G',
                                         rhs = [self.lb_p[i]])
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'L',
                                         rhs = [self.ub_p[i]])

        # Linearized price: alpha is equal to 0 if alternative is not choosen
        for i in range(self.I + self.I_opt_out):
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

        # Linearized price: alpha is equal to the price if alternative is chosen
        for i in range(self.I + self.I_opt_out):
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

        #### Utility constraints
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'p[' + str(i) + ']']
                    if len(self.endo_coef.shape) == 3:
                        # Mixed logit
                        coefs = [1.0, -self.endo_coef[i, n, r]]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [self.exo_utility[i, n, r] + self.xi[i, n, r]])
                    else:
                        # Logit
                        coefs = [1.0, -self.endo_coef[i, n]]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [self.exo_utility[i, n] + self.xi[i, n, r]])

        #### Discounted utility constraints
        if self.capacity is not None:
            # Capacitated model
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        # Discounted utility greater than utility lower bound
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'G',
                                                     rhs = [self.lb_Umin[n, r]])
                        # Discounted utility equal to utility lower bound if alternative not available
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, -self.M[n, r]]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.lb_Umin[n, r]])
                        # Discounted utility equal to utility if alternative available
                        indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, self.M[n, r], -1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.M[n, r]])
                        # Discounted utility smaller than utility
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, -1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [0.0])
        else:
            # Uncapacitated model
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0, -1.0]
                        # Discounted utility equal to utility
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [0.0])

        #### Utility maximization constraints
        # The selected alternative is the one with the highest utility
        for i in range(self.I + self.I_opt_out):
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

        #### Auxiliary constraints to compute the demand
        for i in range(self.I + self.I_opt_out):
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
            # Do not print the solver output
            #model.set_results_stream(None)
            #model.set_warning_stream(None)
            model.solve()
            print('Objective function value (benefit): %r' %model.solution.get_objective_value())
            for i in range(self.I  + self.I_opt_out):
                print('Price of alt %r : %r' %(i, model.solution.get_values('p[' + str(i) + ']')))
                print('Demand of alt %r : %r' %(i, model.solution.get_values('d[' + str(i) + ']')))
            return model
        except CplexSolverError as e:
            raise Exception('Exception raised during dual of restricted problem')

if __name__ == '__main__':
    # Get the data and preprocess
    dict = data_file.getData()
    data_file.preprocess(dict)
    # Instanciate a Stackelberg game and solve it
    game = Stackelberg(**dict)
    model = game.getModel()
    game.solveModel(model)
