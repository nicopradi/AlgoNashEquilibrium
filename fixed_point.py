# Modelisation of the fixed point game with free initial prices and capacity

# General
import sys
import time
# CPLEX
import cplex
from cplex.exceptions import CplexSolverError
# numpy
import numpy as np
#data
import Data.Parking_lot.Fixed_Point.ProbLogit_n10 as data_file

class Fixed_Point:

    def __init__(self, **kwargs):
        ''' Construct a Fixed_point game
            KeywordArgs:
                I               Number of alternatives (without opt-out) [int]
                I_opt_out       Number of opt-out alternatives [int]
                N               Number of customers [int]
                R               Number of draws [int]
                K               Number of operators [int]
                operator        Mapping between alternative and operators [list]
                lb_p            Lower bound on price for each alternatives [list]
                ub_p            Upper bound on price for each alternatives [list]
                n_price_levels  Number of strategy for each operator [int]
                capacity        Maximum capacity for each alternative [list]
                choice_set      Individual choice sets [list]
                xi              Error term values [list]
                M_rev           Big M value for the revenue [int]
                # Preprocessed
                priority_list   Priority list for each alternative [list]
                p               Set of possible prices for each alternative [list]
                lb_U            Lower bound on utility for each alternative and customer [list]
                ub_U            Upper bound on utility for each alternative and customer [list]
                lb_Umin         Lower bound on utility for each customer [list]
                ub_Umax         Upper bound on utility for each customer [list]
                M_U             Big M value for each customer for the utility [list]
                exo_utility     Value of the utility for the exogene variables [list]
                endo_coef        Beta coefficient of the endogene variables [list]

        '''
        ## TODO: Check correctness of the attributes value
        self.I = kwargs.get('I', 2)
        self.I_opt_out = kwargs.get('I_opt_out', 1)
        self.N = kwargs.get('N', 10)
        self.R = kwargs.get('R', 50)
        self.K = kwargs.get('K', 2)
        self.operator = kwargs.get('operator', None)
        self.lb_p = kwargs.get('lb_p', 1.0)
        self.ub_p = kwargs.get('ub_p', 1.0)
        self.n_price_levels = kwargs.get('n_price_levels', 60)
        self.capacity = kwargs.get('capacity')
        self.choice_set = kwargs.get('choice_set', None)
        self.xi = kwargs.get('xi')
        self.M_rev = kwargs.get('M_rev')
        # Preprocessed
        self.priority_list = kwargs.get('priority_list')
        self.p = kwargs.get('p')
        self.lb_U = kwargs.get('lb_U')
        self.ub_U = kwargs.get('ub_U')
        self.lb_Umin = kwargs.get('lb_Umin')
        self.ub_Umax = kwargs.get('ub_Umax')
        self.M_U = kwargs.get('M_U')
        self.exo_utility = kwargs.get('exo_utility')
        self.endo_coef = kwargs.get('endo_coef')
        # Attributes
        self.reverse_operator = {}
        for k in range(self.K + 1):
            self.reverse_operator[k] = [i for i, ope in enumerate(self.operator) if ope == k]

    def getModel(self):
        ''' Construct a CPLEX model corresponding to a Stackelberg game (1 leader,
        1 follower).
            Returns:
                model          CPLEX model
        '''

        model = cplex.Cplex() # Initialize the model
        model.objective.set_sense(model.objective.sense.minimize) ## Set the objective function to maximization

        ##### DECISION VARIABLES #####
        ## LEVEL 1 : OPERATOR LEVEL

        ## BEFORE
        # Revenue for each operator
        for k in range(1, self.K + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['revenue[' + str(k) + ']'])

        # Price set for the alternatives
        for i in range(self.I + self.I_opt_out):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['price[' + str(i) + ']'])

        # Availability at operator level
        for i in range(self.I + self.I_opt_out):
            model.variables.add(types = [model.variables.type.binary],
                                names = ['y[' + str(i) + ']'])

        ## AFTER
        # Revenue maximal for each operator
        for k in range(1, self.K + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['revenueAftMax[' + str(k) + ']'])

        # Revenue for each operator for each strategy
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for l in range(self.n_price_levels):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['revenueAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']'])

        # Availability at operator level
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                model.variables.add(types = [model.variables.type.binary],
                                    names = ['yAft[' + str(k) + ']' + '[' + str(i) + ']'])

        # Strategy picked by each operator
        # Strategy picked by each operator
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for l in range(self.n_price_levels):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['vAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']'])

        ## LEVEL 2 : CUSTOMER LEVEL
        ## BEFORE
        # Availability at scenario level, as result of the choices of other customers
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Customer Utility
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Discounted Utility based on choice available
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Maximum Discounted Utility
        for n in range(self.N):
            for r in range(self.R):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['Umax[' + str(n) + ']' + '[' + str(r) + ']'])

        # Choice among alternatives
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Linearized price
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        ## AFTER
        # Availability of the alternative for each customer, draw and strategy
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            model.variables.add(types = [model.variables.type.binary],
                                                names = ['y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Choice among alternative for each strategy
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            model.variables.add(types = [model.variables.type.binary],
                                                names = ['wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Utility
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            model.variables.add(types = [model.variables.type.continuous],
                                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                                names = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Discounted Utility
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            model.variables.add(types = [model.variables.type.continuous],
                                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                                names = ['zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Maximum Discounted Utility
        for k in range(1, self.K + 1):
            for n in range(self.N):
                for r in range(self.R):
                    for l in range(self.n_price_levels):
                        model.variables.add(types = [model.variables.type.continuous],
                                            lb = [-cplex.infinity], ub = [cplex.infinity],
                                            names = ['UmaxAft[' + str(k) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        ## AUXILIARY VARIABLES
        # Distance between the previous price and next price
        for i in range(self.I + self.I_opt_out):
            if i >= self.I_opt_out:
                model.variables.add(obj = [1.0],
                                    types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['a[' + str(i) + ']'])
                model.variables.add(obj = [1.0],
                                    types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['b[' + str(i) + ']'])
            else:
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['a[' + str(i) + ']'])
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['b[' + str(i) + ']'])

        # Demand
        for i in range(self.I + self.I_opt_out):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['demand[' + str(i) + ']'])

        # Demand after
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for l in range(self.n_price_levels):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['demandAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']'])

        ##### Add the constraints #####
        ##### Linearization
        for i in range(self.I_opt_out, self.I + self.I_opt_out):
            k = self.operator[i]
            indices = ['price[' + str(i) + ']']
            coefs = [1.0]
            for l in range(self.n_price_levels):
                indices.append('vAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']')
                coefs.append(-self.p[i, l])
            indices.append('a[' + str(i) + ']')
            indices.append('b[' + str(i) + ']')
            coefs.append(-1.0)
            coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [0.0])

        for i in range(self.I + self.I_opt_out):
            indices = ['a[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'G',
                                         rhs = [0.0])
            indices = ['b[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'G',
                                         rhs = [0.0])

        ##### Price choice - Customer choices
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    # Alpha is 0 is alternative is not chosen
                    indices = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [self.lb_p[i],
                             -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
                    indices = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0,
                             -self.ub_p[i]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
                    # Alpha is equal to the price if alternative is chosen
                    indices = ['price[' + str(i) + ']',
                               'alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0,
                             -1.0,
                             self.ub_p[i]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.ub_p[i]])
                    indices = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'price[' + str(i) + ']']
                    coefs = [1.0,
                             -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        ### LEVEL 1 : OPERATOR UTILITY MAXIMIZATION
        ## Price Bound
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            indices = ['price[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'G',
                                         rhs = [self.lb_p[i]])
            indices = ['price[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'L',
                                         rhs = [self.ub_p[i]])

        ## Utility function of the operator (revenue)
        # BEFORE
        for k in range(1, self.K + 1):
            indices = ['revenue[' + str(k) + ']']
            coefs = [1.0]
            for i in range(self.I + self.I_opt_out):
                if self.operator[i] == k:
                    for n in range(self.N):
                        for r in range(self.R):
                            indices.append('alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                            coefs.append(-1.0/self.R)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [0.0])

        # AFTER
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                indices = ['revenueAft[' + str(k) + ']' + '[' + str(l) + ']']
                coefs = [1.0]
                for i in range(self.I_opt_out, self.I + self.I_opt_out):
                    if self.operator[i] == k:
                        for n in range(self.N):
                            for r in range(self.R):
                                indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                coefs.append(-1.0*self.p[i, l]/self.R)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [0.0])

        ## The selected strategy is the one giving the highest revenue
        # AFTER (best response)
        for k in range(1, self.K + 1):
            for i in self.reverse_operator[k]:
                indices = []
                coefs = []
                for l in range(self.n_price_levels):
                    indices.append('vAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']')
                    coefs.append(1.0)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [1.0])
        for k in range(1, self.K + 1):
            # Set of all the possible strategies for the operator k
            strategy = [list(range(self.n_price_levels)) for i in range(len(self.reverse_operator[k]))]
            strategies = itertools.product(*strategy)
            # Mapping between alternative index in I and alternative index in strategies
            map_index = {}
            index = 0
            for i in self.reverse_operator[k]:
                map_index[i] = index
                index += 1
            # For each strategy of the operator k, add the revenue maximization constraints
            for s in strategies:
                # Lower bound on revenueAftMax
                indices = ['revenueAftMax[' + str(k) + ']']
                coefs = [-1.0]
                for i in self.reverse_operator[k]:
                    indices.append('revenueAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(s[map_index[i]]) + ']')
                    coefs.append(1.0)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'L',
                                             rhs = [0.0])
                # Upper bound on revenueAftMax if s is chosen
                indices = ['revenueAftMax[' + str(k) + ']']
                coefs = [1.0]
                for i in self.reverse_operator[k]:
                    indices.append('revenueAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(s[map_index[i]]) + ']')
                    coefs.append(-1.0)
                    indices.append('vAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(s[map_index[i]]) + ']')
                    coefs.append(self.M_rev/len(self.reverse_operator[k]))

                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'L',
                                             rhs = [len(self.reverse_operator[k])*self.M_rev])

        ### LEVEL 2 : CUSTOMER UTILITY MAXIMIZATION
        ## Choice-availability constraints
        # All customers choose one option
        # BEFORE
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
        # AFTER
        for k in range(1, self.K + 1):
            for n in range(self.N):
                for r in range(self.R):
                    for l in range(self.n_price_levels):
                        indices = []
                        coefs = []
                        for i in range(self.I + self.I_opt_out):
                            indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(1.0)
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [1.0])

        # Customers can only choose options that are available at scenario level
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0,
                             -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            indices = ['wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0,
                                     -1.0]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])

        # Availability at scenario level subject to availability at operator level
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y[' + str(i) + ']']
                    coefs = [1.0,
                             -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
        for i in range(self.I + self.I_opt_out):
            indices = ['y[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [1.0])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            indices = ['y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'yAft[' + str(k) + ']' + '[' + str(i) + ']']
                            coefs = [1.0,
                                     -1.0]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])
                indices = ['yAft[' + str(k) + ']' + '[' + str(i) + ']']
                coefs = [1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [1.0])

        # Opt-out is always an available option (if available in the choice set)
        # TODO: Remove this constraint ?
        if np.count_nonzero(self.choice_set[:self.I_opt_out, :] == 1) == np.prod(self.choice_set[:self.I_opt_out, :].shape):
            # BEFORE
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['y_scen[' + str(0) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [1.0])
            # AFTER
            for k in range(1, self.K + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            indices = ['y_scenAft[' + str(k) + ']' + '[' + str(0) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'E',
                                                         rhs = [1.0])

        # Alternatives not available at operator level if not included in the customer's choice set
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                if self.choice_set[i, n] == 0:
                    for r in range(self.R):
                        indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [0.0])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    if self.choice_set[i, n] == 0:
                        for r in range(self.R):
                            for l in range(self.n_price_levels):
                                indices = ['y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'E',
                                                             rhs = [0.0])

        # Capacity constraints
        # BEFORE
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
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for r in range(self.R):
                    for l in range(self.n_price_levels):
                        indices = []
                        coefs = []
                        for n in range(self.N):
                            indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(1.0)
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.capacity[i]])

        # Priority list
        # BEFORE
        # If alternative not available at scenario level,
        # then the capacity is reached, or the alternative is not available in the choice set
        for i in range(self.I_opt_out, self.I + self.I_opt_out): # Do not consider opt-out
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
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I_opt_out, self.I + self.I_opt_out): # Do not consider opt-out
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            indices = []
                            coefs = []
                            # Sum of the customers which have priority
                            for m in range(self.N):
                                if self.priority_list[i, m] < self.priority_list[i, n]:
                                    indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                    coefs.append(-1.0)
                            indices.append('yAft[' + str(k) + ']' + '[' + str(i) + ']')
                            coefs.append(self.capacity[i]*self.choice_set[i, n])
                            indices.append('y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(-self.capacity[i]*self.choice_set[i, n])
                            # Add the constraint
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])

        # BEFORE
        # Priority list: if alternative is available at scenario level,
        # then there is still some free capacity for the customer
        for i in range(self.I_opt_out, self.I + self.I_opt_out): # Do not consider opt-out
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

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I_opt_out, self.I + self.I_opt_out): # Do not consider opt-out
                for r in range(self.R):
                    for l in range(self.n_price_levels):
                        for n in range(self.N):
                            if (self.priority_list[i, n] > self.capacity[i]) and \
                               (self.choice_set[i, n] == 1):
                               indices = []
                               coefs = []
                               # Sum of the customers which have priority
                               for m in range(self.N):
                                   if self.priority_list[i, m] < self.priority_list[i, n]:
                                       indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                       coefs.append(1.0)
                               indices.append('y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                               coefs.append(-self.capacity[i] + self.priority_list[i, n])
                               # Add the constraint
                               model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                            senses = 'L',
                                                            rhs = [self.priority_list[i, n] - 1.0])

        # Utility function
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'price[' + str(i) + ']']
                    coefs = [1.0, -self.endo_coef[i, n]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [self.exo_utility[i, n] + self.xi[i, n, r]])

        # AFTER
        for i in range(self.I + self.I_opt_out):
            for k in range(1, self.K + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            if self.operator[i] == k:
                                indices = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'E',
                                                             rhs = [self.exo_utility[i, n] + self.endo_coef[i, n] * self.p[i, l] + self.xi[i, n, r]])
                            else:
                                indices = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                           'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                                coefs = [1.0,
                                         -1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'E',
                                                             rhs = [0.0])

        # Discounted utility
        # BEFORE
        for i in range(self.I + self.I_opt_out):
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
                    coefs = [1.0, -self.M_U[n, r]]
                    # Discounted utility equal to utility lower bound if alternative not available
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.lb_Umin[n, r]])
                    indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, self.M_U[n, r], -1.0]
                    # Discounted utility equal to utility if alternative available
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.M_U[n, r]])
                    indices = ['z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0]
                    # Discounted utility greater than utility lower bound
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            indices = ['zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0]
                            # Discounted utility greater than utility lower bound
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'G',
                                                         rhs = [self.lb_Umin[n, r]])
                            indices = ['zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0, -self.M_U[n, r]]
                            # Discounted utility equal to utility lower bound if alternative not available
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [self.lb_Umin[n, r]])
                            indices = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0, self.M_U[n, r], -1.0]
                            # Discounted utility equal to utility if alternative available
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [self.M_U[n, r]])
                            indices = ['zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0, -1.0]
                            # Discounted utility greater than utility lower bound
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])

        # Utility-choice constraints
        # The selected alternative is the one with maximum discounted utility
        # BEFORE
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
                    coefs = [1.0, -1.0, self.M_U[n, r]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.M_U[n, r]])

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            indices = ['zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'UmaxAft[' + str(k) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0, -1.0]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])
                            indices = ['UmaxAft[' + str(k) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0, -1.0, self.M_U[n, r]]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [self.M_U[n, r]])

        # Auxiliary constraints to calculate the demands (not part of the model)
        # BEFORE
        for i in range(self.I + self.I_opt_out):
            indices = []
            coefs = []
            for n in range(self.N):
                for r in range(self.R):
                    indices.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    coefs.append(-1.0/self.R)
            indices.append('demand[' + str(i) + ']')
            coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [0.0])

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for l in range(self.n_price_levels):
                    indices = []
                    coefs = []
                    for n in range(self.N):
                        for r in range(self.R):
                            indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(-1.0/self.R)
                    indices.append('demandAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']')
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
        print("--SOLUTION : --")
        #model.set_results_stream(None)
        model.set_warning_stream(None)
        model.solve()
        print(model.solution.get_objective_value())
        for i in range(self.I  + self.I_opt_out):
            print('Previous price of alt %r : %r' %(i, model.solution.get_values('price[' + str(i) + ']')))
            print('Previous demand alt %r : %r' %(i, model.solution.get_values('demand[' + str(i) + ']')))
        print()
        # Print the strategy chosen by the operator
        print()
        for k in range(1, self.K + 1):
            for i in self.reverse_operator[k]:
                for l in range(self.n_price_levels):
                    if model.solution.get_values('vAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']') == 1.0:
                        print('Operator %r has chosen the strategy %r for alternative %r with price = %r' %(k, l, i, self.p[np.where(self.operator == k), l][0][0]))
                        print('The after revenue of operator %r = %r' %(k, model.solution.get_values('revenueAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']')))
        print()
        for k in range(1, self.K + 1):
            for i in range(self.I + self.I_opt_out):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            print('After Discounted Utility (k = %r, i = %r, n = %r, r = %r, l = %r): %r' %(k, i, n ,r ,l ,
                                   model.solution.get_values('zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')))

        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    print('Previous Discounted Utility(i = %r, n = %r, r = %r): %r' %(i, n, r,
                            model.solution.get_values('z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')))

        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    print('Previous Discounted Utility(i = %r, n = %r, r = %r): %r' %(i, n, r,
                            model.solution.get_values('z[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')))

        for i in range(self.I + self.I_opt_out):
            for n in range(self.N):
                for r in range(self.R):
                    print('Previous choice(i = %r, n = %r, r = %r): %r' %(i, n, r,
                            model.solution.get_values('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')))

        for i in range(self.I + self.I_opt_out):
            if i >= self.I_opt_out:
                print('A Distance(i = %r): %r' %(i,
                        model.solution.get_values('a[' + str(i) + ']')))
                print('B Distance(i = %r): %r' %(i,
                        model.solution.get_values('b[' + str(i) + ']')))

        for n in range(self.N):
            for r in range(self.R):
                print('Umax (n = %r, r = %r): %r' %(n, r, model.solution.get_values('Umax[' + str(n) + ']' + '[' + str(r) + ']')))

        return model

if __name__ == '__main__':
    t_0 = time.time()
    # Get the data and preprocess
    dict = data_file.getData()
    data_file.preprocess(dict)
    t_1 = time.time()
    #data_file.preprocess2(dict)
    # Instanciate a Stackelberg game and solve it
    game = Fixed_Point(**dict)
    model = game.getModel()
    print('MODEL COMPUTED')
    t_2 = time.time()
    game.solveModel(model)
    t_3 = time.time()
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Get the model: %r sec' %(t_2 - t_1))
    print('Solve the model: %r sec' %(t_3 - t_2))
    print('--------------')
    print('Total running time: %r sec' %(t_3 - t_0))
