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
import Data.Fixed_Point.ProbLogit_n10 as data_file

class Fixed_Point:

    def __init__(self, **kwargs):
        ''' Construct a Fixed_point game
            KeywordArgs:
                I               Number of alternatives
                N               Number of customers
                R               Number of draws
                K               Number of operators
                operator        Mapping between alternative and operators
                lb_p            Lower bound on price for each alternatives
                ub_p            Upper bound on price for each alternatives
                n_price_levels  Number of strategy for each operator
                xi              Error term values
                M_rev           Big M value for the revenue
                # Preprocessed
                p               Set of possible prices for each alternative
                lb_U            Lower bound on utility for each alternative and customer
                ub_U            Upper bound on utility for each alternative and customer
                lb_Umin         Lower bound on utility for each customer
                ub_Umax         Upper bound on utility for each customer
                M_U             Big M value for each customer for the utility
                exo_utility     Value of the utility for the exogene variables
                endo_coef       Beta coefficient of the endogene variables
                wAft_precomputed wAft value when is was possible to guess its value
                fixed_cost      Initial cost of an alternative
                customer_cost   Additional cost of an alternative for each addition customer

        '''
        ## TODO: Check correctness of the attributes value
        self.I = kwargs.get('I', 2)
        self.N = kwargs.get('N', 10)
        self.R = kwargs.get('R', 50)
        self.K = kwargs.get('K', 2)
        self.operator = kwargs.get('operator', None)
        self.lb_p = kwargs.get('lb_p', 1.0)
        self.ub_p = kwargs.get('ub_p', 1.0)
        self.n_price_levels = kwargs.get('n_price_levels', 60)
        self.xi = kwargs.get('xi')
        self.M_rev = kwargs.get('M_rev')
        # Preprocessed
        self.p = kwargs.get('p')
        self.lb_U = kwargs.get('lb_U')
        self.ub_U = kwargs.get('ub_U')
        self.lb_Umin = kwargs.get('lb_Umin')
        self.ub_Umax = kwargs.get('ub_Umax')
        self.M_U = kwargs.get('M_U')
        self.exo_utility = kwargs.get('exo_utility')
        self.endo_coef = kwargs.get('endo_coef')
        self.wAft_precomputed = kwargs.get('wAft_precomputed', None)
        # Optional
        self.fixed_cost = kwargs.get('fixed_cost', None)
        self.customer_cost = kwargs.get('customer_cost', None)

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
        # Initial strategy picked by each operator
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                model.variables.add(types = [model.variables.type.binary],
                                    names = ['v[' + str(k) + ']' + '[' + str(l) + ']'])

        ## AFTER
        # Revenue maximal for each operator
        for k in range(1, self.K + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['revenueMaxAft[' + str(k) + ']'])

        # Revenue for each operator for each strategy
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['revenueAft[' + str(k) + ']' + '[' + str(l) + ']'])

        # Best response strategy picked by each operator
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                model.variables.add(types = [model.variables.type.binary],
                                    names = ['vAft[' + str(k) + ']' + '[' + str(l) + ']'])

        ## LEVEL 2 : CUSTOMER LEVEL
        ## BEFORE
        # Customer Utility
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Maximum Discounted Utility
        for n in range(self.N):
            for r in range(self.R):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['Umax[' + str(n) + ']' + '[' + str(r) + ']'])

        # Choice among alternatives
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Linearized price
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        ## AFTER
        # Choice among alternative for each strategy
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            model.variables.add(types = [model.variables.type.binary],
                                                names = ['wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Utility
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            model.variables.add(types = [model.variables.type.continuous],
                                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                                names = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

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
        for i in range(self.I + 1):
            if i > 0:
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
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['demand[' + str(i) + ']'])

        # Demand after
        for i in range(self.I + 1):
            for l in range(self.n_price_levels):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['demandAft[' + str(i) + ']' + '[' + str(l) + ']'])

        # Linearized revenue after
        for k in range(1, self.K + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['linRevenueAft[' + str(k) + ']'])

        ##### Add the constraints #####
        ##### Linearization
        for i in range(1, self.I + 1):
            k = self.operator[i]
            indices = []
            coefs = []
            for l in range(self.n_price_levels):
                indices.append('vAft[' + str(k) + ']' + '[' + str(l) + ']')
                coefs.append(-self.p[i, l])
                indices.append('v[' + str(k) + ']' + '[' + str(l) + ']')
                coefs.append(self.p[i, l])
            indices.append('a[' + str(i) + ']')
            indices.append('b[' + str(i) + ']')
            coefs.append(-1.0)
            coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [0.0])
        for i in range(self.I + 1):
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
        for i in range(self.I + 1):
            k = self.operator[i]
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
                    if i > 0:
                        # Alpha is equal to the price if alternative is chosen
                        # The linearized price is already equal to 0 for the opt-out
                        indices = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                                   'w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [-1.0,
                                 self.ub_p[i]]
                        for l in range(self.n_price_levels):
                            indices.append('v[' + str(k) + ']' + '[' + str(l) + ']')
                            coefs.append(self.p[i, l])
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.ub_p[i]])
                        indices = ['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        for l in range(self.n_price_levels):
                            indices.append('v[' + str(k) + ']' + '[' + str(l) + ']')
                            coefs.append(-self.p[i, l])
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [0.0])

        ### LEVEL 1 : OPERATOR UTILITY MAXIMIZATION

        ## Utility function of the operator (revenue)
        # BEFORE

        # AFTER
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                indices = ['revenueAft[' + str(k) + ']' + '[' + str(l) + ']']
                coefs = [1.0]
                rhs = 0.0
                for i in range(self.I + 1):
                    if self.operator[i] == k:
                        if self.fixed_cost is not None:
                            rhs += -self.fixed_cost[i]
                        for n in range(self.N):
                            for r in range(self.R):
                                indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                if self.customer_cost is not None:
                                    coefs.append((-1.0*self.p[i, l]+self.fixed_cost[i])/self.R)
                                coefs.append(-1.0*self.p[i, l]/self.R)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [rhs])

        ## The operator pick exactly one strategy. The second strategy is the one which give the highest revenue
        # BEFORE
        for k in range(1, self.K + 1):
            indices = []
            coefs = []
            for l in range(self.n_price_levels):
                indices.append('v[' + str(k) + ']' + '[' + str(l) + ']')
                coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [1.0])
        # AFTER (best response)
        for k in range(1, self.K + 1):
            indices = []
            coefs = []
            for l in range(self.n_price_levels):
                indices.append('vAft[' + str(k) + ']' + '[' + str(l) + ']')
                coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [1.0])
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                indices = ['revenueAft[' + str(k) + ']' + '[' + str(l) + ']',
                           'revenueMaxAft[' + str(k) + ']']
                coefs = [1.0,
                         -1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'L',
                                             rhs = [0.0])
                indices = ['revenueMaxAft[' + str(k) + ']',
                           'revenueAft[' + str(k) + ']' + '[' + str(l) + ']',
                           'vAft[' + str(k) + ']' + '[' + str(l) + ']']
                coefs = [1.0,
                         -1.0,
                         self.M_rev]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'L',
                                             rhs = [self.M_rev])

        ### LEVEL 2 : CUSTOMER UTILITY MAXIMIZATION
        ## Choice-availability constraints
        # All customers choose one option
        # BEFORE
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
        # AFTER
        for k in range(1, self.K + 1):
            for n in range(self.N):
                for r in range(self.R):
                    for l in range(self.n_price_levels):
                        # If the preprocessing worked :
                        # Replace the 4 'Final configuration' constraints by the following one
                        if (self.wAft_precomputed is not None) and (self.wAft_precomputed[k, 0, n ,r, l] is not None):
                            print('Precomputed : wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            for i in range(self.I + 1):
                                indices = ['wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'E',
                                                             rhs = [self.wAft_precomputed[k, i, n, r, l]])
                        else:
                            indices = []
                            coefs = []
                            for i in range(self.I + 1):
                                indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                coefs.append(1.0)
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'E',
                                                         rhs = [1.0])

        # Utility function
        # BEFORE
        for i in range(self.I + 1):
            k = self.operator[i]
            for n in range(self.N):
                for r in range(self.R):
                    if i == 0:
                        # Opt-out utility
                        indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [self.exo_utility[i, n] + self.xi[i, n, r]])
                    else:
                        indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        for l in range(self.n_price_levels):
                            indices.append('v[' + str(k) + ']' + '[' + str(l) + ']')
                            coefs.append(-self.p[i, l]*self.endo_coef[i, n])
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [self.exo_utility[i, n] + self.xi[i, n, r]])

        # AFTER
        for i in range(self.I + 1):
            for k in range(1, self.K + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            # If wAft is precomputed, no need to add the constraint
                            if not((self.wAft_precomputed is not None) and (self.wAft_precomputed[k, i, n ,r, l] is not None)):
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

        # Utility-choice constraints
        # The selected alternative is the one with maximum discounted utility
        # BEFORE
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'Umax[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
                    indices = ['Umax[' + str(n) + ']' + '[' + str(r) + ']',
                               'U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                    coefs = [1.0, -1.0, self.M_U[n, r]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.M_U[n, r]])

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.n_price_levels):
                            # If wAft is precomputed, no need to add the constraint
                            if not((self.wAft_precomputed is not None) and (self.wAft_precomputed[k, i, n ,r, l] is not None)):
                                indices = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                           'UmaxAft[' + str(k) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0, -1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'L',
                                                             rhs = [0.0])
                                indices = ['UmaxAft[' + str(k) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                           'UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                           'wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0, -1.0, self.M_U[n, r]]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'L',
                                                             rhs = [self.M_U[n, r]])

        # Auxiliary constraints to calculate the demands (not part of the model)
        # BEFORE
        for i in range(self.I + 1):
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
        for i in range(1, self.I + 1):
            k = self.operator[i]
            for l in range(self.n_price_levels):
                indices = []
                coefs = []
                for n in range(self.N):
                    for r in range(self.R):
                        indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                        coefs.append(-1.0/self.R)
                indices.append('demandAft[' + str(i) + ']' + '[' + str(l) + ']')
                coefs.append(1.0)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'E',
                                             rhs = [0.0])

        # Auxiliary variable: Linearized revenue after
        # linRevenueAft is an upperbound on revenueAft
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                indices = ['linRevenueAft[' + str(k) + ']',
                           'revenueAft[' + str(k) + ']' + '[' + str(l) + ']']
                coefs = [1.0, -1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'G',
                                             rhs = [0.0])

        # RevenueAft is an upperbound on linRevenueAft
        # if the corresponding strategy is played
        for k in range(1, self.K + 1):
            for l in range(self.n_price_levels):
                indices = ['vAft[' + str(k) + ']' + '[' + str(l) + ']',
                           'revenueAft[' + str(k) + ']' + '[' + str(l) + ']',
                           'linRevenueAft[' + str(k) + ']']
                coefs = [-self.M_rev, 1.0, -1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'G',
                                             rhs = [-self.M_rev])

        return model

    def solveModel(self, model):
        ''' Solve the given model, return the solved model.
            Args:
                model          cplex model to solve
            Returns:
                model          cplex model solved
        '''
        #TODO: Add print solution
        try:
            print("--SOLUTION : --")
            #model.set_results_stream(None)
            model.set_warning_stream(None)
            model.solve()
            print('Objective function: %r '%(model.solution.get_objective_value()))
            # Print the previous price
            print()
            for i in range(1, self.I + 1):
                k = self.operator[i]
                for l in range(self.n_price_levels):
                    if model.solution.get_values('v[' + str(k) + ']' + '[' + str(l) + ']') == 1.0:
                        print('Previous price of alt %r : %r' %(i, self.p[i, l]))
            # Print the strategy chosen by the operator
            print()
            for k in range(1, self.K + 1):
                for l in range(self.n_price_levels):
                    if model.solution.get_values('vAft[' + str(k) + ']' + '[' + str(l) + ']') == 1.0:
                        print('Operator %r has chosen the strategy %r with price = %r' %(k, l, self.p[np.where(self.operator == k), l][0][0]))
                        print('The after revenue of operator %r = %r' %(k, model.solution.get_values('revenueAft[' + str(k) + ']' + '[' + str(l) + ']')))
            # Print the demand
            print()
            for i in range(self.I +1):
                for k in range(1, self.K + 1):
                    if self.operator[i] == k:
                        for l in range(self.n_price_levels):
                            if model.solution.get_values('vAft[' + str(k) + ']' + '[' + str(l) + ']') == 1.0:
                                print('The after demand for the alternative %r = %r ' %(i, model.solution.get_values('demandAft[' + str(i) + ']' + '[' + str(l) + ']')))
            print()
            # Print the linearized revenue
            for k in range(1, self.K + 1):
                print('The linearized revenue of operator %r is: %r' %(k, model.solution.get_values('linRevenueAft[' + str(k) + ']')))
            print()
            # Print the distance auxiliary variables
            for i in range(self.I + 1):
                if i > 0:
                    print('A Distance(i = %r): %r' %(i,
                            model.solution.get_values('a[' + str(i) + ']')))
                    print('B Distance(i = %r): %r' %(i,
                            model.solution.get_values('b[' + str(i) + ']')))

            return model
        except CplexSolverError as e:
            print('Exception raised during dual of restricted problem')

if __name__ == '__main__':
    t_0 = time.time()
    # Get the data and preprocess
    dict = data_file.getData()
    data_file.preprocess(dict)
    #data_file.preprocess2(dict)
    t_1 = time.time()
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