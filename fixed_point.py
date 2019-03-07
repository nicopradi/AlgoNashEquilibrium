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
import Data.Fixed_Point.Parking_MLF_NoCap_i2k2n10r50 as data_file

class Fixed_Point:

    def __init__(self, **kwargs):
        ''' Construct a Fixed_point game
            KeywordArgs:
                I               Number of alternatives
                N               Number of customers
                R               Number of draws
                K               Number of operators
                Operator        Mapping between alternative and operators
                lb_p            Lower bound on price for each alternatives
                ub_p            Upper bound on price for each alternatives
                nPriceLevels    Number of strategy for each operator
                Capacity        Maximum capacity for each alternative
                ChoiceSet       Individual choice sets
                xi              Error term values
                M_Rev           Big M value for the revenue
                # Preprocessed
                PriorityList    Priority list for each alternative
                p               Set of possible prices for each alternative
                lb_U            Lower bound on utility for each alternative and customer
                ub_U            Upper bound on utility for each alternative and customer
                lb_Umin         Lower bound on utility for each customer
                ub_Umax         Upper bound on utility for each customer
                M_U             Big M value for each customer for the utility
                ExoUtility      Value of the utility for the exogene variables
                EndoCoef        Beta coefficient of the endogene variables

        '''
        ## TODO: Check correctness of the attributes value
        self.I = kwargs.get('I', 2)
        self.N = kwargs.get('N', 10)
        self.R = kwargs.get('R', 50)
        self.K = kwargs.get('K', 2)
        self.Operator = kwargs.get('Operator', None)
        self.lb_p = kwargs.get('lb_p', 1.0)
        self.ub_p = kwargs.get('ub_p', 1.0)
        self.nPriceLevels = kwargs.get('nPriceLevels', 60)
        self.Capacity = kwargs.get('Capacity')
        self.ChoiceSet = kwargs.get('ChoiceSet')
        self.xi = kwargs.get('xi')
        self.M_Rev = kwargs.get('M_Rev')
        # Preprocessed
        self.PriorityList = kwargs.get('PriorityList')
        self.p = kwargs.get('p')
        self.lb_U = kwargs.get('lb_U')
        self.ub_U = kwargs.get('ub_U')
        self.lb_Umin = kwargs.get('lb_Umin')
        self.ub_Umax = kwargs.get('ub_Umax')
        self.M_U = kwargs.get('M_U')
        self.ExoUtility = kwargs.get('ExoUtility')
        self.EndoCoef = kwargs.get('EndoCoef')

    def getModel(self):
        ''' Construct a CPLEX model corresponding the a Stackelberg game (1 leader,
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
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['price[' + str(i) + ']'])

        # Availability at operator level
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.binary],
                                names = ['y[' + str(i) + ']'])

        ## AFTER
        # Revenue maximal for each operator
        for k in range(1, self.K + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['revenueMaxAft[' + str(k) + ']'])

        # Revenue for each operator for each strategy
        for k in range(1, self.K + 1):
            for l in range(self.nPriceLevels):
                model.variables.add(types = [model.variables.type.continuous],
                                    lb = [-cplex.infinity], ub = [cplex.infinity],
                                    names = ['revenueAft[' + str(k) + ']' + '[' + str(l) + ']'])

        # Availability at operator level
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                model.variables.add(types = [model.variables.type.binary],
                                    names = ['yAft[' + str(k) + ']' + '[' + str(i) + ']'])

        # Strategy picked by each operator
        for k in range(1, self.K + 1):
            for l in range(self.nPriceLevels):
                model.variables.add(types = [model.variables.type.binary],
                                    names = ['vAft[' + str(k) + ']' + '[' + str(l) + ']'])

        ## LEVEL 2 : CUSTOMER LEVEL

        ## BEFORE

        # Availability at scenario level, as result of the choices of other customers
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.binary],
                                        names = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Customer Utility
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])

        # Discounted Utility based on choice available
        for i in range(self.I + 1):
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
        # Availability of the alternative for each customer, draw and strategy
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            model.variables.add(types = [model.variables.type.binary],
                                                names = ['y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Choice among alternative for each strategy
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            model.variables.add(types = [model.variables.type.binary],
                                                names = ['wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Utility
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            model.variables.add(types = [model.variables.type.continuous],
                                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                                names = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Discounted Utility
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            model.variables.add(types = [model.variables.type.continuous],
                                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                                names = ['zAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        # Maximum Discounted Utility
        for k in range(1, self.K + 1):
            for n in range(self.N):
                for r in range(self.R):
                    for l in range(self.nPriceLevels):
                        model.variables.add(types = [model.variables.type.continuous],
                                            lb = [-cplex.infinity], ub = [cplex.infinity],
                                            names = ['UmaxAft[' + str(k) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])

        ## AUXILIARY VARIABLES
        # Distance between the previous price and next price
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                if i > 0:
                    model.variables.add(obj = [1.0],
                                        types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['a[' + str(k) + ']' + '[' + str(i) + ']'])
                    model.variables.add(obj = [1.0],
                                        types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['b[' + str(k) + ']' + '[' + str(i) + ']'])
                else:
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['a[' + str(k) + ']' + '[' + str(i) + ']'])
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['b[' + str(k) + ']' + '[' + str(i) + ']'])

        # Demand
        for i in range(self.I + 1):
            model.variables.add(types = [model.variables.type.continuous],
                                lb = [-cplex.infinity], ub = [cplex.infinity],
                                names = ['demand[' + str(i) + ']'])

        # Demand after
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for l in range(self.nPriceLevels):
                    model.variables.add(types = [model.variables.type.continuous],
                                        lb = [-cplex.infinity], ub = [cplex.infinity],
                                        names = ['demandAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']'])

        ##### Add the constraints #####

        ##### Linearization
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                if self.Operator[i] == k:
                    indices = ['price[' + str(i) + ']']
                    coefs = [1.0]
                    for l in range(self.nPriceLevels):
                        indices.append('vAft[' + str(k) + ']' + '[' + str(l) + ']')
                        coefs.append(-self.p[i, l])
                    indices.append('a[' + str(k) + ']' + '[' + str(i) + ']')
                    indices.append('b[' + str(k) + ']' + '[' + str(i) + ']')
                    coefs.append(-1.0)
                    coefs.append(1.0)
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [0.0])
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                indices = ['a[' + str(k) + ']' + '[' + str(i) + ']']
                coefs = [1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'G',
                                             rhs = [0.0])
                indices = ['b[' + str(k) + ']' + '[' + str(i) + ']']
                coefs = [1.0]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'G',
                                             rhs = [0.0])

        ##### Price choice - Customer choices
        # BEFORE
        for i in range(self.I + 1):
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
        for i in range(self.I + 1):
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
            for i in range(self.I + 1):
                if self.Operator[i] == k:
                    for n in range(self.N):
                        for r in range(self.R):
                            indices.append('alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                            coefs.append(-1.0/self.R)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [0.0])

        # AFTER
        for k in range(1, self.K + 1):
            for l in range(self.nPriceLevels):
                indices = ['revenueAft[' + str(k) + ']' + '[' + str(l) + ']']
                coefs = [1.0]
                for i in range(self.I + 1):
                    if self.Operator[i] == k:
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
            indices = []
            coefs = []
            for l in range(self.nPriceLevels):
                indices.append('vAft[' + str(k) + ']' + '[' + str(l) + ']')
                coefs.append(1.0)
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [1.0])
        for k in range(1, self.K + 1):
            for l in range(self.nPriceLevels):
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
                         self.M_Rev]
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'L',
                                             rhs = [self.M_Rev])

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
                    for l in range(self.nPriceLevels):
                        indices = []
                        coefs = []
                        for i in range(self.I + 1):
                            indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(1.0)
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [1.0])

        # Customers can only choose options that are available at scenario level
        # BEFORE
        for i in range(self.I + 1):
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
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            indices = ['wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']',
                                       'y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                            coefs = [1.0,
                                     -1.0]
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])

        # Availability at scenario level subject to availability at operator level
        # BEFORE
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'y[' + str(i) + ']']
                    coefs = [1.0,
                             -1.0]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
        for i in range(self.I + 1):
            indices = ['y[' + str(i) + ']']
            coefs = [1.0]
            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                         senses = 'E',
                                         rhs = [1.0])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
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

        # Opt-out is always an available option
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
                    for l in range(self.nPriceLevels):
                        indices = ['y_scenAft[' + str(k) + ']' + '[' + str(0) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                        coefs = [1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [1.0])

        # Alternatives not available at operator level if not included in the customer's choice set
        # BEFORE
        for i in range(self.I + 1):
            for n in range(self.N):
                if self.ChoiceSet[i, n] == 0:
                    for r in range(self.R):
                        indices = ['y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']
                        coefs = [1.0]
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'E',
                                                     rhs = [0.0])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    if self.ChoiceSet[i, n] == 0:
                        for r in range(self.R):
                            for l in range(self.nPriceLevels):
                                indices = ['y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'E',
                                                             rhs = [0.0])

        # Capacity constraints
        # BEFORE
        for i in range(self.I + 1):
            for r in range(self.R):
                indices = []
                coefs = []
                for n in range(self.N):
                    indices.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    coefs.append(1.0)
                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                             senses = 'L',
                                             rhs = [self.Capacity[i]])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for r in range(self.R):
                    for l in range(self.nPriceLevels):
                        indices = []
                        coefs = []
                        for n in range(self.N):
                            indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(1.0)
                        model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                     senses = 'L',
                                                     rhs = [self.Capacity[i]])

        # Priority list
        # BEFORE
        # If alternative not available at scenario level,
        # then the capacity is reached, or the alternative is not available in the choice set
        for i in range(1, self.I + 1): # Do not consider opt-out
            for n in range(self.N):
                for r in range(self.R):
                    indices = []
                    coefs = []
                    # Sum of the customers which have priority
                    for m in range(self.N):
                        if self.PriorityList[i, m] < self.PriorityList[i, n]:
                            indices.append('w[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']')
                            coefs.append(-1.0)
                    indices.append('y[' + str(i) + ']')
                    coefs.append(self.Capacity[i]*self.ChoiceSet[i, n])
                    indices.append('y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    coefs.append(-self.Capacity[i]*self.ChoiceSet[i, n])
                    # Add the constraint
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [0.0])
        # AFTER
        for k in range(1, self.K + 1):
            for i in range(1, self.I + 1): # Do not consider opt-out
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            indices = []
                            coefs = []
                            # Sum of the customers which have priority
                            for m in range(self.N):
                                if self.PriorityList[i, m] < self.PriorityList[i, n]:
                                    indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                    coefs.append(-1.0)
                            indices.append('yAft[' + str(k) + ']' + '[' + str(i) + ']')
                            coefs.append(self.Capacity[i]*self.ChoiceSet[i, n])
                            indices.append('y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            coefs.append(-self.Capacity[i]*self.ChoiceSet[i, n])
                            # Add the constraint
                            model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                         senses = 'L',
                                                         rhs = [0.0])

        # BEFORE
        # Priority list: if alternative is available at scenario level,
        # then there is still some free capacity for the customer
        for i in range(1, self.I + 1): # Do not consider opt-out
            for r in range(self.R):
                for n in range(self.N):
                    if (self.PriorityList[i, n] > self.Capacity[i]) and \
                       (self.ChoiceSet[i, n] == 1):
                       indices = []
                       coefs = []
                       # Sum of the customers which have priority
                       for m in range(self.N):
                           if self.PriorityList[i, m] < self.PriorityList[i, n]:
                               indices.append('w[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']')
                               coefs.append(1.0)
                       indices.append('y_scen[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                       coefs.append(-self.Capacity[i] + self.PriorityList[i, n])
                       # Add the constraint
                       model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                    senses = 'L',
                                                    rhs = [self.PriorityList[i, n] - 1.0])

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(1, self.I + 1): # Do not consider opt-out
                for r in range(self.R):
                    for l in range(self.nPriceLevels):
                        for n in range(self.N):
                            if (self.PriorityList[i, n] > self.Capacity[i]) and \
                               (self.ChoiceSet[i, n] == 1):
                               indices = []
                               coefs = []
                               # Sum of the customers which have priority
                               for m in range(self.N):
                                   if self.PriorityList[i, m] < self.PriorityList[i, n]:
                                       indices.append('wAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(m) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                                       coefs.append(1.0)
                               indices.append('y_scenAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                               coefs.append(-self.Capacity[i] + self.PriorityList[i, n])
                               # Add the constraint
                               model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                            senses = 'L',
                                                            rhs = [self.PriorityList[i, n] - 1.0])

        # Utility function
        # BEFORE
        for i in range(self.I + 1):
            for n in range(self.N):
                for r in range(self.R):
                    indices = ['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']',
                               'price[' + str(i) + ']']
                    coefs = [1.0, -self.EndoCoef[i, n, r]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'E',
                                                 rhs = [self.ExoUtility[i, n, r] + self.xi[i, n, r]])

        # AFTER
        for i in range(self.I + 1):
            for k in range(1, self.K + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
                            if self.Operator[i] == k:
                                indices = ['UAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']
                                coefs = [1.0]
                                model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                             senses = 'E',
                                                             rhs = [self.ExoUtility[i, n, r] + self.EndoCoef[i, n, r] * self.p[i, l] + self.xi[i, n, r]])
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

        # BEFORE
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
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
                    coefs = [1.0, -1.0, self.M_U[n, r]]
                    model.linear_constraints.add(lin_expr = [[indices, coefs]],
                                                 senses = 'L',
                                                 rhs = [self.M_U[n, r]])

        # AFTER
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for n in range(self.N):
                    for r in range(self.R):
                        for l in range(self.nPriceLevels):
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
        for k in range(1, self.K + 1):
            for i in range(self.I + 1):
                for l in range(self.nPriceLevels):
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
        #TODO: Add print solution
        try:
            print("--SOLUTION : --")
            model.set_results_stream(None)
            model.set_warning_stream(None)
            model.solve()
            print(model.solution.get_objective_value())
            return model
        except CplexSolverError as e:
            print('Exception raised during dual of restricted problem')

if __name__ == '__main__':
    # Get the data and preprocess
    dict = data_file.getData()
    data_file.preprocess(dict)
    # Instanciate a Stackelberg game and solve it
    game = Fixed_Point(**dict)
    model = game.getModel()
    print('MODEL COMPUTED')
    game.solveModel(model)
