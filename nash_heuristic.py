# Modelisation of the heuristic algorithm.
# For each operator the endogene variables are discretized in order to construct a set of possible strategies.
# Then the sequential game is performed, starting from an initial unvisited configuration.
# The algorithm stops when all the possible configurations () have been explored

# General
import sys
import time
import copy
import math
import random
# Ipopt
import ipopt
# numpy
import numpy as np
# data
import Data.Non_linear_Stackelberg.ProbLogit_n10 as data_file
import non_linear_stackelberg

class NashHeuristic(object):
    def __init__(self, **kwargs):
        ''' Construct a Stackelberg game.
            KeywordArgs:
                K                Number of operators
                I                Number of alternatives
                Operator         Mapping between alternatives and operators
                Optimizer        Index of the current operator
                InitialOptimizer Index of the operator who started the run
                EndoVar          Mapping between alternatives and corresponding endogene variables
        '''
        self.K = kwargs.get('K')
        self.I = kwargs.get('I')
        # TODO: For the moment, suppose that an operator is in charge of exactly one alternative
        self.Operator = kwargs.get('Operator')
        self.Optimizer = 1
        # Copy the initial optimizer
        self.InitialOptimizer = self.Optimizer
        # TODO: For the moment, suppose EndoVar contains only the price
        self.EndoVar = kwargs.get('EndoVar')
        # Mapping between an integer and a specific strategy
        self.mapping = [[] for n in range(self.K + 1)]

    def constructMapping(self):
        ''' For each operator, construct a mapping between its strategies set and
            the corresponding endogene variables value.
        '''
        for k in range(1, self.K + 1):
            for i in range(1, self.I + 1):
                if self.Operator[i] == k:
                    # Get the variable corresponding to the alternative
                    vars = self.EndoVar[i]
                    for key in vars.keys():
                        var = vars[key]
                        if var['domain'] == 'C':
                            range_lb = [var['lb'] + n * (var['ub'] - var['lb'])/var['step']
                                        for n in range(var['step'])]

                            range_ub = [var['lb'] + (n + 1) * (var['ub'] - var['lb'])/var['step']
                                        for n in range(var['step'])]
                        else:
                            raise NotImplementedError('To be implemented')
                        for n in range(var['step']):
                            self.mapping[k].append({key: [range_lb[n], range_ub[n]]})

    def constructTables(self):
        ''' Construct tables. Each table represent all the possible configurations
            (cartesian product of the operator's strategies) given the current optimizer
        '''
        self.tables = []
        for k in range(1, self.K + 1):
            table_dim = [len(self.mapping[l]) for l in range(1, self.K + 1) if l != k]
            table = np.full(table_dim, -1)
            self.tables.append(table)

    def sequentialGame(self, data):
        ''' Run the sequential game with the Stackelberg game
            Args:
                data:          dictionary containing data to instanciate the Stackelberg game
        '''
        iter = 1
        cycle = False
        cycle_iter = 0
        visited = False
        p_history = [data['p_fixed']]

        while not visited and not cycle:
            # Initialize the future price history list for the current iteration
            p_history.append([])
            print('--- ITERATION %r ----\n' %iter)
            print('Optimizer: %r' %self.Optimizer)
            for (index, p) in enumerate(data['p_fixed']):
                if p != -1.0:
                    print('Initial price of alternative %r set by operator %r : %r'
                           %(index, self.Operator[index], p))
            # Run the game for the current optimizer
            prices = non_linear_stackelberg.main(data)
            # Update the price history
            p_history[iter] = copy.deepcopy(prices)
            # Check for the cycle
            if iter > self.K:
                iteration_to_check = range(iter-self.K, 0, -self.K)
                for j in iteration_to_check:
                    cycle = True
                    for i in range(self.I + 1):
                        # TODO: Numerical error ?
                        if abs(p_history[j][i] - p_history[iter][i]) > 1e-4:
                            cycle = False
                    if cycle is True:
                        cycle_iter = j
                        if iter-cycle_iter == self.K:
                            print('\nNash equilibrium detected')
                        else:
                            print('\nCycle detected')
                        break
            # Update the data for the next iteration
            #TODO: Add the possibility select the specific order
            data['Optimizer'] = (data['Optimizer'] % self.K) + 1
            self.Optimizer = data['Optimizer']
            # Check if the next configuration has already been visited
            visited = self.alreadyVisited(prices)
            # Fix the price of the operators except for the next optimizer
            data['p_fixed'] = copy.deepcopy(prices)
            for (i, k) in enumerate(self.Operator):
                if k == data['Optimizer']:
                    data['p_fixed'][i] = -1.0
            # Go to the next iteration
            iter += 1

        if cycle and iter-cycle_iter == self.K:
            self.updateTables(p_history, nash = 1)
        elif cycle:
            self.updateTables(p_history, nash = 0)
        else:
            # Remove the last entry since we have already visited it
            del p_history[-1]
            self.updateTables(p_history, nash = 0)

    def alreadyVisited(self, prices):
        ''' Check whether a given configuration have already been visited before.
            Args:
                prices          list of prices for each alternatives
        '''
        reverse_indices = []
        for k in range(1, self.K + 1):
            for i, price in enumerate(prices):
                if i > 0 and self.Operator[i] == k and self.Operator[i] != self.Optimizer:
                    lb = self.EndoVar[i]['p']['lb']
                    ub = self.EndoVar[i]['p']['ub']
                    step = self.EndoVar[i]['p']['step']
                    reverse_index = math.floor(-lb + (price*step)/(ub-lb))
                    reverse_indices.append(reverse_index)
        if self.K == 2:
            if self.tables[self.Optimizer - 1][reverse_indices[0]] in [0, 1]:
                print('The next configuration \n Optimizer: %r \n Prices: %r \n\
has already been visited before.'%(self.Optimizer, prices))
                return True
        else:
            raise NotImplementedError('To be implemented')
        return False

    def updateTables(self, p_history, nash = 0):
        ''' Update the tables according to the prices history and whether
            a Nash equilibrium is found.
            Args:
                p_history          history of the prices list of the sequential game
                nash               1 if a nash equilibrium has been found, 0 ow.
        '''
        print('\n- Update the tables -')
        current_oper = self.InitialOptimizer
        for (iter, prices) in enumerate(p_history):
            reverse_indices = []
            for k in range(1, self.K + 1):
                for (i, price) in enumerate(prices):
                    if i > 0 and self.Operator[i] == k and self.Operator[i] != current_oper:
                        lb = self.EndoVar[i]['p']['lb']
                        ub = self.EndoVar[i]['p']['ub']
                        step = self.EndoVar[i]['p']['step']
                        reverse_index = math.floor(-lb + (price*step)/(ub-lb))
                        reverse_indices.append(reverse_index)
            if self.K == 2:
                # Update the initial state for the next run
                if (current_oper == self.initial_state[0]) and (reverse_indices[0] == self.initial_state[1]):
                    not_found = True
                    while not_found:
                        self.initial_state[1] += 1
                        if self.initial_state[1] == len(self.mapping[current_oper]):
                            # The next operator starts
                            self.initial_state[0] += 1
                            if self.initial_state[0] > self.K:
                                # The table is fully filled
                                break
                            else:
                                self.initial_state[1] = 0
                        if self.tables[self.initial_state[0] - 1][self.initial_state[1]] == -1:
                            not_found = False
                # Update the table
                if nash == 0:
                    self.tables[current_oper - 1][reverse_indices[0]] = 0
                else:
                    if len(p_history) - iter <= self.K:
                        self.tables[current_oper - 1][reverse_indices[0]] = 1
                    else:
                        self.tables[current_oper - 1][reverse_indices[0]] = 0
            else:
                raise NotImplementedError('To be implemented')

            current_oper = (current_oper % self.K) + 1

    def run(self, data):
        self.constructMapping()
        self.constructTables()
        # Initialize the initial state of the sequential game
        self.initial_state = [self.InitialOptimizer, 0]
        run_number = 1
        # While the tables are not fully filled
        while self.initial_state[0] <= self.K:
            print ('\n--- RUN %r ----' %run_number)
            # Update the optimizer index
            self.InitialOptimizer = self.initial_state[0]
            data['Optimizer'] = self.initial_state[0]
            self.Optimizer = self.initial_state[0]
            # Get an initial fixed price in the initial state
            p_fixed = []
            for i in range(self.I + 1):
                if i == 0:
                    p_fixed.append(0.0)
                elif self.Operator[i] == self.initial_state[0]:
                    p_fixed.append(-1.0)
                else:
                    lb = self.EndoVar[i]['p']['lb']
                    ub = self.EndoVar[i]['p']['ub']
                    step = self.EndoVar[i]['p']['step']
                    p_fixed.append(random.uniform(lb + self.initial_state[1]*(ub-lb)/step,
                                                  lb + (self.initial_state[1] + 1)*(ub-lb)/step))

            data['p_fixed'] = copy.deepcopy(np.asarray(p_fixed))
            self.sequentialGame(data)
            run_number += 1
        print('\n--- FINAL RESULTS ---')
        for (k, table) in enumerate(self.tables):
            # The operator index starts at 1. 0 is opt-out
            print('\nOptimizer %r' %(k + 1))
            print('Prices \t\t Nash Equilibrium found\n')
            for (p_range, value) in zip(self.mapping[k + 1], self.tables[k]):
                print('%r \t\t %r' %(list(p_range.values())[0], value))


if __name__ == '__main__':
    # Get the data and preprocess
    data = data_file.getData()
    data_file.preprocess(data)

    nash_dict = {'K': 2,
                'I': 2,
                'Operator': [0, 1, 2],
                'Optimizer': 1,
                'EndoVar': {0:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 0.0, 'step':1}},
                            1:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':50}},
                            2:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':50}}}
                }
    # Instanciate the heuristic game
    game = NashHeuristic(**nash_dict)
    data.update(nash_dict)
    game.run(data)
