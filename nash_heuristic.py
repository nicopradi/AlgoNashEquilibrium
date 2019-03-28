# Modelisation of the heuristic algorithm.
# For each operator the endogene variables are discretized in order to construct a set of possible strategies.
# Then the sequential game is performed, starting from an initial unvisited configuration.
# The algorithm stops when all the possible configurations () have been explored

# TODO: Make the variables/method names consistent

# General
import sys
import time
import copy
import math
import random
# Ipopt
import ipopt
# Numpy
import numpy as np
# Data
import Data.Non_linear_Stackelberg.ProbLogit_n50 as data_file
import non_linear_stackelberg

class NashHeuristic(object):
    def __init__(self, **kwargs):
        ''' Construct a Nash Heuristic game.
            KeywordArgs:
                K                  Number of operators
                I                  Number of alternatives
                operator           Mapping between alternatives and operators
                endo_var            Mapping between alternatives and corresponding endogene variables
        '''
        self.K = kwargs.get('K')
        self.I = kwargs.get('I')
        # TODO: For the moment, suppose that an operator is in charge of exactly one alternative
        self.operator = kwargs.get('operator')
        self.optimizer = 1
        # Copy the initial optimizer
        self.initial_optimizer = self.optimizer
        # TODO: For the moment, suppose endo_var contains only the price
        self.endo_var = kwargs.get('endo_var')
        # Mapping between an integer and a specific strategy
        self.mapping = [[] for n in range(self.K + 1)]

    def constructMapping(self):
        ''' For each operator, construct a mapping between its strategies set and
            the corresponding endogene variables value.
        '''
        for k in range(1, self.K + 1):
            for i in range(1, self.I + 1):
                if self.operator[i] == k:
                    # Get the endogenous variables corresponding to the alternative
                    vars = self.endo_var[i]
                    for key in vars.keys():
                        var = vars[key]
                        if var['domain'] == 'C':
                            # Generate all the possible ranges for the corresponding variables
                            # All the subsets (price's range) have the same size
                            range_lb = [var['lb'] + n * (var['ub'] - var['lb'])/var['step']
                                        for n in range(var['step'])]

                            range_ub = [var['lb'] + (n + 1) * (var['ub'] - var['lb'])/var['step']
                                        for n in range(var['step'])]
                        else:
                            raise NotImplementedError('To be implemented')
                        for n in range(var['step']):
                            # For the moment we suppose that the price is the only endogenous
                            # variables. Each strategy is added to the corresponding operator strategy set
                            self.mapping[k].append({key: [range_lb[n], range_ub[n]]})

    def constructTables(self):
        ''' Construct tables. Each table represents all the possible configurations
            (cartesian product of the operator's strategies) given the current optimizer.
            The entry in the tables will be equal to 1 if the corresponding configuration yields a
            strategy nash equilibrium in the next itertion. Otherwise it is equal to 0.
        '''
        # Note that the first array in tables corresponds to: operator 1 optimizing.
        self.tables = []
        for k in range(1, self.K + 1):
            # Get the size of the strategy set for each operator, except the current one
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
        # visited is True if the current configuration has already been visited before
        visited = False
        p_history = [data['p_fixed']]

        while not visited and not cycle:
            # Initialize the future price history list for the current iteration
            p_history.append([])
            print('--- ITERATION %r ----\n' %iter)
            print('Optimizer: %r' %self.optimizer)
            for (index, p) in enumerate(data['p_fixed']):
                if p != -1.0:
                    print('Initial price of alternative %r set by operator %r : %r'
                           %(index, self.operator[index], p))
            # Run the game with the initial configuration
            prices, _ = non_linear_stackelberg.main(data)
            # Update the price history
            p_history[iter] = copy.deepcopy(prices)
            # Check for the cycle
            if iter > self.K:
                iteration_to_check = range(iter-self.K, 0, -self.K)
                for j in iteration_to_check:
                    cycle = True
                    for i in range(self.I + 1):
                        # TODO: Tolerance value ?
                        if abs(p_history[j][i] - p_history[iter][i]) > 1e-3:
                            cycle = False
                    if cycle is True:
                        cycle_iter = j
                        if iter-cycle_iter == self.K:
                            print('\nNash equilibrium detected')
                        else:
                            print('\nCycle detected')
                        #TODO: Get the score of the cycle
                        score = self.getCycleAmplitude(p_history[cycle_iter:])
                        break
            # Update the data for the next iteration
            #TODO: Add the possibility select the specific order
            data['optimizer'] = (data['optimizer'] % self.K) + 1
            self.optimizer = data['optimizer']
            # Check if the next configuration has already been visited
            visited = self.alreadyVisited(prices)
            # Change the fixed price of the operators except for the next optimizer
            data['p_fixed'] = copy.deepcopy(prices)
            for (i, k) in enumerate(self.operator):
                if k == data['optimizer']:
                    data['p_fixed'][i] = -1.0
            # Go to the next iteration
            iter += 1

        # Update the tables
        if cycle:
            self.updateTables(p_history, length=(iter - 1) - cycle_iter, score=score)
        else:
            # Remove the last entry since we have already visited it
            del p_history[-1]
            self.updateTables(p_history, length=len(p_history), score=np.inf)

    def getCycleAmplitude(self, p_history):
        ''' Compute the amplitude of the cycle. For each alternative, compute
            the highest difference in prices in the cycle. The score of the cycle
            is the sum of these differences
            Args:
                p_history          price history
        '''
        score = 0
        for i in range(1, self.I + 1):
            history = [price[i] for price in p_history]
            p_min = min(history)
            p_max = max(history)
            score = p_max-p_min

        return score

    def alreadyVisited(self, prices):
        ''' Check whether a given configuration has already been visited before.
            Args:
                prices          list of prices for each alternatives
        '''
        # Get the strategy indices for the current operators's strategy
        reverse_indices = []
        for k in range(1, self.K + 1):
            for i, price in enumerate(prices):
                if i > 0 and self.operator[i] == k and self.operator[i] != self.optimizer:
                    lb = self.endo_var[i]['p']['lb']
                    ub = self.endo_var[i]['p']['ub']
                    step = self.endo_var[i]['p']['step']
                    # reverse_index contains the strategy index
                    reverse_index = math.floor(-lb + (price*step)/(ub-lb))
                    reverse_indices.append(reverse_index)
        if self.K == 2:
            if self.tables[self.optimizer - 1][reverse_indices[0]] >= 0:
                print('The next configuration \n Optimizer: %r \n Prices: %r \n\
has already been visited before.'%(self.optimizer, prices))
                return True
        else:
            raise NotImplementedError('To be implemented')
        return False

    def updateTables(self, p_history, length=0, score=np.inf):
        ''' Update the tables according to the prices history and whether
            a Nash equilibrium is found.
            Args:
                p_history          history of the prices list of the sequential game
                nash               1 if a nash equilibrium has been found, 0 ow.
        '''
        print('\n- Update the tables -')
        current_oper = self.initial_optimizer
        for (iter, prices) in enumerate(p_history):
            # Get the strategy indices for the current operators's strategy defined by prices
            reverse_indices = []
            for k in range(1, self.K + 1):
                for (i, price) in enumerate(prices):
                    if i > 0 and self.operator[i] == k and self.operator[i] != current_oper:
                        lb = self.endo_var[i]['p']['lb']
                        ub = self.endo_var[i]['p']['ub']
                        step = self.endo_var[i]['p']['step']
                        reverse_index = math.floor(-lb + (price*step)/(ub-lb))
                        reverse_indices.append(reverse_index)
            if self.K == 2:
                # Update the initial state for the next run
                if (current_oper == self.initial_state[0]) and (reverse_indices[0] == self.initial_state[1]):
                    not_found = True
                    while not_found:
                        # Try the increment the strategy index of the operation in initial_state
                        self.initial_state[1] += 1
                        if self.initial_state[1] == len(self.mapping[current_oper]):
                            # All the strategy indices of the operation in initial_state have been visited.
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
                if score == np.inf:
                    self.tables[current_oper - 1][reverse_indices[0]] = np.inf
                else:
                    if len(p_history) - iter <= length:
                        self.tables[current_oper - 1][reverse_indices[0]] = score
                    else:
                        self.tables[current_oper - 1][reverse_indices[0]] = np.inf
            else:
                raise NotImplementedError('To be implemented')

            current_oper = (current_oper % self.K) + 1

    def run(self, data):
        # Construct the mapping and the tables
        self.constructMapping()
        self.constructTables()
        # Initialize the initial state of the sequential game
        self.initial_state = [self.initial_optimizer, 0]
        run_number = 1
        # While the tables are not fully filled
        while self.initial_state[0] <= self.K:
            print ('\n--- RUN %r ----' %run_number)
            # Update the optimizer index
            self.initial_optimizer = self.initial_state[0]
            data['optimizer'] = self.initial_state[0]
            self.optimizer = self.initial_state[0]
            # Compute the initial fixed prices to the corresponding initial state
            p_fixed = []
            for i in range(self.I + 1):
                if i == 0:
                    p_fixed.append(0.0)
                elif self.operator[i] == self.initial_state[0]:
                    p_fixed.append(-1.0)
                else:
                    lb = self.endo_var[i]['p']['lb']
                    ub = self.endo_var[i]['p']['ub']
                    step = self.endo_var[i]['p']['step']
                    p_fixed.append(random.uniform(lb + self.initial_state[1]*(ub-lb)/step,
                                                  lb + (self.initial_state[1] + 1)*(ub-lb)/step))

            data['p_fixed'] = copy.deepcopy(np.asarray(p_fixed))
            # Run the sequential game
            self.sequentialGame(data)
            run_number += 1

        # Print the final results
        print('\n--- FINAL RESULTS ---')
        for (k, table) in enumerate(self.tables):
            # The operator index starts at 1. 0 is opt-out
            print('\nOptimizer %r' %(k + 1))
            print('Prices \t\t Nash Equilibrium found\n')
            for (p_range, value) in zip(self.mapping[k + 1], self.tables[k]):
                print('%r \t\t %r' %(list(p_range.values())[0], value))


if __name__ == '__main__':
    # Get the data and preprocess
    t_0 = time.time()
    data = data_file.getData()
    data_file.preprocess(data)

    nash_dict = {'K': 2,
                'I': 2,
                'operator': [0, 1, 2],
                'optimizer': 1,
                'endo_var': {0:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 0.0, 'step':1}},
                            1:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':100}},
                            2:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':100}}}
                }
    # Instanciate the heuristic game
    game = NashHeuristic(**nash_dict)
    data.update(nash_dict)
    game.run(data)
    t_1 = time.time()
    print('TOTAL TIMING: %r' %(t_1 - t_0))
