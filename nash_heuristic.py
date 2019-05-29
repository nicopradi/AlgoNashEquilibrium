# Modelisation of the heuristic algorithm.
# For each operator the endogene variables domains are discretized in order to construct a finite set of strategies.
# Then the NLP sequential game is performed, starting from an initial unvisited configuration.
# The algorithm stops when all the possible configurations have been explored

# General
import sys
import time
import copy
import math
import random
import warnings
import itertools
# Ipopt
import ipopt
# Numpy
import numpy as np
# Data
import Data.Italian.Non_linear_Stackelberg.ProbLogit_n40 as data_file
import non_linear_stackelberg

class NashHeuristic(object):
    def __init__(self, **kwargs):
        ''' Construct a Nash Heuristic game.
            KeywordArgs:
                K                  Number of operators [int]
                I                  Number of alternatives (without opt-out) [int]
                I_opt_out          Number of opt-out alternatives [int]
                operator           Mapping between alternatives and operators [list]
                endo_var           Mapping between alternatives and endogene variables [dict]
                tolerance          Tolerance used for the cycle detection [float]
        '''
        self.K = kwargs.get('K', 2)
        self.I = kwargs.get('I', 2)
        self.I_opt_out = kwargs.get('I_opt_out', 1)
        # TODO: For the moment, an operator is in charge of exactly one alternative
        self.operator = kwargs.get('operator')
        # TODO: For the moment, suppose endo_var contains only the price
        # The keys correspond to an alternative,
        # the values are the endogene variables for the corresponding alternative
        self.endo_var = kwargs.get('endo_var')
        self.tolerance = kwargs.get('tolerance', 1e-3)
        # Set the optimizer to be the first operator
        self.optimizer = 1
        # Optimizer id at the beggining of the current run
        self.initial_optimizer = self.optimizer
        # Mapping between an integer and an operator's specific strategy
        self.mapping = [[] for n in range(self.K + 1)]
        # Seed
        self.seed = kwargs.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)

    def constructMapping(self):
        ''' For each operator, construct a mapping between its strategies and
            an index. The strategy belongs to the finite strategy set of the operator.
        '''
        for k in range(1, self.K + 1):
            # Keep track of all the strategy of the current operator
            strategies = []
            for i in range(self.I_opt_out, self.I + self.I_opt_out):
                # Keep track of the strategy of the current alternative
                strategy = []
                if self.operator[i] == k:
                    # Get the endogenous variables corresponding to the alternative
                    vars = self.endo_var[i]
                    for key in vars.keys():
                        # For each endogenous variable discretize it based on its domain
                        var = vars[key]
                        if var['domain'] == 'C':
                            # Generate all the possible ranges for the continuous variable
                            # All the subsets (price's range) have the same size
                            range_lb = [var['lb'] + n * (var['ub'] - var['lb'])/var['step']
                                        for n in range(var['step'])]

                            range_ub = [var['lb'] + (n + 1) * (var['ub'] - var['lb'])/var['step']
                                        for n in range(var['step'])]
                        else:
                            raise NotImplementedError('Domain of the variable is not continuous. To be implemented')
                        # Mapping between the strategy in the finite strategy set and an index (key)
                        for n in range(var['step']):
                            # For the moment we suppose that the price is the only endogenous
                            # variables. Each strategy is added to the corresponding operator strategy set
                            strategy.append({i: [round(range_lb[n], 3), round(range_ub[n], 3)]})
                        strategies.append(strategy)
            self.mapping[k] = list(itertools.product(*strategies))
        # Merge the alternative strategies in the same dictionary to represent an operator strategy
        for k in range(1, self.K + 1):
            for s in range(len(self.mapping[k])):
                strat = {}
                for i_k in range(len(self.mapping[k][s])):
                    strat.update(self.mapping[k][s][i_k])
                self.mapping[k][s] = strat

    def constructTables(self):
        ''' Construct the tables. Each table represents all the possible configurations
            (cartesian product of the optimizer's competitor strategies) given the current optimizer.
            The index of the table correspond to the optimizer.
            The entry in the tables will be equal to the resulting cycle amplitude. The closer the
            value is to 0, the more likely it is a Nash equilibrium candidate solution.
            When a configuration is not part of a cycle, its entry value is infinite.
        '''
        # Note that the first table corresponds to the configuration when the operator 1 is playing.
        self.tables = []
        for k in range(1, self.K + 1):
            # Get the size of the strategy set for each operator, except the current one
            table_dim = [len(self.mapping[l]) for l in range(1, self.K + 1) if l != k]
            # Construct the resulting table for the optimizer k
            table = np.full(table_dim, -1.0)
            self.tables.append(table)

    def sequentialGame(self, data):
        ''' Run the sequential game with the NLP Stackelberg game.
            The initial configuration is given in data.
            Args:
                data          dictionary containing data to instanciate the Stackelberg game [dict]
        '''
        # Keep track of the iteration number
        iter = 1
        # Keep track of the detection of cycle as well as the iteration from which it starts
        cycle = False
        cycle_iter = 0
        # visited is True if the current configuration has already been visited before
        visited = False
        # Initialize the price history of the run.
        # The entry is the initial configuration of the run
        p_history = [data['p_fixed']]

        # The sequential game stops when a cycle is detected or
        # a configuration has already been visited in a previous run
        while not visited and not cycle:
            # Initialize the price history list of the current iteration
            p_history.append([])
            print('--- ITERATION %r ----\n' %iter)
            print('Optimizer: %r' %self.optimizer)
            for (index, p) in enumerate(data['p_fixed']):
                if p != -1.0:
                    print('Initial price of alternative %r set by operator %r : %r'
                           %(index, self.operator[index], p))
            ### Run the NLP Stackelberg game
            prices, _, x0, status, status_msg = non_linear_stackelberg.main(data)
            # Update the price history
            p_history[iter] = copy.deepcopy(prices)

            ### Cycle detection in the current run based the tolerance parameter
            if iter > self.K:
                # Check the prices at the iteration for which the current optimizer played
                iteration_to_check = range(iter-self.K, 0, -self.K)
                for j in iteration_to_check:
                    cycle = True
                    for i in range(self.I + self.I_opt_out):
                        if abs(p_history[j][i] - p_history[iter][i]) > self.tolerance:
                            cycle = False
                    if cycle is True:
                        # Cycle detected
                        cycle_iter = j
                        if iter-cycle_iter == self.K:
                            print('\nNash equilibrium detected')
                        else:
                            print('\nCycle detected')
                        # Compute the score of the cycle
                        score = self.getCycleAmplitude(p_history[cycle_iter:])
                        print('Cycle score: %r' %score)
                        break

            ### Update the data for the next iteration
            #TODO: Add the possibility to select the specific order
            data['optimizer'] = (data['optimizer'] % self.K) + 1
            self.optimizer = data['optimizer']

            # Change the fixed price of the operators except for the next optimizer
            data['p_fixed'] = copy.deepcopy(prices)
            for (i, k) in enumerate(self.operator):
                if k == data['optimizer']:
                    data['p_fixed'][i] = -1.0

            #### Check if the next configuration has already been visited
            if cycle is False:
                visited, cycle_iter = self.alreadyVisited(prices, iter)
                if cycle_iter is not None:
                    # The next configuration has already been visited in the CURRENT run
                    # Compute the score of the cycle
                    score = self.getCycleAmplitude(p_history[int(cycle_iter)+1:])
                    cycle = True

            # Set the current solution to be the starting point in the next interation
            # except if it is unfeasible
            print('\nIPOPT Status %r : %r\n' %(status, status_msg))
            if status not in [-2, -1, 2]:
                print('Previous solution used as the starting point for the next iteration.')
                data['x0'] = copy.deepcopy(x0)

            # Go to the next iteration
            iter += 1

        #### Update the tables
        if cycle:
            # A cycle has been detected
            self.updateTables(p_history, length=(iter - 1) - cycle_iter, score=score)
        else:
            # A configuration from a previous run has been visited
            # Remove the last entry since we have already visited it
            del p_history[-1]
            self.updateTables(p_history, length=len(p_history), score=np.inf)

    def getCycleAmplitude(self, p_history):
        ''' Compute the amplitude of the cycle. For each alternative, compute
            the highest difference in prices in the cycle. The score of the cycle
            is the sum of these differences.
            Args:
                p_history          price history of the cycle [list]
        '''
        score = 0
        for i in range(self.I_opt_out, self.I + self.I_opt_out):
            history = [price[i] for price in p_history]
            p_min = min(history)
            p_max = max(history)
            score += p_max - p_min

        return score

    def alreadyVisited(self, prices, iter):
        ''' Check whether a given configuration has already been visited before.
            Keep track of the current configuration by updating the corresponding table entry.
            Args:
                prices          list of prices for each alternatives [list]
                iter            current iteration number [int]
            Returns:
                True if the given configuration has already been visited before.
                Also return the iteration number for which the cycle starts.
        '''
        # Keep track of the strategy indices of the current configuration
        reverse_indices = []
        competitor_history = [(i, price) for i, price in enumerate(prices) if (i >= self.I_opt_out) and (self.operator[i] != self.optimizer)]
        for i, price in competitor_history:
            # Compute the index of the strategy of the alternative i
            lb = self.endo_var[i]['p']['lb']
            ub = self.endo_var[i]['p']['ub']
            step = self.endo_var[i]['p']['step']
            if price == ub:
                # the upper bound is located in the last subset
                reverse_index = math.floor((price-lb)*step/(ub-lb)) - 1
            else:
                reverse_index = math.floor((price-lb)*step/(ub-lb))
            reverse_indices.append(reverse_index)
        # TODO: Clean the lines below
        if len(reverse_indices) == 1:
            reverse_indices = reverse_indices[0]
        elif len(reverse_indices) == 2:
            reverse_indices = step*reverse_indices[0] + reverse_indices[1]
        elif len(reverse_indices) > 2:
            raise NotImplementedError('To be implemented')

        if self.K == 2:
            table_value = self.tables[self.optimizer - 1][reverse_indices]
            if table_value >= 0:
                print('The next configuration \n Optimizer: %r \n Prices: %r \n\
has already been visited before.'%(self.optimizer, prices))
                if table_value % 1 == 0:
                    # The current configuration has already been visited in the CURRENT run
                    return True, self.tables[self.optimizer - 1][reverse_indices]
                else:
                    # The current configuration has already been visited in a PREVIOUS run
                    return True, None
        else:
            raise NotImplementedError('To be implemented')

        # The current configuration has not been visited before.
        # Mark its entry to keep track of it
        # Update the current configuration in the table
        self.tables[self.optimizer - 1][reverse_indices] = iter

        return False, None

    def updateTables(self, p_history, length=0, score=np.inf):
        ''' Update the tables according to the prices history,
            length and score of the cycle.
            Args:
                p_history          history of the prices [list]
                length             number of iteration in the cycle [int]
                score              score of the cycle [int]
        '''
        print('\n- Update the tables -')
        # Get the optimizer if at the beginning of the run
        current_oper = self.initial_optimizer
        #### Update the tables for each iteration
        for (iter, prices) in enumerate(p_history):
            # Get the strategy indices for the current operators's strategy defined by prices
            reverse_indices = []
            competitor_history = [(i, price) for i, price in enumerate(prices) if (i >= self.I_opt_out) and (self.operator[i] != current_oper)]
            for i, price in competitor_history:
                # Compute the index of the strategy of the alternative i
                lb = self.endo_var[i]['p']['lb']
                ub = self.endo_var[i]['p']['ub']
                step = self.endo_var[i]['p']['step']
                if price == ub:
                    # the upper bound is located in the last subset
                    reverse_index = math.floor((price-lb)*step/(ub-lb)) - 1
                else:
                    reverse_index = math.floor((price-lb)*step/(ub-lb))
                reverse_indices.append(reverse_index)
            # TODO: Clean the lines below
            if len(reverse_indices) == 1:
                reverse_indices = reverse_indices[0]
            elif len(reverse_indices) == 2:
                reverse_indices = step*reverse_indices[0] + reverse_indices[1]
            elif len(reverse_indices) > 2:
                raise NotImplementedError('To be implemented')
            if self.K == 2:
                # Update the table
                if score == np.inf:
                    self.tables[current_oper - 1][reverse_indices] = np.inf
                else:
                    if len(p_history) - iter <= length:
                        self.tables[current_oper - 1][reverse_indices] = score
                    else:
                        # Iteration number is not part of the cycle
                        self.tables[current_oper - 1][reverse_indices] = np.inf
            else:
                raise NotImplementedError('To be implemented')

            #### Update the initial state for the next run
            if (current_oper == self.initial_state[0]) and (reverse_indices == self.initial_state[1]):
                not_found = True
                while not_found:
                    # Try to increment the strategy index of the operator in initial_state
                    self.initial_state[1] += 1
                    if self.initial_state[1] == len(self.mapping[current_oper]):
                        # All the strategy indices of the operator in initial_state have been visited.
                        # Complete the remaining entries in the tables with infinity
                        for k in range(len(self.tables)):
                            for strat in range(len(self.tables[k])):
                                if self.tables[k][strat] < 0:
                                    self.tables[k][strat] = np.inf
                        # Set the config operator at K+1 to indicate that the tables are filled
                        self.initial_state[0] = self.K + 1
                        break
                    if self.tables[self.initial_state[0] - 1][self.initial_state[1]] == -1:
                        # A unvisited configuration has been found
                        not_found = False
            #### Update the optimizer id for the next iteration to analyse
            current_oper = (current_oper % self.K) + 1

    def run(self, data):
        ''' Run the heuristic game.
            Args:
                data          data for the MINLP Stackelberg formulation [dict]
        '''
        #### Construct the mapping and the tables
        self.constructMapping()
        self.constructTables()
        # Initialize the initial state for the current run
        # [operator_id, strategy index]
        self.initial_state = [self.initial_optimizer, 0]
        # Keep track of the number of run
        run_number = 1

        #### While the tables are not completed starts a new run
        while self.initial_state[0] <= self.K:
            print ('\n--- RUN %r ----' %run_number)
            # Update the optimizer index
            self.initial_optimizer = self.initial_state[0]
            data['optimizer'] = self.initial_state[0]
            self.optimizer = self.initial_state[0]
            # Compute the initial fixed prices of the initial state
            p_fixed = copy.deepcopy(data['lb_p'])
            for i in range(self.I_opt_out, self.I + self.I_opt_out):
                if self.operator[i] == self.initial_state[0]:
                    p_fixed[i] = -1.0
                else:
                    lb = self.endo_var[i]['p']['lb']
                    ub = self.endo_var[i]['p']['ub']
                    step = self.endo_var[i]['p']['step']
                    p_fixed[i] = random.uniform(self.mapping[self.operator[i]][self.initial_state[1]][i][0],
                                                self.mapping[self.operator[i]][self.initial_state[1]][i][1])

            data['p_fixed'] = copy.deepcopy(np.asarray(p_fixed))
            # Run the sequential game
            self.sequentialGame(data)
            # Remove the previous solution for the next run
            # since it is unlikely a feasible solution
            if 'x0' in data.keys():
                del data['x0']
            # Go to the next run
            run_number += 1

        #### Print the final results
        print('\n--- FINAL RESULTS ---')
        for (k, table) in enumerate(self.tables):
            # The operator index starts at 1. 0 is opt-out
            print('\nOptimizer %r' %(k + 1))
            print('Prices \t\t Nash Equilibrium found\n')
            for (p_range, value) in zip(self.mapping[((k+1)%2) + 1], self.tables[k]):
                print('%r \t\t\t\t\t %r' %(p_range, value))

if __name__ == '__main__':
    # Get the data and preprocess
    t_0 = time.time()
    data = data_file.getData()
    data_file.preprocess(data)

    # Define the keywords arguments
    nash_dict = {'K': 2,
                'operator': [0, 0, 0, 0, 1, 1, 2, 2],
                'optimizer': 1,
                'endo_var': {0:{'p':{'domain': 'C', 'lb': data['lb_p'][0], 'ub': data['ub_p'][0], 'step':1}},
                            1:{'p':{'domain': 'C', 'lb': data['lb_p'][1], 'ub': data['ub_p'][1], 'step':1}},
                            2:{'p':{'domain': 'C', 'lb': data['lb_p'][2], 'ub': data['ub_p'][2], 'step':1}},
                            3:{'p':{'domain': 'C', 'lb': data['lb_p'][3], 'ub': data['ub_p'][3], 'step':1}},
                            4:{'p':{'domain': 'C', 'lb': data['lb_p'][4], 'ub': data['ub_p'][4], 'step':20}},
                            5:{'p':{'domain': 'C', 'lb': data['lb_p'][5], 'ub': data['ub_p'][5], 'step':20}},
                            6:{'p':{'domain': 'C', 'lb': data['lb_p'][6], 'ub': data['ub_p'][6], 'step':20}},
                            7:{'p':{'domain': 'C', 'lb': data['lb_p'][7], 'ub': data['ub_p'][7], 'step':20}}},
                'seed': 1
                }

    data.update(nash_dict)
    # Instanciate and run the heuristic game
    game = NashHeuristic(**data)
    game.run(data)
    t_1 = time.time()
    print('TOTAL TIMING: %r' %(t_1 - t_0))
