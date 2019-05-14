# Python script containing several options to run the blocks of the algorithmic framework together.

# General
import sys
import time
import copy
import math
import random
import warnings
from decimal import Decimal
# Ipopt
import ipopt
# Numpy
import numpy as np
# Project
import stackelberg_game
import nash_heuristic
import sequential_game
import fixed_point
import fixed_point_no_capacity
# Data
import Data.Non_linear_Stackelberg.ProbLogit_n10 as non_linear_data
import Data.Stackelberg.MILPLogit_n10r100 as linear_data
import Data.Fixed_Point.ProbLogit_n10 as fixed_point_data

def heuristicIntensification(non_linear_data, step, capacity=[60.0, 6.0, 6.0]):
    ''' Call the heuristic game. Once the tables are completed, re-run the
        heuristic game on the region containing candidate solution only.
    '''

    print('\n--- HEURISTIC INTENSIFICATION METHOD ---\n')
    # Read the data file
    data = non_linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    non_linear_data.preprocess(data)
    # Define the keyword argument required to instanciate the heuristic game
    heu_keywords = {'K': 2,
                    'I': 2,
                    'operator': [0, 1, 2],
                    'optimizer': 1,
                    'endo_var': {0:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 0.0, 'step':1}},
                                1:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':step}},
                                2:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':step}}},
                    'seed': 1
                    }
    # Instanciate the heuristic game
    heu_game = nash_heuristic.NashHeuristic(**heu_keywords)
    # Update the original data
    data.update(heu_keywords)
    # Run the game
    heu_game.run(data)
    # Keep track of the bounds of the candidate solutions for each operator
    lower_bound_candidate_solutions = [[] for k in range(2)]
    upper_bound_candidate_solutions = [[] for k in range(2)]
    # Search for the candidate Nash eq. found by the heuristic game
    for (k, table) in enumerate(heu_game.tables):
        for (p_range, value) in zip(heu_game.mapping[k + 1], heu_game.tables[k]):
            if value != np.inf:
                # Candidate solution found
                lower_bound_candidate_solutions[((k+1)%2)].append(list(p_range.values())[0][0])
                upper_bound_candidate_solutions[((k+1)%2)].append(list(p_range.values())[0][1])
    # Define the bound of the
    lb = [min(prices) for prices in lower_bound_candidate_solutions]
    ub = [max(prices) for prices in upper_bound_candidate_solutions]

    #### Re-run the heuristic game on the pruned region
    # Read the data file
    data = non_linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    non_linear_data.preprocess(data)
    # Define the keyword argument required to instanciate the heuristic game
    operator = [0, 1, 2]
    heu_keywords = {'K': 2,
                    'I': 2,
                    'operator': operator,
                    'optimizer': 1,
                    'endo_var': {0:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 0.0, 'step':1}},
                                1:{'p':{'domain': 'C', 'lb': max(0.0, lb[operator[1] - 1]-0.1), 'ub': min(1.0, ub[operator[1] - 1]+0.1), 'step':step}},
                                2:{'p':{'domain': 'C', 'lb': max(0.0, lb[operator[2] - 1]-0.1), 'ub': min(1.0, ub[operator[2] - 1]+0.1), 'step':step}}},
                    'seed': 1
                    }
    # Instanciate the heuristic game
    heu_game = nash_heuristic.NashHeuristic(**heu_keywords)
    # Update the original data
    data.update(heu_keywords)
    # Run the game
    heu_game.run(data)

def heuristicToMILP(non_linear_data, linear_data, capacity=[60.0, 6.0, 6.0]):
    ''' Run the heuristic game on non_linear_data. Then, for each candidate solution
        found by the heuristic algorithm, run the MILP sequential game on linear_data
        starting from the corresponding configuration.
        Args:
            non_linear_data          data for the MINLP Stackelberg formulation
            linear_data              data for the MILP Stackelberg formulation
    '''

    print('\n--- HEURISTIC AND MILP SEQUENTIAL METHOD ---\n')
    print('\n---HEURISTIC GAME: ---\n')
    # Read the data file
    data = non_linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    non_linear_data.preprocess(data)
    # Define the keyword argument required to instanciate the heuristic game
    heu_keywords = {'K': 2,
                    'I': 2,
                    'operator': [0, 1, 2],
                    'optimizer': 1,
                    'endo_var': {0:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 0.0, 'step':1}},
                                1:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':20}},
                                2:{'p':{'domain': 'C', 'lb': 0.0, 'ub': 1.0, 'step':20}}},
                    'seed': 1
                    }
    # Instanciate the heuristic game
    heu_game = nash_heuristic.NashHeuristic(**heu_keywords)
    # Update the original data
    data.update(heu_keywords)
    # Run the game
    heu_game.run(data)
    # Initialize the candidate Nash equilibrium solutions
    candidate_solutions = []
    # Search for the candidate Nash eq. found by the heuristic game
    for (k, table) in enumerate(heu_game.tables):
        for (p_range, value) in zip(heu_game.mapping[k + 1], heu_game.tables[k]):
            if value != np.inf:
                # Candidate solution found
                candidate_solutions.append(list(p_range.values())[0])
                print('\n Candidate solution n°%r: Price of operator %r: %r.'
                       %(len(candidate_solutions), ((k+1)%2)+1, list(p_range.values())[0]))
                # Update the table to not detect the same candidate solution twice
                for index in range(len(heu_game.tables)):
                    np.place(heu_game.tables[index], heu_game.tables[index]==value, np.inf)

                # Run the MILP sequential game
                data = linear_data.getData()
                if capacity is None:
                    data.pop('capacity', None)
                else:
                    data['capacity'] = capacity
                linear_data.preprocess(data)
                # Compute the starting price vector of the sequential game
                if (k+1) == 1:
                    p_fixed = [0.0, -1.0, random.uniform(list(p_range.values())[0][0],
                                                         list(p_range.values())[0][1])]
                else:
                    p_fixed = [0.0, random.uniform(list(p_range.values())[0][0],
                                                   list(p_range.values())[0][1]), -1.0]
                #import IPython
                #IPython.embed()
                sequential_keywords = {'K': 2,
                                        'operator': [0, 1, 2],
                                        'max_iter': 50,
                                        'optimizer': k+1,
                                        'p_fixed': p_fixed,
                                        'y_fixed': [1.0, 1.0, 1.0]}
                seq_game = sequential_game.Sequential(**sequential_keywords)
                data.update(sequential_keywords)
                # Run the MILP sequential game
                print('\n--- MILP SEQUENTIAL GAME N°%r: ---\n' %(len(candidate_solutions)))
                seq_game.run(data, linearized=True)
                # Plot the graphs
                seq_game.plotGraphs('candidate_solution_%r' %(str(len(candidate_solutions))))

def MILPtoFixedPoint(linear_data, fixed_point_data, n_price_levels, capacity=[60.0, 6.0, 6.0]):
    ''' Run the MILP Sequential game. Once a cycle is detected, run the fixed point
        iterative method on the region containing the cycle

    '''
    print('\n--- MILP Sequential game to Fixed-point Iterative Method ---\n')

    # Run the MILP sequential game
    data = linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    linear_data.preprocess(data)
    # Compute the starting price vector of the sequential game
    sequential_keywords = { 'K': 2,
                            'operator': [0, 1, 2],
                            'max_iter': 50,
                            'optimizer': 1,
                            'p_fixed': [0.0, -1.0, 0.5],
                            'y_fixed': [1.0, 1.0, 1.0]}
    seq_game = sequential_game.Sequential(**sequential_keywords)
    data.update(sequential_keywords)
    # Run the MILP sequential game
    seq_game.run(data, linearized=True)

    # Nb of alternatives
    I = len(seq_game.p_history[0])
    # Keep track of the min/max prices in the cycle detected by the MILP seq. game
    lb = []
    ub = []
    for i in range(I):
        history = [p for p in seq_game.p_history[seq_game.cycle_iter:][:, i] if p != -1.0]
        lb.append(min(history))
        ub.append(max(history))

    # Get the fixed point data and preprocess it
    data = fixed_point_data.getData()
    # Update the fixed point data
    data['lb_p'] = lb
    data['ub_p'] = ub
    data['n_price_levels'] = n_price_levels
    #TODO: Make it possible not to add capacity constraints
    if capacity is None:
        data.pop('capacity', None)
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point_no_capacity.Fixed_Point(**data)
    else:
        data['capacity'] = capacity
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point.Fixed_Point(**data)

    model = fixed_point_game.getModel()
    print('\n MODEL COMPUTED')
    fixed_point_game.solveModel(model)

def cyclesToFixedPoint(cycles, fixed_point_data, n_price_levels, capacity=[60.0, 6.0, 6.0]):
    ''' Given cycles, construct a corresponding strategy set for the fixed-point model.
        Run the fixed-point model on this strategy set.
        Args:
            cycles            list containing the lowest and highest price of each altertative in each cycle [dict]
            fixed_point_data  data for the fixed-point model [dict]
            n_price_levels    number of strategy per alternative in the fixed-point model [int]
            capacity          capacity of each alternative [list]

                             # Cycle 1     # Cycle 2    # Cycle 3
        Example: cycles = [ [[0,1; 0,15], [0,5; 0.55], [0,7; 0,85]], # Prices of alternative 1
                            [[0,2; 0,25], [0,6; 0,65], [0,9; 0,95]] ] # Price of alternative 2
                  Each entry contains the extremes values of the prices in the corresponding cycle
    '''
    print('\nCycles: %r' %np.asarray(cycles))
    #### Get the fixed point data
    print('\nGet the data and initialize the Fixed point model')
    data = fixed_point_data.getData()
    # Preprocess the data based on capacity parameter
    if capacity is None:
        data.pop('capacity', None)
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point_no_capacity.Fixed_Point(**data)
    else:
        data['capacity'] = capacity
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point.Fixed_Point(**data)

    #### Merge the cycles ranges if they interesect
    print('\nConstruct the strategy set of the operators based on cycles')
    print('-Merge the cycles ranges')
    # Verify if any two cycles intersect
    # List containing the cycles to merge (price range intersect)
    to_merge = [[] for i in range(data['I'])]
    for i in range(data['I']):
        # Compare cycles pair-wise
        for index_1 in range(len(cycles[i]) - 1):
            for index_2 in range(index_1 + 1, len(cycles[i])):
                if ((cycles[i][index_1][0] <= cycles[i][index_2][1]) and
                   (cycles[i][index_1][0] >= cycles[i][index_2][0])):
                   # Lower bound of cycle index_1 included in cycle index_2
                   # The price range of cycle index_1 and index_2 need to be merged
                   to_merge[i].append([index_1, index_2])

                elif ((cycles[i][index_1][1] <= cycles[i][index_2][1]) and
                   (cycles[i][index_1][1] >= cycles[i][index_2][0])):
                   # Upper bound of cycle index_1 included in cycle index_2
                   # The price range of cycle index_1 and index_2 need to be merged
                   to_merge[i].append([index_1, index_2])

    # Map the initial cycle index with the merged cycle index
    cycles_indices = [{} for i in range(data['I'])]
    for i in range(data['I']):
        for index in range(len(cycles[i])):
            cycles_indices[i][index] = index

    # Construct merged cycle
    merged_cycles = copy.deepcopy(cycles)
    for i in range(data['I']):
        for indices in range(len(to_merge[i])):
            # Cycle index in cycle
            cycle_id_0 = to_merge[i][indices][0]
            cycle_id_1 = to_merge[i][indices][1]
            # Cycle index in merged cycle
            map_index_0 = cycles_indices[i][cycle_id_0]
            map_index_1 = cycles_indices[i][cycle_id_1]
            # Merge the two corresponding cycles
            merged_cycles[i][map_index_0] = [min(cycles[i][cycle_id_0][0], cycles[i][cycle_id_1][0]), \
                                            max(cycles[i][cycle_id_0][1], cycles[i][cycle_id_1][1])]
            # Delete the merged cycle
            del merged_cycles[i][map_index_1]
            # Update the index mapping between the cycles in cycles and merged_cycles
            # The map index of cycle_id_1 becomes map_index_0
            cycles_indices[i][cycle_id_1] = map_index_0
            # Since map_index_1 is deleted, decrement the map_index of the
            # cycle having a map_index higher than map_index_1
            for key in cycles_indices[i].keys():
                if cycles_indices[i][key] > map_index_1:
                    cycles_indices[i][key] -= 1
    print('\nMerged cycles: %r' %np.asarray(merged_cycles))

    #### Construct the strategy set of the operator
    print('\n-Discritize the merged cycles evenly')
    # Compute the total length of the price ranges for each alternative in merged_cycle
    distance = [[] for i in range(data['I'])]
    for i in range(data['I']):
        distance[i] = np.sum([abs(merged_cycles[i][index][1] - merged_cycles[i][index][0]) \
                              for index in range(len(merged_cycles[i]))])
    # Discretize the prices ranges in merged_cycles
    for i in range(data['I']):
        # Each strategy is evenly spread, the gap between them is step
        step = float(distance[i])/n_price_levels
        # Initialize the strategy set
        strategy_set = [merged_cycles[i][0][0]]
        carry = 0.0
        for cycle in merged_cycles[i]:
            # Length of the current cycle
            distance_cycle = cycle[1] - cycle[0] + carry
            # Number of strat to add in this cycle (round to avoid numerical errors)
            nb_strat = round(distance_cycle / step, 6)
            if nb_strat > 0:
                strategy_set.extend(list(np.linspace(cycle[0] - carry + step, cycle[1], nb_strat, endpoint=True)))
            # Remaining distance in the current cycle
            # Note that the Decimal library is needed to compute the remaining of a float division
            carry = int(round(Decimal(str(distance_cycle)) % Decimal(str(step)), 6))
        #TODO: Alternative : Each cycle is assigned a number of strategy (Motivation:
        # What if a cycle is huge and the other one tiny)
        # Delete the extra strategy
        del strategy_set[-1]
        strategy_set.sort()
        # i+1 not to change opt-out prices
        data['p'][i+1] = copy.deepcopy(strategy_set)
    print('\nStrategy sets: %r' %data['p'])

    #### Launch the Fixed-Point model
    print('\nLaunch the fixed point model')
    if capacity is None:
        data.pop('capacity', None)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point_no_capacity.Fixed_Point(**data)
    else:
        data['capacity'] = capacity
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point.Fixed_Point(**data)

    model = fixed_point_game.getModel()
    fixed_point_game.solveModel(model)

def fixedPointToStackelberg(linear_data, fixed_point_data, n_price_levels, capacity=[60.0, 6.0, 6.0]):
    ''' Run the fixed point model. Then, Stackelberg games with the linear model are
        launched, starting from the 'previous prices' decision variables found by the fixed point model. The goal
        is to check whether the optimal solution found by the fixed point model is a good
        approximation in case the strategy set is infinite.
        Args:
            linear_data       data for the Stackelberg game [dict]
            fixed_point_data  data for the fixed-point model [dict]
            n_price_levels    number of strategy per alternative in the fixed-point model [int]
            capacity          capacity of each alternative [list]
    '''

    #### Get the fixed point data
    print('\nGet the fixed point model data')
    data = fixed_point_data.getData()
    data['n_price_levels'] = n_price_levels
    # Preprocess the data based on capacity parameter
    if capacity is None:
        data.pop('capacity', None)
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point_no_capacity.Fixed_Point(**data)
    else:
        data['capacity'] = capacity
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point.Fixed_Point(**data)

    # Solve the model
    print('\nConstruct the fixed point model')
    model = fixed_point_game.getModel()
    print('\nSolve the fixed point model')
    fixed_point_game.solveModel(model)

    #### Keep track of the decision variables of the fixed point model
    # Price
    price_before = [0.0]
    for i in range(1, data['I'] + 1):
        price_before.append(model.solution.get_values('price[' + str(i) + ']'))

    price_after_fixed_point = [0.0]
    for i in range(1, data['I'] + 1):
        k = data['operator'][i]
        for l in range(n_price_levels):
            # Check which strategy the current operator picked, get the corresponding alternative price
            if model.solution.get_values('vAft[' + str(k) + ']' + '[' + str(l) + ']') == 1:
                price_after_fixed_point.append(data['p'][i, l])
    # Revenue
    revenue_after_fixed_point = []
    for k in range(1, data['K'] + 1):
        revenue_after_fixed_point.append(model.solution.get_values('linRevenueAft[' + str(k) + ']'))

    #### Run the Stackelberg games
    print('\nRun the Stackelberg games')
    # Keep track of the decision variables
    price_after_stackelberg = [0.0]
    revenue_after_stackelberg = [0.0 for k in range(data['K'] + 1)]
    # Get the data and add parameters to fix the price
    data = linear_data.getData()
    data['operator'] = [0, 1, 2]
    data['y_fixed'] = [1.0, 1.0, 1.0]
    # Launch a Stackelberg game for each alternative
    for i in range(1, data['I'] + 1):
        print('\nStackelberg game number %r' %i)
        data['optimizer'] = data['operator'][i]
        if i == 1:
            data['p_fixed'] = [0.0, -1.0, price_before[2]]
        elif i == 2:
            data['p_fixed'] = [0.0, price_before[1], -1.0]
        # Prepross the data, initialize the Stackelberg game and solve it
        linear_data.preprocess(data)
        stack_game = stackelberg_game.Stackelberg(**data)
        model = stack_game.getModel()
        stack_game.solveModel(model)
        # Keep track of the price obtained
        price_after_stackelberg.append(model.solution.get_values('p[' + str(i) + ']'))
        revenue_after_stackelberg[data['operator'][i]] = model.solution.get_objective_value()

    print('\n-------------------------------------------\n')
    ### Print the prices
    for i in range(data['I'] + 1):
        print('\nInitial price of alternative %r: %r' %(i, price_before[i]))
        print('After price of alternative %r (Fixed-point): %r' %(i, price_after_fixed_point[i]))
        print('After price of alternative %r (Stackelberg): %r' %(i, price_after_stackelberg[i]))

    ### Revenue of each operator
    for k in range(1, data['K'] + 1):
        print('\nAfter revenue of operator %r (Fixed-point): %r' %(k, revenue_after_fixed_point[k]))
        print('After revenue of operator %r (Stackelberg): %r' %(k, revenue_after_stackelberg[k]))

    ### Price the distance
    for i in range(data['I'] + 1):
        print('\nPrice difference of alternative %r (Fixed-point): %r' %(i, abs(price_before[i]-price_after_fixed_point[i])))
        print('Price difference of alternative %r (Stackelberg): %r' %(i, abs(price_before[i]-price_after_stackelberg[i])))

if __name__ == '__main__':
    '''
    # Run the heuristic intensification
    heuristicIntensification(non_linear_data, 20)
    # Run the heuristic towards MILP sequential method
    heuristicToMILP(non_linear_data, linear_data)
    # Run the MILP sequential method toward the fixed-point method
    MILPtoFixedPoint(linear_data, fixed_point_data, 20)
    cycles = [[[0.6, 0.7], [0.8, 0.9], [0.65, 0.85], [0.1, 0.2], [0.3, 0.4], [0.39, 0.45]],
              [[0.1, 0.2], [0.3, 0.4], [0.39, 0.45], [0.6, 0.7], [0.8, 0.9], [0.65, 0.85]]]
    cyclesToFixedPoint(cycles, fixed_point_data, 10, capacity=None)
    '''
    fixedPointToStackelberg(linear_data, fixed_point_data, 10, capacity=None)
