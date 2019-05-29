# Python script containing several methods connecting the different blocks
# of the algorithmic framework

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
import Data.Italian.Non_linear_Stackelberg.ProbLogit_n40 as non_linear_data
import Data.Italian.Stackelberg.MILPLogit_n20r50 as linear_data
import Data.Italian.Fixed_Point.ProbLogit_n20r50 as fixed_point_data

def heuristicIntensification(non_linear_data, heu_game_keywords, step, capacity=[60.0, 6.0, 6.0]):
    ''' 2-stage method. In the first stage, the heuristic game is called.
        In the second stage, the domain of the endogene variable is pruned and
        the heuristic game is called. The pruned region contains the candidate solution
        identified in the first stage.
        Args:
            non_linear_data          data for the MINLP Stackelberg formulation [dict]
            heu_game_keywords        keyword argument to instanciate the heuristic game [dict]
            step                     number of price subset in the second stage [int]
            capacity                 capacity of each alternative [list]
    '''
    print('\n--- Heuristic Intensification Method ---\n')
    print('\nStage 1: 1st Run the Heuristic game')
    # Read the data file
    data = non_linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    # Preprocess the data
    non_linear_data.preprocess(data)
    # Update the original data
    data.update(heu_game_keywords)
    # Instanciate the heuristic game
    heu_game = nash_heuristic.NashHeuristic(**data)
    #### Run the heuristic game
    heu_game.run(data)
    # Keep track of the bounds of the endogene variables for each alternative of each operator
    lower_bound_candidate_solutions = {}
    upper_bound_candidate_solutions = {}
    for i in range(heu_game.I_opt_out, heu_game.I_opt_out + heu_game.I):
        lower_bound_candidate_solutions[i] = []
        upper_bound_candidate_solutions[i] = []
    # Search for the candidate Nash eq. and cycles found by the heuristic game
    for (k, table) in enumerate(heu_game.tables):
        # Map the current table with the strategy of the competitor
        for (p_range, value) in zip(heu_game.mapping[((k+1)%2) + 1], table):
            if value != np.inf:
                # Candidate solution found
                for i in p_range.keys():
                    lower_bound_candidate_solutions[i].append(p_range[i][0])
                    upper_bound_candidate_solutions[i].append(p_range[i][1])
    # Compute the lower/upper bound of the endogenous variable among all the candidate solution
    # These bound will be the ones used in the next run of the heuristic game
    for i in range(heu_game.I_opt_out, heu_game.I_opt_out + heu_game.I):
        lower_bound_candidate_solutions[i] = min(lower_bound_candidate_solutions[i])
        upper_bound_candidate_solutions[i] = max(upper_bound_candidate_solutions[i])

    #### Re-run the heuristic game on the pruned region
    print('\nStage 2: 2nd Run the Heuristic game')
    # Read the data file
    data = non_linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    # Adjust the price lb and ub, and the number of strategies per alternative
    for i in range(heu_game.I_opt_out, heu_game.I_opt_out + heu_game.I):
        data['lb_p'][i] = lower_bound_candidate_solutions[i]
        data['ub_p'][i] = upper_bound_candidate_solutions[i]
        heu_game_keywords['endo_var'][i]['p']['lb'] = lower_bound_candidate_solutions[i]
        heu_game_keywords['endo_var'][i]['p']['ub'] = upper_bound_candidate_solutions[i]
        heu_game_keywords['endo_var'][i]['p']['step'] = step

    # Preprocess the data
    non_linear_data.preprocess(data)
    # Update the original data
    data.update(heu_game_keywords)
    # Instanciate the heuristic game
    heu_game = nash_heuristic.NashHeuristic(**data)
    # Run the game
    heu_game.run(data)

def heuristicToMILP(non_linear_data, linear_data, heu_game_keywords, capacity=[60.0, 6.0, 6.0]):
    ''' Run the heuristic game. Then, run the MILP sequential game
        starting from the each candidate solution.
        The graphs of each MILP sequential game are generated and saved.
        Args:
            non_linear_data          data for the MINLP Stackelberg formulation [dict]
            linear_data              data for the MILP Stackelberg formulation [dict]
            capacity                 capacity of each alternative [list]
    '''
    print('\n--- Heuristic to MILP Sequential Game Method ---\n')
    print('\nRun the Heuristic game')
    # Read the data file
    data = non_linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    # Preprocess the data
    non_linear_data.preprocess(data)
    # Update the original data
    data.update(heu_game_keywords)
    # Instanciate the heuristic game
    heu_game = nash_heuristic.NashHeuristic(**data)
    # Run the game
    heu_game.run(data)
    # Initialize the candidate Nash equilibrium solutions
    candidate_solutions = []
    # Search for the candidate Nash eq. and cycles found by the heuristic game
    for (k, table) in enumerate(heu_game.tables):
        for (p_range, value) in zip(heu_game.mapping[((k+1)%2) + 1], heu_game.tables[k]):
            if value != np.inf:
                # Candidate solution found
                candidate_solutions.append(p_range)
                print('\n Candidate solution n°%r: Price of operator %r: %r.'
                       %(len(candidate_solutions), ((k+1)%2)+1, p_range))
                # Update the table to not detect the same candidate solution twice ?
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
                p_fixed = copy.deepcopy(data['lb_p'])
                for i in range(self.I_opt_out, self.I + self.I_opt_out):
                    if i in p_range.keys():
                        p_fixed[i] = random.uniform(p_range[i][0],
                                                    p_range[i][1])
                    else:
                        p_fixed[i] = -1.0

                sequential_keywords = {'K': heu_game_keywords['K'],
                                       'operator': heu_game_keywords['operator'],
                                       'max_iter': 50,
                                       'optimizer': k+1,
                                       'p_fixed': p_fixed,
                                       'y_fixed': np.full((data['I'] + data['I_opt_out']), 1.0)}
                seq_game = sequential_game.Sequential(**sequential_keywords)
                data.update(sequential_keywords)
                # Run the MILP sequential game
                print('\n--- MILP Sequential Game n°%r: ---\n' %(len(candidate_solutions)))
                seq_game.run(data, linearized=True)
                # Plot the graphs
                seq_game.plotGraphs('candidate_solution_%r' %(str(len(candidate_solutions))))

def MILPtoFixedPoint(linear_data, fixed_point_data, seq_game_keywords, n_price_levels, capacity=[60.0, 6.0, 6.0]):
    ''' Run the MILP Sequential game. Once a cycle is detected, run the fixed point
        iterative method on the region containing the cycle.
        Args:
            linear_data              data for the MILP Stackelberg formulation [dict]
            fixed_point_data         data for the fixed-point model [dict]
            n_price_levels           number of strategy per alternative in the fixed-point model [int]
            capacity                 capacity of each alternative [list]

    '''
    print('\n--- MILP Sequential game to Fixed-point Iterative Method ---\n')
    print('\nRun the MILP Sequential game')
    # Run the MILP sequential game
    data = linear_data.getData()
    if capacity is None:
        data.pop('capacity', None)
    else:
        data['capacity'] = capacity
    # Construct the initial prices for the opt-out alternatives
    for i in range(data['I_opt_out']):
        seq_game_keywords['p_fixed'][i] = data['lb_p'][i]
    # Preprocess the data
    linear_data.preprocess(data)
    # Instanciate the sequential game
    seq_game = sequential_game.Sequential(**seq_game_keywords)
    # Update the original data
    data.update(seq_game_keywords)
    # Run the MILP sequential game
    seq_game.run(data, linearized=True)

    # Keep track of the lower and upper bound on the prices
    # in the cycle detected by the MILP seq. game
    lb = []
    ub = []
    for i in range(data['I'] + data['I_opt_out']):
        history = [p for p in seq_game.p_history[seq_game.cycle_iter:][:, i] if p != -1.0]
        lb.append(min(history))
        ub.append(max(history))

    print('\nRun the Fixed-point Iterative Method')
    t0 = time.time()
    # Get the fixed point data
    data = fixed_point_data.getData()
    # Update the fixed point data price
    data['lb_p'] = lb
    data['ub_p'] = ub
    data['n_price_levels'] = n_price_levels
    if capacity is None:
        data.pop('capacity', None)
        # Preprocess the data
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point_no_capacity.Fixed_Point(**data)
    else:
        data['capacity'] = capacity
        # Preprocess the data
        fixed_point_data.preprocess(data)
        # Instanciate a Fixed point method game and solve it
        fixed_point_game = fixed_point.Fixed_Point(**data)

    for i in range(data['I'] + data['I_opt_out']):
        print('Strategy set of the alternative %r: %r' %(i, data['p'][i, :]))
    model = fixed_point_game.getModel()
    fixed_point_game.solveModel(model)
    print('Running time of the Fixed-Point MIP: %r' %(time.time() -t0))

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
    print('\n--- Cycles to Fixed Point Model Method ---\n')
    # Print the initial cycles
    print('\nCycles: %r' %np.asarray(cycles))

    #### Get the fixed point data
    print('\nGet the data and initialize the Fixed point model')
    data = fixed_point_data.getData()
    # Preprocess the data based on the capacity parameter
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
    # List containing the cycles id to merge (price range intersect)
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

    # Map the indices of the cycles in cycles to the indices of the cycles in merged_cycles
    cycles_indices = [{} for i in range(data['I'])]
    for i in range(data['I']):
        for index in range(len(cycles[i])):
            # Initially no merged has been made
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
            # cycles having a map_index higher than map_index_1
            # Note that map_index_0 is smaller than map_index_1
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
        # Compute the contribution of each cycle to the total price range length
        distribution = [(cycle[1] - cycle[0])/distance[i] \
                            for cycle in merged_cycles[i]]
        # Initialize the number of strategies per cycle
        # Round to the lowest integer the contribution of each cycle
        nb_strat = [int(max(math.floor(n_price_levels*distr), 1)) \
                            for distr in distribution]
        # For all the remaining strategy to add, pick the cycles with the highest carry (based on length)
        carry = [(n_price_levels*distr) % 1 for distr in distribution]
        for step in range(sum(nb_strat), n_price_levels):
            highest = np.argmax(carry)
            nb_strat[highest] += 1
            # Discard the cycle for the remaining strategy to add
            carry[highest] = 0

        # Compute the strategies for each cycle based on nb_strat
        strategy_set = []
        for (index, cycle) in enumerate(merged_cycles[i]):
            strategy_set.extend(list(np.linspace(cycle[0], cycle[1], num=nb_strat[index])))
        # Sort the strategy in increasing order
        strategy_set.sort()
        # i+I_opt_out not to change opt-out prices
        data['p'][i + data['I_opt_out']] = copy.deepcopy(strategy_set)
    # The following commented lines is another method to compute the strategy set
    '''
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
            nb_strat = round(distance_cycle/step, 6)
            if nb_strat > 0:
                strategy_set.extend(list(np.linspace(cycle[0] - carry + step, cycle[1], nb_strat, endpoint=True)))
            # Remaining distance in the current cycle
            # Note that the Decimal library is needed to compute the remaining of a float division
            carry = int(round(Decimal(str(distance_cycle)) % Decimal(str(step)), 6))
        #TODO: Alternative : Each cycle is assigned a number of strategy (Motivation:
        # What if a cycle is huge and the other one tiny)
        # Delete the extra strategy
        del strategy_set[-1]
        # Note that it is crucial to sort the strategy set for the preprocessing
        strategy_set.sort()
        # i+1 not to change opt-out prices
        data['p'][i + data['I_opt_out']] = copy.deepcopy(strategy_set)
    '''

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

def fixedPointToStackelberg(linear_data, fixed_point_data, n_price_levels, price_tol, revenue_tol, lb_p, ub_p, capacity=[60.0, 6.0, 6.0]):
    ''' Run the fixed point model. Then, Stackelberg games with the linear model are
        launched, starting from the 'previous prices' decision variables found by the fixed point model. The goal
        is to check whether the optimal solution found by the fixed point model is a good
        approximation in case the strategy set is infinite.
        While both tolerances are not satisfied, the fixed point model is relaunch with more strategies.
        Args:
            linear_data       data for the Stackelberg game [dict]
            fixed_point_data  data for the fixed-point model [dict]
            n_price_levels    number of strategy per alternative in the fixed-point model [int]
            price_tol         price tolerance between the fixed point model and stackelberg game [float]
            revenue_tol       revenue difference between the fixed point model and stackelberg game [float]
            lb_p              price lower bound for each alternative [list]
            ub_p              price upper bound for each alternative [list]
            capacity          capacity of each alternative [list]
    '''
    print('\n--- Fixed Point Model to Stackelberg Game Method ---\n')
    #### Get the fixed point data
    print('\nGet the fixed point model data')
    fp_data = fixed_point_data.getData()
    # Initial number of strategy per operator
    fp_data['n_price_levels'] = n_price_levels
    # Set the price lower and upper bound
    fp_data['lb_p'] = copy.deepcopy(lb_p)
    fp_data['ub_p'] = copy.deepcopy(ub_p)
    # While the fixed point model is not accurate enough, add more strategies
    # and re-launch the fixed-point model
    accurate = False
    while accurate is False:
        # Preprocess the data based on capacity parameter
        if capacity is None:
            fp_data.pop('capacity', None)
            fixed_point_data.preprocess(fp_data)
            # Instanciate a Fixed point method game and solve it
            fixed_point_game = fixed_point_no_capacity.Fixed_Point(**fp_data)
        else:
            fp_data['capacity'] = capacity
            fixed_point_data.preprocess(fp_data)
            # Instanciate a Fixed point method game and solve it
            fixed_point_game = fixed_point.Fixed_Point(**fp_data)

        # Solve the model
        print('\nConstruct the fixed point model. Number of strategy per alternative : %r' %fp_data['n_price_levels'])
        model = fixed_point_game.getModel()
        print('\nSolve the fixed point model')
        fixed_point_game.solveModel(model)

        #### Keep track of the decision variables of the fixed point model
        # Price
        price_before = []
        for i in range(fp_data['I'] + fp_data['I_opt_out']):
            if i < fp_data['I_opt_out']:
                price_before.append(fp_data['lb_p'][i])
            else:
                price_before.append(model.solution.get_values('price[' + str(i) + ']'))

        price_after_fixed_point = []
        for i in range(fp_data['I'] + fp_data['I_opt_out']):
            if i < fp_data['I_opt_out']:
                price_after_fixed_point.append(price_before[i])
            else:
                k = fp_data['operator'][i]
                for l in range(fp_data['n_price_levels']):
                    # Check which strategy the current operator picked, get the corresponding alternative price
                    if model.solution.get_values('vAft[' + str(k) + ']' + '[' + str(i) + ']' + '[' + str(l) + ']') == 1:
                        price_after_fixed_point.append(fp_data['p'][i, l])
        # Revenue
        revenue_after_fixed_point = [0.0]
        for k in range(1, fp_data['K'] + 1):
            revenue_after_fixed_point.append(model.solution.get_values('revenueAftMax[' + str(k) + ']'))

        #### Run the Stackelberg games
        print('\nRun the Stackelberg games')
        # Keep track of the decision variables
        price_after_stackelberg = copy.deepcopy(price_before)
        revenue_after_stackelberg = [0.0 for k in range(fp_data['K'] + 1)]
        # Get the data and construct the keyword arguments for the stackelberg game
        l_data = linear_data.getData()
        # Set the price lower and upper bound
        l_data['lb_p'] = copy.deepcopy(lb_p)
        l_data['ub_p'] = copy.deepcopy(ub_p)
        # Add keyword arguments to fix the price of an operator
        l_data['operator'] = copy.deepcopy(fp_data['operator'])
        l_data['y_fixed'] = np.full((l_data['I'] + l_data['I_opt_out']), 1.0)
        # Launch a Stackelberg game for each operator
        for k in range(1, fp_data['K'] + 1):
            print('\nStackelberg game number %r' %k)
            l_data['optimizer'] = k
            # Fix the prices
            l_data['p_fixed'] = copy.deepcopy(l_data['lb_p'])
            for i in range(l_data['I_opt_out'], (l_data['I'] + l_data['I_opt_out'])):
                if l_data['operator'][k] == k:
                    l_data['p_fixed'][i] = -1.0
                else:
                    l_data['p_fixed'][i] = price_before[i]

            # Prepross the data, initialize the Stackelberg game and solve it
            linear_data.preprocess(l_data)
            stack_game = stackelberg_game.Stackelberg(**l_data)
            model = stack_game.getModel()
            stack_game.solveModel(model)
            # Keep track of the price and revenue obtained
            for i in [i for i, ope in enumerate(l_data['operator']) if ope == k]:
                price_after_stackelberg[i] = model.solution.get_values('p[' + str(i) + ']')
            revenue_after_stackelberg[k] = model.solution.get_objective_value()

        print('\n-------------------------------------------\n')
        ### Print the prices
        for i in range(l_data['I'] + l_data['I_opt_out']):
            print('\nInitial price of alternative %r: %r' %(i, price_before[i]))
            print('After price of alternative %r (Fixed-point): %r' %(i, price_after_fixed_point[i]))
            print('After price of alternative %r (Stackelberg): %r' %(i, price_after_stackelberg[i]))

        ### Print the price difference between the fixed point and Stackelberg game
        for i in range(l_data['I'] + l_data['I_opt_out']):
            print('\nPrice difference of alternative %r (Fixed-point): %r' %(i, abs(price_before[i]-price_after_fixed_point[i])))
            print('Price difference of alternative %r (Stackelberg): %r' %(i, abs(price_before[i]-price_after_stackelberg[i])))

        ### Print the revenue of each operator
        for k in range(1, fp_data['K'] + 1):
            print('\nAfter revenue of operator %r (Fixed-point): %r' %(k, revenue_after_fixed_point[k]))
            print('After revenue of operator %r (Stackelberg): %r' %(k, revenue_after_stackelberg[k]))

        ### Check whether the price and revenue differences are small enough.
        # If at least one of them satify its tolerance threshold, the solution is considered accurate
        price_diff = [abs(price_after_fixed_point[i] - price_after_stackelberg[i]) for i in range(l_data['I'] + l_data['I_opt_out'])]
        revenue_diff = [abs(revenue_after_fixed_point[k] - revenue_after_stackelberg[k]) for k in range(fp_data['K'] + 1)]
        if all(entry < price_tol for entry in price_diff):
            accurate = True
            print('\nPrice tolerance satisfied. The fixed point model is accurate enough.')
        elif all(entry < revenue_tol for entry in revenue_diff):
            accurate = True
            print('Revenue tolerance satisfied. The fixed point model is accurate enough.')
        else:
            # Increase the number of strategies for the next run
            fp_data['n_price_levels'] = int(math.ceil(fp_data['n_price_levels']*1.5))

if __name__ == '__main__':
    # Data for the heuristic game
    nl_data = non_linear_data.getData()
    heu_keywords = {'K': 2,
                'operator': [0, 0, 0, 0, 1, 1, 2, 2],
                'optimizer': 1,
                'endo_var': {0:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][0], 'ub': nl_data['ub_p'][0], 'step':1}},
                             1:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][1], 'ub': nl_data['ub_p'][1], 'step':1}},
                             2:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][2], 'ub': nl_data['ub_p'][2], 'step':1}},
                             3:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][3], 'ub': nl_data['ub_p'][3], 'step':1}},
                             4:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][4], 'ub': nl_data['ub_p'][4], 'step':10}},
                             5:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][5], 'ub': nl_data['ub_p'][5], 'step':10}},
                             6:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][6], 'ub': nl_data['ub_p'][6], 'step':10}},
                             7:{'p':{'domain': 'C', 'lb': nl_data['lb_p'][7], 'ub': nl_data['ub_p'][7], 'step':10}}},
                'seed': 1
                }
    # Data for the sequential game
    seq_keywords = {'K': 2,
                    'operator': [0, 0, 0, 0, 1, 1, 2, 2],
                    'max_iter': 50,
                    'optimizer': 1,
                    'p_fixed': [-1, -1, -1, -1, -1, -1, 50.0, 50.0],
                    'y_fixed': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}

    # Run the heuristic intensification
    heuristicIntensification(non_linear_data, heu_keywords, 10, capacity=None)
    '''
    # Run the heuristic towards MILP sequential method
    heuristicToMILP(non_linear_data, linear_data, heu_keywords, capacity=None)
    # Run the MILP sequential method toward the fixed-point model method
    MILPtoFixedPoint(linear_data, fixed_point_data, seq_keywords, 20, capacity=None)
    # Run the cyles to Fixed point model method
    cycles = [[[0.6, 0.7], [0.8, 0.9], [0.65, 0.85], [0.1, 0.2], [0.3, 0.4], [0.39, 0.45]],
              [[0.1, 0.2], [0.3, 0.4], [0.39, 0.45], [0.6, 0.7], [0.8, 0.9], [0.65, 0.85]]]
    cyclesToFixedPoint(cycles, fixed_point_data, 10, capacity=None)
    # Run the fixed point towards Stackelberg game method
    fixedPointToStackelberg(linear_data, fixed_point_data, 10, 0.02, 0.02, [0.0, 0.0, 0.0], [0.0, 1.0, 1.0], capacity=None)
    '''
