# Italian transport instance: Stackelberg game
import numpy as np

# This function gets information (travel time, arrival/departure) about different journeys that are offered to every customers.
def getInfo():
    ''' Construct a dictionary containing all information concerning
        alternatives between Milan and Rome
        The dictionnary contains departure, arrival times and travel time of each train, plane and IC
        Returns:
            dict          dictionarry with these information
    '''
    # Initialize the output dictionary
    dict = {}

    # ---------- PLANE ----------- #
    p_dep = np.array([    7*60,   7.5*60,     8*60, 8.333*60, 8.667*60,      9*60,     10*60, 11*60])
    p_arr = np.array([8.167*60, 8.667*60, 9.167*60,   9.5*60, 9.833*60, 10.167*60, 11.167*60, 12.167*60])
    p_tt = np.subtract(p_arr, p_dep)

    # ---------- AV ------------- #
    av_dep = np.array([6.133*60, 6.500*60, 7.000*60, 7.500*60, 8.000*60, 8.500*60, 9.000*60]) # 5.333*60
    av_arr = np.array([9.167*60, 9.483*60, 9.983*60,10.483*60,10.917*60,11.467*60,11.967*60]) # 8.617*60
    av_tt = np.subtract(av_arr, av_dep)

    # --------- NTV --------------- #
    ntv_dep = np.array([6.000*60, 6.417*60, 7.250*60, 7.583*60, 8.250*60, 8.750*60]) # 5.583*60
    ntv_arr = np.array([9.117*60, 9.850*60,10.200*60,11.017*60,11.217*60,12.017*60]) # 9.017*60
    ntv_tt = np.subtract(ntv_arr, ntv_dep)

    # -------------- IC ------------ #
    # This is a fake travel
    ic_dep = np.array([0])
    ic_arr = np.array([10.000*60])
    ic_tt = np.array([6*60])

    # ------------- Alternatives ----------- #
    # The first is an AV train, the second is an NTV train
    I_dep = np.array([5.333*60, 5.583*60])
    I_arr = np.array([8.617*60, 9.017*60])
    I_tt = np.subtract(I_arr, I_dep)

    # Travel times for all different modes
    dict['C_TTIME'] = 407.4 #TODO double check
    dict['IC_TTIME'] = ic_tt
    dict['P_TTIME'] = p_tt
    dict['AV_TTIME'] = av_tt
    dict['NTV_TTIME'] = ntv_tt
    dict['I_TTIME'] = I_tt

    # Departure times for all different modes
    dict['P_DEP'] = p_dep
    dict['IC_DEP'] = ic_dep
    dict['AV_DEP'] = av_dep
    dict['NTV_DEP'] = ntv_dep
    dict['I_DEP'] = I_dep

    # Arrival times for all different modes
    dict['P_ARR'] = p_arr
    dict['IC_ARR'] = ic_arr
    dict['AV_ARR'] = av_arr
    dict['NTV_ARR'] = ntv_arr
    dict['I_ARR']= I_arr

    # Number of alternatives of each mode. They are all opt-out options except the one in "I".
    dict['P_SIZE'] = np.prod(dict['P_ARR'].shape)
    dict['IC_SIZE'] = np.prod(dict['IC_ARR'].shape)
    dict['AV_SIZE'] = np.prod(dict['AV_ARR'].shape)
    dict['NTV_SIZE'] = np.prod(dict['NTV_ARR'].shape)
    dict['I_SIZE'] = np.prod(dict['I_ARR'].shape)

    ## Alternatives' features
    # Price of each mode
    # Every travel by plane is assumed to have same price. Same for IC/opt-out-AV/opt-out-NTV
    dict['PRICE_CAR'] = [104.79]
    dict['PRICE_PLANE'] = [153.0, 153.0, 153.0, 153.0, 153.0, 153.0, 153.0, 153.0] #153.0 * np.ones(dict['P_SIZE'])
    dict['PRICE_IC_1'] = [60.0] #60.0 * np.ones(dict['IC_SIZE'])
    dict['PRICE_IC_2'] = [30.0] #30.0 * np.ones(dict['IC_SIZE'])
    dict['PRICE_AV_1'] = [125.0, 125.0, 125.0, 125.0, 125.0, 125.0, 125.0] #125.0 * np.ones(dict['AV_SIZE'])
    dict['PRICE_AV_2'] = [ 80.0,  80.0,  80.0,  80.0,   80.0, 80.0,  80.0] #80.0 * np.ones(dict['AV_SIZE'])
    dict['PRICE_NTV_1'] = [105.0, 105.0, 105.0, 105.0, 105.0, 105.0] #105.0 * np.ones(dict['NTV_SIZE'])
    dict['PRICE_NTV_2'] = [ 60.0,  60.0,  60.0,  60.0,  60.0,  60.0] #60.0 * np.ones(dict['NTV_SIZE'])

    return dict

# This function reads the optimization problem and store its information
def getData():
    ''' Construct a dictionary containing all the data_file
        Returns:
            dict          dictionarry containing all the data
    '''
    # Initialize the output dictionary
    dict = {}

    # Store all train informations in the dictionary
    dict = getInfo()

    # Number of alternatives in the choice set (without considering opt-out)
    dict['I'] = 4
    dict['I_opt_out'] = 8

    # Number of customers
    dict['N'] = 1

    # Number of draws
    dict['R'] = 1

    # Random term (Gumbel distribution (0,1)) - 8 alternatives x 40 customers x 50 draws
    np.random.seed(1)
    #dict['xi'] = np.random.gumbel(size=(dict['I'] + dict['I_opt_out'], dict['N'], dict['R']))
    dict['xi'] = np.zeros((dict['I']+dict['I_opt_out'], dict['N'], dict['R']))

    #### Parameters of the utility function
    # Alternative specific coefficients
    dict['ASC_CAR'] = [0.0, 0.0]
    dict['ASC_PLANE'] = [1.09334736, -1.16793213]
    dict['ASC_IC_1'] =[-1.97906711, -1.2320878]
    dict['ASC_IC_2'] = [-1.93463076, -1.27745065]
    dict['ASC_AV_1'] = [0.94130557, -0.69902074] #0.94130557, -0.69902074
    dict['ASC_AV_2'] = [0.9218437, -0.76585163] # 0.9218437, -0.76585163
    dict['ASC_NTV_1'] = [-0.45324599, -1.09663822] #-0.45324599, -1.09663822
    dict['ASC_NTV_2'] = [-0.50465664, -1.17420312] #-0.50465664, -1.17420312

    # Constants
    dict['ASC_AV'] = [-0.462559, 0.098879]  #0.098879
    dict['ASC_HSR'] = [-0.536877, 0.010849] #0.010849
    dict['ASC_IC'] = [-0.474552, 0.184999]
    dict['ASC_NTV'] = [-0.074317, -0.088030]


    # Beta coefficients
    dict['BETA_TTIME'] = [-0.003920, -0.00745]
    dict['BETA_COST'] = [-0.031949, -0.010734]
    dict['BETA_ORIGIN'] = [0.352496, 0.343312]
    dict['BETA_EARLY'] = [-0.001744, -0.001764]
    dict['BETA_LATE'] = [-0.009349, -0.13561]

    ## Alternatives' features
    # Price of each mode
    # Every travel by plane is assumed to have same price. Same for IC/opt-out-AV/opt-out-NTV
    dict['PRICE_CAR'] = [104.79]
    dict['PRICE_PLANE'] = [153.0, 153.0, 153.0, 153.0, 153.0, 153.0, 153.0, 153.0] #153.0 * np.ones(dict['P_SIZE'])
    dict['PRICE_IC_1'] = [60.0] #60.0 * np.ones(dict['IC_SIZE'])
    dict['PRICE_IC_2'] = [30.0] #30.0 * np.ones(dict['IC_SIZE'])
    dict['PRICE_AV_1'] = [125.0, 125.0, 125.0, 125.0, 125.0, 125.0, 125.0] #125.0 * np.ones(dict['AV_SIZE'])
    dict['PRICE_AV_2'] = [ 80.0,  80.0,  80.0,  80.0,   80.0, 80.0,  80.0] #80.0 * np.ones(dict['AV_SIZE'])
    dict['PRICE_NTV_1'] = [105.0, 105.0, 105.0, 105.0, 105.0, 105.0] #105.0 * np.ones(dict['NTV_SIZE'])
    dict['PRICE_NTV_2'] = [ 60.0,  60.0,  60.0,  60.0,  60.0,  60.0] #60.0 * np.ones(dict['NTV_SIZE'])

    # Customer's socio-economic characteristics
    # Customer id          18  121  291  476  630  860 1061 1265 1550 2043 2192 2504 2840 3174 3339 3470 4017 4287 5073 5500  371  664  801  948 1058 1450 1466 1662 1712 2439 2478 2745 2963 3384 3456 3523 3567 4115 4119 4337
    dict['BUSINESS'] =   [1]
    dict['ORIGIN'] =     [1]

    dict['DAT'] = [546.0000]
    # Costs
    #                                   CAR   PLANE  IC_1   IC_2   AV_1   AV_2   NTV_1  NTV_2
    # Initial cost for each alternative
    dict['fixed_cost'] =    np.concatenate((np.zeros(4),
                                            150 * np.ones(1), 125 * np.ones(1),
                                            150 * np.ones(1), 125 * np.ones(1),
                                            np.array([0.0, 0.0, 0.0, 0.0])))
    assert(np.prod(dict['fixed_cost'].shape) == dict['I'] + dict['I_opt_out'])

    # Additional cost for each customer
    dict['customer_cost'] = np.concatenate((np.zeros(4),
                                            0 * np.ones(1), 0 * np.ones(1),
                                            0 * np.ones(1), 0 * np.ones(1),
                                            np.array([0.0, 0.0, 0.0, 0.0])))
    assert(np.prod(dict['customer_cost'].shape) == dict['I'] + dict['I_opt_out'])

    # Lower and upper bound on prices
    # The lower and upper bound are equal iff it is an opt-out
    # It is not necessary to have non-zero lb and ub for opt-out options since the endogenous coefficient is zero
    # for these alternatives.
    dict['lb_p'] = np.concatenate((np.array([0.0, 0.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 0.0, 0.0])))
    assert(np.prod(dict['lb_p'].shape) == dict['I'] + dict['I_opt_out'])

    dict['ub_p'] = np.concatenate((np.array([0.0, 0.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 0.0, 0.0]),
                                   np.array([200.0, 200.0, 200.0, 200.0])))
    assert(np.prod(dict['ub_p'].shape) == dict['I'] + dict['I_opt_out'])

    # Choice set of the customers
    # We assume that everyone can take any alternative
    dict['choice_set'] = np.full((dict['I'] + dict['I_opt_out'], dict['N']), 1)

    # Mapping between alternatives index and their names
    dict['name_mapping'] = {0: 'Car', 1: 'Plane', 2: 'IC_1st', 3: 'IC_2nd',
                            4: 'AV1', 5: 'AV2', 6: 'NTV1', 7: 'NTV1',
                            8: 'AV1_1', 9: 'AV2_1', 10: 'NTV1_1', 11: 'NTV1_1'}

    # Capacities -- Opt-out alternatives are always available
    dict['capacity'] = np.concatenate(( dict['N'] * np.ones(dict['I_opt_out']), np.array([1.0, dict['N'], 1.0, dict['N']])))
    assert(np.prod(dict['capacity'].shape) == dict['I'] + dict['I_opt_out'])
    return dict

def preprocess(dict):
    ''' Precomputation on the data in order to create the corresponding
        cplex model.
    '''

    # Precomputation of difference between desired and actual arrival time for each mode and each journey.
    # Start with initialization of arrays to store the early/late arrival
    # Recall that car has no early or late arrival
    p_early = np.empty([dict['P_SIZE'], dict['N']])
    p_late = np.empty([dict['P_SIZE'], dict['N']])
    ic_early = np.empty([dict['IC_SIZE'], dict['N']])
    ic_late = np.empty([dict['IC_SIZE'], dict['N']])
    av_early = np.empty([dict['AV_SIZE'], dict['N']])
    av_late = np.empty([dict['AV_SIZE'], dict['N']])
    ntv_early = np.empty([dict['NTV_SIZE'], dict['N']])
    ntv_late = np.empty([dict['NTV_SIZE'], dict['N']])
    I_early = np.empty([dict['I_SIZE'], dict['N']])
    I_late = np.empty([dict['I_SIZE'], dict['N']])

    # ----------------- PLANE ---------------- #
    for i in range(dict['P_SIZE']):
        for n in range(dict['N']):
            diff = dict['P_ARR'][i] - dict['DAT'][n]
            if( diff >= 0 ):
                p_early[i, n] = 0.0
                p_late[i, n] = diff
            else:
                p_early[i, n] = -diff # since diff is negative in this case
                p_late[i, n] = 0.0

    # ----------------- IC ---------------- #
    for i in range(dict['IC_SIZE']):
        for n in range(dict['N']):
            diff = dict['IC_ARR'][i] - dict['DAT'][n]
            if( diff >= 0):
                ic_early[i, n] = 0.0
                ic_late[i, n] = diff
            else:
                ic_early[i, n] = -diff # since diff is negative in this case
                ic_late[i, n] = 0.0

    # ---------------- AV ------------------ #
    for i in range(dict['AV_SIZE']):
        for n in range(dict['N']):
            diff = dict['AV_ARR'][i] - dict['DAT'][n]
            if( diff >= 0 ):
                av_early[i, n] = 0.0
                av_late[i, n] = diff
            else:
                av_early[i, n] = -diff # since diff is negative in this case
                av_late[i, n] = 0.0


    # ----------------- NTV ---------------- #
    for i in range(dict['NTV_SIZE']):
        for n in range(dict['N']):
            diff = dict['NTV_ARR'][i] - dict['DAT'][n]
            if( diff >= 0):
                ntv_early[i, n] = 0.0
                ntv_late[i, n] = diff
            else:
                ntv_early[i, n] = -diff # since diff is negative in this case
                ntv_late[i, n] = 0.0

    # ----------------- Alternatives ---------------- #
    for i in range(dict['I_SIZE']):
        for n in range(dict['N']):
            diff = dict['I_ARR'][i] - dict['DAT'][n]
            if( diff >= 0):
                I_early[i, n] = 0.0
                I_late[i, n] = diff
            else:
                I_early[i, n] = -diff # since diff is negative in this case
                I_late[i, n] = 0.0
    #TODO: could have been more compact.

    # Store early/late arrival for every train and every customer.
    dict['P_EARLY'] = p_early
    dict['P_LATE'] = p_late
    dict['AV_EARLY'] = av_early
    dict['AV_LATE'] = av_late
    dict['NTV_EARLY'] = ntv_early
    dict['NTV_LATE'] = ntv_late
    dict['IC_EARLY'] = ic_early
    dict['IC_LATE'] = ic_late
    dict['I_EARLY'] = I_early
    dict['I_LATE'] = I_late

    ########## Precomputation ##########
    # Priority list
    priority_list = np.empty([dict['I'] + dict['I_opt_out'], dict['N']])
    for i in range(dict['I'] + dict['I_opt_out']):
        min = 1
        max = dict['N']
        for n in range(dict['N']):
            if dict['choice_set'][i, n] == 1:
                priority_list[i, n] = min
                min += 1
            else:
                priority_list[i, n] = max
                max -= 1
    dict['priority_list'] = priority_list

    # Exogene utility
    exo_utility = (-np.inf) * np.ones([dict['I'] + dict['I_opt_out'], dict['N']])
    choice = np.empty([dict['I'] + dict['I_opt_out'], dict['N']]) # For each opt-out mode, selects the one with higher utility

    # Compute the deterministic utility of each alternative
    # TODO: Reduce I_opt_out! Do the same as for the plane loop (we don't need to store the deterministic utility of every AV-opt-out)
    for n in range(dict['N']):
        # default customer n is not business
        idx = 0
        if( dict['BUSINESS'][n] == 1 ):
            idx = 1
        for i in range(dict['I'] + dict['I_opt_out']):
            if( i == 0 ):
                # ------ CAR ------ #
                exo_utility[i, n] = (dict['ASC_CAR'][idx] +
                                     dict['BETA_TTIME'][idx] * dict['C_TTIME'] +
                                     dict['BETA_COST'][idx] * dict['PRICE_CAR'][0])
            elif( i == 1 ):
                # ----- PLANE ----- #
                for l in range(dict['P_SIZE']):
                    tmp = (dict['ASC_PLANE'][idx] +
                            dict['BETA_TTIME'][idx] * dict['P_TTIME'][l] +
                            dict['BETA_EARLY'][idx] * dict['P_EARLY'][l, n] + dict['BETA_LATE'][idx] * dict['P_LATE'][l, n] +
                            dict['BETA_COST'][idx] * dict['PRICE_PLANE'][l])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 2 ):
                # ------ IC 1 ------ #
                for l in range(dict['IC_SIZE']):
                    tmp = (dict['ASC_IC_1'][idx] + dict['ASC_IC'][idx] +
                            dict['BETA_TTIME'][idx] * dict['IC_TTIME'][l] +
                            dict['BETA_EARLY'][idx] * dict['IC_EARLY'][l, n] + dict['BETA_LATE'][idx] * dict['IC_LATE'][l, n] +
                            dict['BETA_COST'][idx] * dict['PRICE_IC_1'][l])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 3 ):
                # ------ IC 2 ------ #
                for l in range(dict['IC_SIZE']):
                    tmp = (dict['ASC_IC_2'][idx] + dict['ASC_IC'][idx] +
                            dict['BETA_TTIME'][idx] * dict['IC_TTIME'][l] +
                            dict['BETA_EARLY'][idx] * dict['IC_EARLY'][l, n] + dict['BETA_LATE'][idx] * dict['IC_LATE'][l, n] +
                            dict['BETA_COST'][idx] * dict['PRICE_IC_2'][l])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 4 ) :
                # ------ AV 1 ----- #
                for l in range(dict['AV_SIZE']):
                    tmp = (dict['ASC_AV_1'][idx] + dict['ASC_HSR'][idx] +
                           dict['ASC_AV'][idx] +
                           dict['BETA_TTIME'][idx] * dict['AV_TTIME'][l] +
                           dict['BETA_EARLY'][idx] * dict['AV_EARLY'][l, n] +
                           dict['BETA_LATE'][idx] * dict['AV_LATE'][l, n] +
                           dict['BETA_COST'][idx] * dict['PRICE_AV_1'][l] +
                           dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 5 ):
                # ------ AV 2 ----- #
                for l in range(dict['AV_SIZE']):
                    tmp = (dict['ASC_AV_2'][idx] + dict['ASC_HSR'][idx] +
                           dict['ASC_AV'][idx] +
                           dict['BETA_TTIME'][idx] * dict['AV_TTIME'][l] +
                           dict['BETA_EARLY'][idx] * dict['AV_EARLY'][l, n] +
                           dict['BETA_LATE'][idx] * dict['AV_LATE'][l, n] +
                           dict['BETA_COST'][idx] * dict['PRICE_AV_2'][l] +
                           dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 6 ):
                # ------ NTV 1 ----- #
                for l in range(dict['NTV_SIZE']):
                    tmp = (dict['ASC_NTV_1'][idx] + dict['ASC_NTV'][idx] +
                           dict['ASC_HSR'][idx] +
                           dict['BETA_TTIME'][idx] * dict['NTV_TTIME'][l] +
                           dict['BETA_EARLY'][idx] * dict['NTV_EARLY'][l, n] +
                           dict['BETA_LATE'][idx] * dict['NTV_LATE'][l, n] +
                           dict['BETA_COST'][idx] * dict['PRICE_NTV_1'][l] +
                           dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
                    # if( n == 0 ):
                    #     print(tmp, dict['NTV_TTIME'][l])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 7 ):
                # ------ NTV 2 ----- #
                for l in range(dict['NTV_SIZE']):
                    tmp = (dict['ASC_NTV_2'][idx] + dict['ASC_NTV'][idx] +
                           dict['ASC_HSR'][idx] +
                           dict['BETA_TTIME'][idx] * dict['NTV_TTIME'][l] +
                           dict['BETA_EARLY'][idx] * dict['NTV_EARLY'][l, n] +
                           dict['BETA_LATE'][idx] * dict['NTV_LATE'][l, n] +
                           dict['BETA_COST'][idx] * dict['PRICE_NTV_2'][l] +
                           dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 8 ):
                # ------ AV 1 ----- #
                exo_utility[i, n] = (dict['ASC_AV_1'][idx] + dict['ASC_HSR'][idx] +
                                    dict['ASC_AV'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][0] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][0, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][0, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 9 ):
                # ------ AV 2 ----- #
                exo_utility[i, n] = (dict['ASC_AV_2'][idx] + dict['ASC_HSR'][idx] +
                                    dict['ASC_AV'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][0] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][0, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][0, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 10 ):
                # ------ NTV 1 ----- #
                exo_utility[i, n] = (dict['ASC_NTV_1'][idx] + dict['ASC_NTV'][idx] +
                                    dict['ASC_HSR'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][1] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][1, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][1, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 11 ):
                # ------ NTV 2 ----- #
                exo_utility[i, n] = (dict['ASC_NTV_2'][idx] + dict['ASC_NTV'][idx] +
                                    dict['ASC_HSR'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][1] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][1, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][1, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            assert(exo_utility[i, n] != -np.inf)

    dict['exo_utility'] = exo_utility
    dict['CHOICE'] = choice

    # Beta coefficient for endogenous variables. Since the price is fixed for op-out option, the B_COST * PRICE is already
    # in the deterministic utility.
    # TODO: reduce size of endo_coef (difficult due to the formulation of the NLP)
    endo_coef = np.full([dict['I'] + dict['I_opt_out'], dict['N']], 0.0)
    for n in range(dict['N']):
        idx = int(dict['BUSINESS'][n])
        endo_coef[dict['I_opt_out']:, n] = dict['BETA_COST'][idx]

    dict['endo_coef'] = endo_coef

    # Calculate bounds on the utility
    lb_U = np.empty([dict['I'] + dict['I_opt_out'], dict['N'], dict['R']])
    ub_U = np.empty([dict['I'] + dict['I_opt_out'], dict['N'], dict['R']])
    lb_Umin = np.full((dict['N'], dict['R']), np.inf)
    ub_Umax = np.full((dict['N'], dict['R']), -np.inf)
    M = np.empty([dict['N'], dict['R']])
    for n in range(dict['N']):
        for r in range(dict['R']):
            for i in range(dict['I'] + dict['I_opt_out']):
                    if dict['endo_coef'][i, n] > 0:
                        lb_U[i, n, r] = (dict['endo_coef'][i, n] * dict['lb_p'][i] +
                                        dict['exo_utility'][i, n] + dict['xi'][i, n, r])
                        ub_U[i, n, r] = (dict['endo_coef'][i, n] * dict['ub_p'][i] +
                                        dict['exo_utility'][i, n] + dict['xi'][i, n, r])
                    else:
                        lb_U[i, n, r] = (dict['endo_coef'][i, n] * dict['ub_p'][i] +
                                        dict['exo_utility'][i, n] + dict['xi'][i, n, r])
                        ub_U[i, n, r] = (dict['endo_coef'][i, n] * dict['lb_p'][i] +
                                        dict['exo_utility'][i, n] + dict['xi'][i, n, r])
                    assert(lb_U[i, n, r] <= ub_U[i, n, r])
                    # Bound for each customer, for each draw
                    if lb_U[i, n, r] < lb_Umin[n, r]:
                        lb_Umin[n, r] = lb_U[i, n, r]
                    if ub_U[i, n, r] > ub_Umax[n, r]:
                        ub_Umax[n, r] = ub_U[i, n, r]

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    dict['lb_U'] = lb_U
    dict['ub_U'] = ub_U
    dict['lb_Umin'] = lb_Umin
    dict['ub_Umax'] = ub_Umax
    dict['M'] = M

    dict['lb_U'] = lb_U
    dict['ub_U'] = ub_U

# Sanity check function to see the utility of a customer.
# TODO: modify it once I_opt_out is reduced
def customer_info(dict, n):
    print('****-------------- Informations about Customer:', n, '------------------------****')
    print('** Basic informations: **\nBusiness customer', dict['BUSINESS'][n])
    print('Coming from the city', dict['ORIGIN'][n])
    print('Desired arrival time is at:', dict['DAT'][n] / 60)
    p = int(dict['CHOICE'][1, n])
    ic1 = int(dict['CHOICE'][2, n])
    ic2 = int(dict['CHOICE'][3, n])
    av1 = int(dict['CHOICE'][4, n])
    av2 = int(dict['CHOICE'][5, n])
    assert( av1 == av2 )
    ntv1 = int(dict['CHOICE'][6, n])
    ntv2 = int(dict['CHOICE'][7, n])
    assert( ntv1 == ntv2 )

    # print opt-out informations #
    print('** Trip informations: **\n\n****** Car: *******\nTakes car which cost', dict['PRICE_CAR'], '/ utility:', dict['exo_utility'][0, n])
    print('****** Plane: *******\nTakes plane number:', p, 'at price', dict['PRICE_PLANE'],
          'which leaves at', dict['P_DEP'][p] / 60, 'and arrives at', dict['P_ARR'][p] / 60, '/ utility:', dict['exo_utility'][1, n])
    print('****** IC1 Train: *******\nTakes IC 1st train number:', ic1, 'at price', dict['PRICE_IC_1'],
          'which leaves at:', dict['IC_DEP'][ic1] / 60,
          'and arrives at:', dict['IC_ARR'][ic1] / 60, '/ utility:', dict['exo_utility'][2, n])
    print('****** IC2 Train: *******\nTakes IC 2nd train number:', ic2, 'at price', dict['PRICE_IC_2'],
          'which leaves at:', dict['IC_DEP'][ic2] / 60,
          'and arrives at:', dict['IC_ARR'][ic2] / 60, '/ utility:', dict['exo_utility'][3, n])
    print('****** AV1 Train: *******\nTakes AV 1st train number', av1, 'at price', dict['PRICE_AV_1'][av1], 'which leaves at',
         dict['AV_DEP'][av1] / 60, 'and arrives at', dict['AV_ARR'][av1] / 60, 'total travel time is', dict['AV_TTIME'][av1] / 60, '\n/ utility:',
         dict['exo_utility'][4, n])
    print('****** AV2 Train: *******\nTakes AV 2nd train number', av2, 'at price', dict['PRICE_AV_2'][av2], 'which leaves at',
         dict['AV_DEP'][av2] / 60, 'and arrives at', dict['AV_ARR'][av2] / 60, 'total travel time is', dict['AV_TTIME'][av2] / 60, '\n/ utility:',
         dict['exo_utility'][5, n])
    print('****** NTV1 Train: *******\nTakes NTV 1st number', ntv1, 'at price', dict['PRICE_NTV_1'][ntv1], 'which leaves at',
         dict['NTV_DEP'][ntv1] / 60, 'and arrives at', dict['NTV_ARR'][ntv1] / 60, 'total travel time is', dict['NTV_TTIME'][ntv1] / 60, '\n/ utility:',
         dict['exo_utility'][6, n])
    print('****** NTV2 Train: *******\nTakes NTV 2nd number', ntv2, 'at price', dict['PRICE_NTV_2'][ntv2], 'which leaves at',
         dict['NTV_DEP'][ntv2] / 60, 'and arrives at', dict['NTV_ARR'][ntv2] / 60, 'total travel time is', dict['NTV_TTIME'][ntv2] / 60, '\n/ utility:',
         dict['exo_utility'][7, n])

    # print alternative informations #
    print('\nIf he takes the first AV1 train at price', dict['PRICE_AV_1'][0], 'which leaves at',
          dict['I_DEP'][0] / 60, 'and arrives at', dict['I_ARR'][0] / 60, 'total travel time is', dict['I_TTIME'][0] / 60, '\n/ utility:',
          dict['exo_utility'][8, n] + dict['endo_coef'][8, n] * dict['PRICE_AV_1'][0])
    print('If he takes the first AV2 train at price', dict['PRICE_AV_2'][0], 'which leaves at',
          dict['I_DEP'][0] / 60, 'and arrives at', dict['I_ARR'][0] / 60, 'total travel time is', dict['I_TTIME'][0] / 60, '\n/ utility:',
          dict['exo_utility'][9, n] + dict['endo_coef'][9, n] * dict['PRICE_AV_2'][0])
    print('If he takes the first NTV1 train at price', dict['PRICE_NTV_1'][0], 'which leaves at',
          dict['I_DEP'][1] / 60, 'and arrives at', dict['I_ARR'][1] / 60, 'total travel time is', dict['I_TTIME'][1] / 60, '\n/ utility:',
          dict['exo_utility'][10, n] + dict['endo_coef'][10, n] * dict['PRICE_NTV_1'][0])
    print('If he takes the first NTV2 train at price', dict['PRICE_NTV_2'][0], 'which leaves at',
          dict['I_DEP'][1] / 60, 'and arrives at', dict['I_ARR'][1] / 60, 'total travel time is', dict['I_TTIME'][1] / 60, '\n/ utility:',
          dict['exo_utility'][11, n] + dict['endo_coef'][11, n] * dict['PRICE_NTV_2'][0])


if __name__ == '__main__':
    dict = getData()
    preprocess(dict)
    customer_info(dict, 0)