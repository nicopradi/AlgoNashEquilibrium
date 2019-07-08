# Italian transport instance: Stackelberg game
import numpy as np

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
    ic_dep = np.array([0])
    ic_arr = np.array([10.000*60])
    ic_tt = np.array([6*60])

    # ------------- Alternatives ----------- #
    I_dep = np.array([5.333*60, 5.583*60])
    I_arr = np.array([8.617*60, 9.017*60])
    I_tt = np.subtract(I_arr, I_dep)

    dict['C_TTIME'] = 407.4 #TODO double check
    dict['P_DEP'] = p_dep
    dict['P_ARR'] = p_arr
    dict['P_TTIME'] = p_tt
    dict['AV_DEP'] = av_dep
    dict['AV_ARR'] = av_arr
    dict['AV_TTIME'] = av_tt
    dict['NTV_DEP'] = ntv_dep
    dict['NTV_ARR'] = ntv_arr
    dict['NTV_TTIME'] = ntv_tt
    dict['IC_DEP'] = ic_dep
    dict['IC_ARR'] = ic_arr
    dict['IC_TTIME'] = ic_tt
    dict['I_DEP'] = I_dep
    dict['I_ARR']= I_arr
    dict['I_TTIME'] = I_tt
    dict['P_SIZE'] = np.prod(dict['P_ARR'].shape)
    dict['AV_SIZE'] = np.prod(dict['AV_ARR'].shape)
    dict['NTV_SIZE'] = np.prod(dict['NTV_ARR'].shape)
    dict['IC_SIZE'] = np.prod(dict['IC_ARR'].shape)
    dict['I_SIZE'] = np.prod(dict['I_ARR'].shape)

    return dict

def getData(dict):
    ''' Construct a dictionary containing all the data_file
        Returns:
            dict          dictionarry containing all the data
    '''

    # Number of alternatives in the choice set (without considering opt-out)
    dict['I'] = 4
    dict['I_opt_out'] = 30

    # Number of customers
    dict['N'] = 40

    #### Parameters of the utility function
    # Alternative specific coefficients
    dict['ASC_CAR'] = [0.0, 0.0]
    dict['ASC_PLANE'] = [1.09334736, -1.16793213]
    dict['ASC_IC_1'] =[-1.97906711, -1.2320878]
    dict['ASC_IC_2'] = [-1.93463076, -1.27745065]
    dict['ASC_AV_1'] = [1.34130557, -0.59902074] #0.94130557, -0.69902074
    dict['ASC_AV_2'] = [0.9218437, -0.76585163] # 0.9218437, -0.76585163
    dict['ASC_NTV_1'] = [-0.05324599, -0.99663822] #-0.45324599, -1.09663822
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
    # Price of opt-out alternatives
    dict['PRICE_CAR'] = 104.79
    dict['PRICE_PLANE'] = 153.0
    dict['PRICE_IC_1'] = 60.0 #60
    dict['PRICE_IC_2'] = 30.0 #30
    dict['PRICE_AV_1'] = 125.0
    dict['PRICE_AV_2'] = 80.0
    dict['PRICE_NTV_1'] = 105.0
    dict['PRICE_NTV_2'] = 60.0

    # Customer's socio-economic characteristics
    # Customer id          18  121  291  476  630  860 1061 1265 1550 2043 2192 2504 2840 3174 3339 3470 4017 4287 5073 5500  371  664  801  948 1058 1450 1466 1662 1712 2439 2478 2745 2963 3384 3456 3523 3567 4115 4119 4337
    dict['BUSINESS'] =   [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    dict['ORIGIN'] =     [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    '''
    dict['DAT'] = [541.20124942, 664.7649158 , 521.07223503, 537.62732104,
                   575.85582691, 594.57374402, 559.03692653, 660.02932047,
                   628.61510731, 675.18314971, 542.34396609, 552.83660053,
                   632.63693098, 632.75592747, 537.92311715, 578.03313214,
                   636.09263532, 676.62483581, 599.47301625, 585.41259711,
                   539.18334737, 676.96839009, 610.80419345, 546.50702603,
                   544.72175599, 651.71559024, 625.83295927, 556.88725288,
                   544.93839282, 603.13847439, 552.29567848, 659.84013809,
                   538.46004302, 516.54014177, 621.12583634, 607.01307963,
                   512.11006864, 534.72343631, 635.50602624, 675.24803215]
    '''

    dict['DAT'] = [551.32908459, 512.88916369, 536.46485985, 550.18868248,
                   568.86309018, 513.76634526, 566.75139594, 528.5964896 ,
                   530.60547262, 565.27655111, 566.73728065, 516.58811877,
                   563.44056397, 536.48924471, 512.27207467, 560.32623426,
                   510.2820726 , 517.1143407 , 516.37744548, 566.31130405,
                   539.87014618, 559.20927541, 559.65507452, 544.11164471,
                   559.04880356, 562.61254468, 535.72860807, 547.46883297,
                   540.3254498 , 556.75905416, 514.59505961, 510.62254766,
                   547.63742331, 535.72147717, 516.55345233, 560.64368306,
                   516.66868849, 514.84387595, 541.22144169, 516.42833475]
    # Costs
    #                                   CAR   PLANE  IC_1   IC_2   AV_1   AV_2   NTV_1  NTV_2
    # Initial cost for each alternative
    dict['fixed_cost'] =    np.concatenate((np.zeros(4),
                                            150 * np.ones(dict['AV_SIZE']), 125 * np.ones(dict['AV_SIZE']),
                                            150 * np.ones(dict['NTV_SIZE']), 125 * np.ones(dict['NTV_SIZE']),
                                            np.array([150, 125, 150, 125])))
    assert(np.prod(dict['fixed_cost'].shape) == dict['I'] + dict['I_opt_out'])
    # Additional cost for each customer
    dict['customer_cost'] = np.concatenate((np.zeros(4),
                                            0 * np.ones(dict['AV_SIZE']), 0 * np.ones(dict['AV_SIZE']),
                                            0 * np.ones(dict['NTV_SIZE']), 0 * np.ones(dict['NTV_SIZE']),
                                            np.array([0, 0, 0, 0])))
    assert(np.prod(dict['customer_cost'].shape) == dict['I'] + dict['I_opt_out'])

    # Lower and upper bound on prices
    #TODO : Adapt lower bound to the costs
    #                                 CAR                PLANE                IC_1                IC_2        AV_1   AV_2   NTV_1  NTV_2
    dict['lb_p'] = np.concatenate((np.array([dict['PRICE_CAR'], dict['PRICE_PLANE'], dict['PRICE_IC_1'], dict['PRICE_IC_2']]),
                                   125 * np.ones(dict['AV_SIZE']), 80 * np.ones(dict['AV_SIZE']),
                                   105 * np.ones(dict['NTV_SIZE']), 60 * np.ones(dict['NTV_SIZE']),
                                   np.array([0.0, 0.0, 0.0, 0.0])))
    assert(np.prod(dict['lb_p'].shape) == dict['I'] + dict['I_opt_out'])

    dict['ub_p'] = np.concatenate((np.array([dict['PRICE_CAR'], dict['PRICE_PLANE'], dict['PRICE_IC_1'], dict['PRICE_IC_2']]),
                                   125 * np.ones(dict['AV_SIZE']), 80 * np.ones(dict['AV_SIZE']),
                                   105 * np.ones(dict['NTV_SIZE']), 60 * np.ones(dict['NTV_SIZE']),
                                   np.array([200.0, 200.0, 200.0, 200.0])))
    assert(np.prod(dict['ub_p'].shape) == dict['I'] + dict['I_opt_out'])

    # Choice set of the customers
    dict['choice_set'] = np.full((dict['I'] + dict['I_opt_out'], dict['N']), 1)

    # Mapping between alternatives index and their names
    dict['name_mapping'] = {0: 'Car', 1: 'Plane', 2: 'IC_1st', 3: 'IC_2nd',
                            4: 'AV1_2', 5: 'AV1_3', 6: 'AV1_4', 7: 'AV1_5', 8: 'AV1_6', 9: 'AV1_7', 10: 'AV1_8',
                            11: 'AV2_2', 12: 'AV2_3', 13: 'AV2_4', 14: 'AV2_5', 15: 'AV2_6', 16: 'AV2_7', 17: 'AV2_8',
                            18: 'NTV1_2', 19: 'NTV1_3', 20: 'NTV1_4', 21: 'NTV1_5', 22: 'NTV1_6', 23: 'NTV1_7',
                            24: 'NTV2_2', 25: 'NTV2_3', 26: 'NTV2_4', 27: 'NTV2_5', 28: 'NTV2_6', 29: 'NTV2_7',
                            30: 'AV1_1', 31: 'AV2_1', 32: 'NTV1_1', 33: 'NTV2_1'}

    dict['alt_mapping'] = {0: 0, 1: 0, 2: 0, 3: 0,
                          4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
                          11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
                          18: 1, 19: 2, 20: 3, 21: 4, 22: 5, 23: 6,
                          24: 1, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6,
                          30: 0, 31: 0, 32: 0, 33: 0}
    # Capacities
    #                              CAR   PLANE  IC_1   IC_2   AV_1   AV_2   NTV_1  NTV_2
    #dict['capacity'] = np.array([20.00, 20.00, 20.00, 20.00, 20.00 ,20.00 ,20.00 ,20.00]) # Availability for each alternative (opt-out always available)

    return dict

def preprocess(dict):
    ''' Precomputation on the data in order to create the corresponding
        cplex model.
    '''

    # Precomputation of difference between desired and actual arrival time for each mode and each journey.
    # Start with initialization of arrays
    p_early = np.empty([np.prod(dict['P_ARR'].shape), dict['N']])
    p_late = np.empty([np.prod(dict['P_ARR'].shape), dict['N']])
    av_early = np.empty([np.prod(dict['AV_ARR'].shape), dict['N']])
    av_late = np.empty([np.prod(dict['AV_ARR'].shape), dict['N']])
    ntv_early = np.empty([np.prod(dict['NTV_ARR'].shape), dict['N']])
    ntv_late = np.empty([np.prod(dict['NTV_ARR'].shape), dict['N']])
    ic_early = np.empty([np.prod(dict['IC_ARR'].shape), dict['N']])
    ic_late = np.empty([np.prod(dict['IC_ARR'].shape), dict['N']])
    I_early = np.empty([np.prod(dict['I_ARR'].shape), dict['N']])
    I_late = np.empty([np.prod(dict['I_ARR'].shape), dict['N']])

    # ----------------- PLANE ---------------- #
    for i in range(np.prod(dict['P_ARR'].shape)):
        for n in range(dict['N']):
            diff = dict['P_ARR'][i] - dict['DAT'][n]
            if( diff >= 0 ):
                p_early[i, n] = 0.0
                p_late[i, n] = diff
            else:
                p_early[i, n] = -diff
                p_late[i, n] = 0.0

    # ---------------- AV ------------------ #
    for i in range(np.prod(dict['AV_ARR'].shape)):
        for n in range(dict['N']):
            diff = dict['AV_ARR'][i] - dict['DAT'][n]
            if( diff >= 0 ):
                av_early[i, n] = 0.0
                av_late[i, n] = diff
            else:
                av_early[i, n] = -diff
                av_late[i, n] = 0.0

    # ----------------- NTV ---------------- #
    for i in range(np.prod(dict['NTV_ARR'].shape)):
        for n in range(dict['N']):
            diff = dict['NTV_ARR'][i] - dict['DAT'][n]
            if( diff >= 0):
                ntv_early[i, n] = 0.0
                ntv_late[i, n] = diff
            else:
                ntv_early[i, n] = -diff
                ntv_late[i, n] = 0.0

    # ----------------- IC ---------------- #
    for i in range(np.prod(dict['IC_ARR'].shape)):
        for n in range(dict['N']):
            diff = dict['IC_ARR'][i] - dict['DAT'][n]
            if( diff >= 0):
                ic_early[i, n] = 0.0
                ic_late[i, n] = diff
            else:
                ic_early[i, n] = -diff
                ic_late[i, n] = 0.0

    # ----------------- Alternatives ---------------- #
    for i in range(np.prod(dict['I_ARR'].shape)):
        for n in range(dict['N']):
            diff = dict['I_ARR'][i] - dict['DAT'][n]
            if( diff >= 0):
                I_early[i, n] = 0.0
                I_late[i, n] = diff
            else:
                I_early[i, n] = -diff
                I_late[i, n] = 0.0

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
    choice = np.empty([dict['I'] + dict['I_opt_out'], dict['N']])
    for n in range(dict['N']):
        # default customer n is not business
        idx = 0
        if( dict['BUSINESS'][n] == 1):
            idx = 1
        for i in range(dict['I'] + dict['I_opt_out']):
            if( i == 0 ):
                # ------ CAR ------ #
                exo_utility[i, n] = (dict['ASC_CAR'][idx] +
                                     dict['BETA_TTIME'][idx] * dict['C_TTIME'] +
                                     dict['BETA_COST'][idx] * dict['PRICE_CAR'])
            elif( i == 1 ):
                # ----- PLANE ----- #
                for l in range(np.prod(dict['P_ARR'].shape)):
                    tmp = (dict['ASC_PLANE'][idx] +
                            dict['BETA_TTIME'][idx] * dict['P_TTIME'][l] +
                            dict['BETA_EARLY'][idx] * dict['P_EARLY'][l, n] + dict['BETA_LATE'][idx] * dict['P_LATE'][l, n] +
                            dict['BETA_COST'][idx] * dict['PRICE_PLANE'])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 2 ):
                # ------ IC 1 ------ #
                for l in range(np.prod(dict['IC_ARR'].shape)):
                    tmp = (dict['ASC_IC_1'][idx] + dict['ASC_IC'][idx] +
                            dict['BETA_TTIME'][idx] * dict['IC_TTIME'][l] +
                            dict['BETA_EARLY'][idx] * dict['IC_EARLY'][l, n] + dict['BETA_LATE'][idx] * dict['IC_LATE'][l, n] +
                            dict['BETA_COST'][idx] * dict['PRICE_IC_1'])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( i == 3 ):
                # ------ IC 2 ------ #
                for l in range(np.prod(dict['IC_ARR'].shape)):
                    tmp = (dict['ASC_IC_2'][idx] + dict['ASC_IC'][idx] +
                            dict['BETA_TTIME'][idx] * dict['IC_TTIME'][l] +
                            dict['BETA_EARLY'][idx] * dict['IC_EARLY'][l, n] + dict['BETA_LATE'][idx] * dict['IC_LATE'][l, n] +
                            dict['BETA_COST'][idx] * dict['PRICE_IC_2'])
                    if( tmp > exo_utility[i, n] ):
                        exo_utility[i, n] = tmp
                        choice[i, n] = l
            elif( (i >= 4) and (i < 11) ):
                # ------ AV 1 ----- #
                exo_utility[i, n] = (dict['ASC_AV_1'][idx] + dict['ASC_HSR'][idx] +
                                    dict['ASC_AV'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['AV_TTIME'][dict['alt_mapping'][i] - 1] +
                                    dict['BETA_EARLY'][idx] * dict['AV_EARLY'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_LATE'][idx] * dict['AV_LATE'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_COST'][idx] * dict['PRICE_AV_1'] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( (i >= 11) and (i < 18) ):
                # ------ AV 2 ----- #
                exo_utility[i, n] = (dict['ASC_AV_2'][idx] + dict['ASC_HSR'][idx] +
                                    dict['ASC_AV'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['AV_TTIME'][dict['alt_mapping'][i] - 1] +
                                    dict['BETA_EARLY'][idx] * dict['AV_EARLY'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_LATE'][idx] * dict['AV_LATE'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_COST'][idx] * dict['PRICE_AV_2'] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( (i >= 18) and (i < 24) ):
                # ------ NTV 1 ----- #
                exo_utility[i, n] = (dict['ASC_NTV_1'][idx] + dict['ASC_NTV'][idx] +
                                    dict['ASC_HSR'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['NTV_TTIME'][dict['alt_mapping'][i] - 1] +
                                    dict['BETA_EARLY'][idx] * dict['NTV_EARLY'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_LATE'][idx] * dict['NTV_LATE'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_COST'][idx] * dict['PRICE_NTV_1'] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( (i >= 24) and (i < 30) ):
                # ------ NTV 2 ----- #
                exo_utility[i, n] = (dict['ASC_NTV_2'][idx] + dict['ASC_NTV'][idx] +
                                    dict['ASC_HSR'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['NTV_TTIME'][dict['alt_mapping'][i] - 1] +
                                    dict['BETA_EARLY'][idx] * dict['NTV_EARLY'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_LATE'][idx] * dict['NTV_LATE'][dict['alt_mapping'][i] - 1, n] +
                                    dict['BETA_COST'][idx] * dict['PRICE_NTV_2'] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 30):
                # ------ AV 1 ----- #
                exo_utility[i, n] = (dict['ASC_AV_1'][idx] + dict['ASC_HSR'][idx] +
                                    dict['ASC_AV'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][0] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][0, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][0, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 31 ):
                # ------ AV 2 ----- #
                exo_utility[i, n] = (dict['ASC_AV_2'][idx] + dict['ASC_HSR'][idx] +
                                    dict['ASC_AV'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][0] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][0, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][0, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 32 ):
                # ------ NTV 1 ----- #
                exo_utility[i, n] = (dict['ASC_NTV_1'][idx] + dict['ASC_NTV'][idx] +
                                    dict['ASC_HSR'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][1] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][1, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][1, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])
            elif( i == 33 ):
                # ------ NTV 2 ----- #
                exo_utility[i, n] = (dict['ASC_NTV_2'][idx] + dict['ASC_NTV'][idx] +
                                    dict['ASC_HSR'][idx] +
                                    dict['BETA_TTIME'][idx] * dict['I_TTIME'][1] +
                                    dict['BETA_EARLY'][idx] * dict['I_EARLY'][1, n] + dict['BETA_LATE'][idx] * dict['I_LATE'][1, n] +
                                    dict['BETA_ORIGIN'][idx] * dict['ORIGIN'][n])

    dict['exo_utility'] = exo_utility
    dict['CHOICE'] = choice

    # Beta coefficient for endogenous variables
    # TODO: reduce size of endo_coe
    endo_coef = np.full([dict['I'] + dict['I_opt_out'], dict['N']], 0.0)
    for n in range(dict['N']):
        idx = int(dict['BUSINESS'][n])
        endo_coef[dict['I_opt_out']:, n] = dict['BETA_COST'][idx]

    dict['endo_coef'] = endo_coef

    # Calculate bounds on the utility
    lb_U = np.empty([dict['I'] + dict['I_opt_out'], dict['N']])
    ub_U = np.empty([dict['I'] + dict['I_opt_out'], dict['N']])
    for n in range(dict['N']):
        for i in range(dict['I'] + dict['I_opt_out']):
            if( i < 4 ):
                lb_U[i,n] = dict['exo_utility'][i, n]
                ub_U[i,n] = dict['exo_utility'][i, n]
            elif( i >= 30 ):
                # WARNING: if dict['endo_coef'] if positive we have to inverse ub_p and lb_p in the computation!
                lb_U[i, n] = dict['endo_coef'][i, n] * dict['ub_p'][i] + dict['exo_utility'][i, n]
                ub_U[i, n] = dict['endo_coef'][i, n] * dict['lb_p'][i] + dict['exo_utility'][i, n]
                assert(lb_U[i, n] <= ub_U[i, n])

    dict['lb_U'] = lb_U
    dict['ub_U'] = ub_U

def customer_info(dict, n):
    print('****-------------- Informations about Customer:', n, '------------------------****')
    print('** Basic informations: **\nBusiness customer', dict['BUSINESS'][n])
    print('Coming from the city', dict['ORIGIN'][n])
    print('Desired arrival time is at:', dict['DAT'][n] / 60)
    p = int(dict['CHOICE'][1, n])
    ic1 = int(dict['CHOICE'][2, n])
    ic2 = int(dict['CHOICE'][3, n])
    av1 = np.argmax((dict['exo_utility'][4, n], dict['exo_utility'][5, n], dict['exo_utility'][6, n], dict['exo_utility'][7, n],
                     dict['exo_utility'][8, n], dict['exo_utility'][9, n], dict['exo_utility'][10, n]))
    av2 = np.argmax((dict['exo_utility'][11, n], dict['exo_utility'][12, n], dict['exo_utility'][13, n], dict['exo_utility'][14, n],
                     dict['exo_utility'][15, n], dict['exo_utility'][16, n], dict['exo_utility'][17, n]))
    assert( av1 == av2 )
    ntv1 = np.argmax((dict['exo_utility'][18, n], dict['exo_utility'][19, n], dict['exo_utility'][20, n], dict['exo_utility'][21, n],
                     dict['exo_utility'][22, n], dict['exo_utility'][23, n]))
    ntv2 = np.argmax((dict['exo_utility'][24, n], dict['exo_utility'][25, n], dict['exo_utility'][26, n], dict['exo_utility'][27, n],
                     dict['exo_utility'][28, n], dict['exo_utility'][29, n]))
    assert( ntv1 == ntv2 )

    # print opt-out informations #
    print('** Trip informations: **\n****** Car: *******\nTakes car which cost', dict['PRICE_CAR'], '/ utility:', dict['exo_utility'][0, n])
    print('****** Plane: *******\nTakes plane number:', p, 'at price', dict['PRICE_PLANE'],
          'which leaves at', dict['P_DEP'][p] / 60, 'and arrives at', dict['P_ARR'][p] / 60, '/ utility:', dict['exo_utility'][1, n])
    print('****** IC1 Train: *******\nTakes IC 1st train number:', ic1, 'at price', dict['PRICE_IC_1'],
          'which leaves at:', dict['IC_DEP'][ic1] / 60,
          'and arrives at:', dict['IC_ARR'][ic1] / 60, '/ utility:', dict['exo_utility'][2, n])
    print('****** IC2 Train: *******\nTakes IC 2nd train number:', ic2, 'at price', dict['PRICE_IC_2'],
          'which leaves at:', dict['IC_DEP'][ic2] / 60,
          'and arrives at:', dict['IC_ARR'][ic2] / 60, '/ utility:', dict['exo_utility'][3, n])
    print('****** AV1 Train: *******\nIf he takes the second AV1 train at price', dict['PRICE_AV_1'], 'which leaves at',
         dict['AV_DEP'][0] / 60, 'and arrives at', dict['AV_ARR'][0] / 60, '/ utility:',
         dict['exo_utility'][4, n])
    print('****** AV2 Train: *******\nIf he takes the second AV2 train at price', dict['PRICE_AV_2'], 'which leaves at',
         dict['AV_DEP'][0] / 60, 'and arrives at', dict['AV_ARR'][0] / 60, '/ utility:',
         dict['exo_utility'][11, n])
    print('****** NTV1 Train: *******\nIf he takes the second NTV1 train at price', dict['PRICE_NTV_1'], 'which leaves at',
         dict['NTV_DEP'][0] / 60, 'and arrives at', dict['NTV_ARR'][0] / 60, '/ utility:',
         dict['exo_utility'][18, n])
    print('****** NTV2 Train: *******\nIf he takes the second NTV2 train at price', dict['PRICE_NTV_2'], 'which leaves at',
         dict['NTV_DEP'][0] / 60, 'and arrives at', dict['NTV_ARR'][0] / 60, '/ utility:',
         dict['exo_utility'][24, n])

    # print alternative informations #
    print('\nIf he takes the first AV1 train at price', dict['PRICE_AV_1'], 'which leaves at',
          dict['I_DEP'][0] / 60, 'and arrives at', dict['I_ARR'][0] / 60, '/ utility:',
          dict['exo_utility'][30, n] + dict['endo_coef'][30, n] * dict['PRICE_AV_1'])
    print('If he takes the first AV2 train at price', dict['PRICE_AV_2'], 'which leaves at',
          dict['I_DEP'][0] / 60, 'and arrives at', dict['I_ARR'][0] / 60, '/ utility:',
          dict['exo_utility'][31, n] + dict['endo_coef'][31, n] * dict['PRICE_AV_2'])
    print('If he takes the first NTV1 train at price', dict['PRICE_NTV_1'], 'which leaves at',
          dict['I_DEP'][1] / 60, 'and arrives at', dict['I_ARR'][1] / 60, '/ utility:',
          dict['exo_utility'][32, n] + dict['endo_coef'][32, n] * dict['PRICE_NTV_1'])
    print('If he takes the first NTV2 train at price', dict['PRICE_NTV_2'], 'which leaves at',
          dict['I_DEP'][1] / 60, 'and arrives at', dict['I_ARR'][1] / 60, '/ utility:',
          dict['exo_utility'][33, n] + dict['endo_coef'][33, n] * dict['PRICE_NTV_2'])


if __name__ == '__main__':
    dict = getInfo()
    dict = getData(dict)
    preprocess(dict)
    customer_info(dict, 3)