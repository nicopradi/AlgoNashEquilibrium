# Data Stackelberg game with capacity, continuous price
import numpy as np

def getData():
    ''' Construct a dictionary containing all the data_file
        Returns:
            dict          dictionarry containing all the data
    '''
    # Initialize the output dictionary
    dict = {}

    # Number of alternatives in the choice set (without considering opt-out)
    dict['I'] = 2

    # Number of customers
    dict['N'] = 10

    # Lower and upper bound on prices
    dict['lb_p'] = np.array([0, 0.0, 0.0]) # lower bound (FSP, PSP, PUP)
    dict['ub_p'] = np.array([0, 1.0, 1.0]) # upper bound (FSP, PSP, PUP)

    #dict['capacity'] = np.array([60.0, 6.0, 6.0]) # Availability for each alternative (opt-out always available)
    # Choice set of the customers
    #	 		           n1 n2 n3...
    dict['choice_set'] = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # OPT-OUT
                       			  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # PSP
                    	 		  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # PUP

    # Parameters choice model
    dict['ASC_PSP'] = 32
    dict['ASC_PUP'] = 34
    dict['Beta_TD'] = -0.612
    dict['Beta_Origin'] = -5.762
    dict['Beta_Age_Veh'] = 4.037
    dict['Beta_FEE_INC_PSP'] = -10.995
    dict['Beta_FEE_RES_PSP'] = -11.440
    dict['Beta_FEE_INC_PUP'] = -13.729
    dict['Beta_FEE_RES_PUP'] = -10.668

    # AT coefficient
    dict['Beta_AT'] = -0.788

    # FEE coefficient
    dict['Beta_FEE'] = -32.328

    # Variables choice model
    dict['AT_FSP'] = 10
    dict['TD_FSP'] = 10
    dict['Origin'] = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    dict['AT_PSP'] = 10
    dict['TD_PSP'] = 10
    dict['AT_PUP'] = 5
    dict['TD_PUP'] = 10
    dict['Age_veh'] = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    dict['Low_inc'] = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    dict['Res'] = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])

    return dict

def preprocess(dict):
    ''' Precomputation on the data in order to create the corresponding
        cplex model.
    '''

    ########## Precomputation ##########
    # Priority list
    priority_list = np.empty([dict['I'] + 1, dict['N']])
    for i in range(dict['I'] + 1):
        min = 1
        max = dict['N']
        for n in range(dict['N']):
            if dict['choice_set'][i, n] == 1:
                priority_list[i, n] = min
                min += 1
            else:
                priority_list[i, n] = max
                min -= 1
    dict['priority_list'] = priority_list

    # Exogene utility
    exo_utility = np.empty([dict['I'] + 1, dict['N']])
    for n in range(dict['N']):
        for i in range(dict['I'] + 1):
            if i == 0:
                # Opt-Out
                exo_utility[i, n] = (dict['Beta_AT'] * dict['AT_FSP'] +
                                       dict['Beta_TD'] * dict['TD_FSP'] +
                                       dict['Beta_Origin'] * dict['Origin'][n])
            elif i == 1:
                # PSP
                exo_utility[i, n] = (dict['ASC_PSP'] +
                                       dict['Beta_AT'] * dict['AT_PSP'] +
                                       dict['Beta_TD'] * dict['TD_PSP'])
            else:
                # PUP
                exo_utility[i, n] = (dict['ASC_PUP'] +
                                       dict['Beta_AT'] * dict['AT_PUP'] +
                                       dict['Beta_TD'] * dict['TD_PUP'] +
                                       dict['Beta_Age_Veh'] * dict['Age_veh'][n])
    dict['exo_utility'] = exo_utility

    # Beta coefficients for endogenous variables
    beta_FEE_PSP = np.empty([dict['N']])
    beta_FEE_PUP = np.empty([dict['N']])
    for n in range(dict['N']):
        beta_FEE_PSP[n] = (dict['Beta_FEE'] +
                             dict['Beta_FEE_INC_PSP'] * dict['Low_inc'][n] +
                             dict['Beta_FEE_RES_PSP'] * dict['Res'][n])
        beta_FEE_PUP[n] = (dict['Beta_FEE'] +
                             dict['Beta_FEE_INC_PUP'] * dict['Low_inc'][n] +
                             dict['Beta_FEE_RES_PUP'] * dict['Res'][n])
    dict['endo_coef'] = np.array([np.zeros([dict['N']]), beta_FEE_PSP, beta_FEE_PUP])

    # Calculate bounds on the utility
    lb_U = np.empty([dict['I'] + 1, dict['N']])
    ub_U = np.empty([dict['I'] + 1, dict['N']])
    for n in range(dict['N']):
        for i in range(dict['I'] + 1):
            if dict['endo_coef'][i, n] > 0:
                lb_U[i, n] = (dict['endo_coef'][i, n] * dict['lb_p'][i] +
                                dict['exo_utility'][i, n])
                ub_U[i, n] = (dict['endo_coef'][i, n] * dict['ub_p'][i] +
                                dict['exo_utility'][i, n])
            else:
                lb_U[i, n] = (dict['endo_coef'][i, n] * dict['ub_p'][i] +
                                dict['exo_utility'][i, n])
                ub_U[i, n] = (dict['endo_coef'][i, n] * dict['lb_p'][i] +
                                dict['exo_utility'][i, n])

    dict['lb_U'] = lb_U
    dict['ub_U'] = ub_U


if __name__ == '__main__':
    dict = getData()
    preprocess(dict)
