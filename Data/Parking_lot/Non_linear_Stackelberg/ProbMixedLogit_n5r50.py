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
    dict['I_opt_out'] = 1

    # Number of customers
    dict['N'] = 5

    # Number of coef draws
    dict['R_coef'] = 50

    # Lower and upper bound on prices
    dict['lb_p'] = np.array([0, 0.0, 0.0]) # lower bound (FSP, PSP, PUP)
    dict['ub_p'] = np.array([0, 1.0, 1.0]) # upper bound (FSP, PSP, PUP)

    #dict['capacity'] = np.array([60.0, 3.0, 3.0]) # Availability for each alternative (opt-out always available)

    #dict['fixed_cost'] = [0.0, 1.0, 1.0] # Initial cost for each alternative
    #dict['customer_cost'] = [0.1, 0.15, 0.3] # Additional cost for each new customer

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

    # AT coefficient (10 customers x 50 draws)
    dict['Beta_AT'] = np.array([[-0.14886844, -1.53648077,  0.00442981,  0.06356133, -1.1207989 ,
         0.49630332,  1.24735094, -0.31497159, -3.05937407,  0.3680428 ,
        -0.89389242, -1.1763363 , -0.73663446, -1.11539   ,  0.34664627,
        -0.83221555, -1.06971027,  1.37833643, -0.8325042 , -0.01831247,
        -0.2895702 , -0.3012322 , -0.81009987,  0.05337716, -1.22434261,
         1.18501047, -2.75313802,  0.21543921,  0.52793544,  0.2093568 ,
        -0.39709224, -0.02779882, -1.85014654,  1.33404123, -0.82630827,
        -2.00308474, -0.38267834, -0.71166062, -3.5883083 , -1.41735144,
        -1.95737934,  0.02268142, -0.18946594, -1.23113272, -0.86038713,
        -1.32440814, -0.68038018, -0.37260709,  0.06863267, -0.94741738],
       [ 0.32356665, -1.72555478, -0.63870944,  1.58986578, -1.60069774,
        -1.52185042, -0.17131153, -2.31686942, -1.2489171 , -1.91373478,
        -0.87279334, -0.65404174, -1.63284361, -0.12851357, -1.36384823,
        -1.31562829, -1.2964533 , -1.97951026, -0.26733487, -1.62677243,
        -0.82479983,  0.79865952, -0.81740346, -1.23141899, -2.48020113,
         1.41528629,  0.86849774, -1.74097568, -0.44036965, -1.93369739,
        -0.98667698, -0.91769546, -0.88089991, -0.48105526, -1.28308139,
        -2.05173872, -0.93812573, -1.31997299,  0.42797459,  1.08127901,
        -0.0036722 , -2.25598855, -1.85221876, -2.16642453, -2.5989144 ,
        -1.38660681, -2.73647224, -0.23829559,  0.17304584, -0.97874761],
       [ 0.66408521, -0.54508642,  0.25101265, -0.57693255, -1.71955452,
         0.59278346, -0.31294125, -2.55591108, -1.94448136, -0.93664537,
        -0.76460469, -2.81541195, -1.62908048, -1.15557252, -0.34690239,
        -0.52765966,  0.22399212, -0.85383343, -2.09445468, -1.00024868,
        -0.49112132, -1.31640591,  0.12340088, -0.63369715, -0.44883843,
        -1.33535189,  0.31202206, -0.3294371 ,  0.51853017, -1.3467551 ,
         0.02893138,  0.81842166, -1.43194616, -2.77602691, -0.02747669,
        -0.77935456,  0.29271517, -3.09505224,  1.44581566, -0.08858741,
         1.03604424, -0.86570837, -1.24585014, -1.72952858,  0.1292708 ,
        -0.35427161, -0.79551744,  0.01255101, -0.22202037, -0.25707665],
       [ 0.09644719, -1.11584377, -1.21083868, -1.43643053, -0.16705761,
        -1.65001913, -2.67387343, -0.79499062, -1.40136056,  0.38626395,
        -1.7506502 , -3.50794605, -2.12800858, -0.23572161, -0.25259187,
        -1.27960594, -0.68048443,  0.30508383, -0.94971167, -2.8115918 ,
         0.25534527, -2.9739932 , -1.26298484, -0.80199397, -2.18922574,
        -1.00270078, -0.22939413, -1.23573041, -0.90389966, -1.83611383,
        -2.97995975,  0.67392029, -1.82319508, -1.23848152, -3.00819846,
        -0.27212657, -1.07781942, -1.36657049, -1.3621166 , -3.54218827,
        -2.42597651, -0.62093272, -1.05098987, -0.4713425 , -1.5066381 ,
         0.60928756, -0.83931992, -0.60249437,  0.17862756, -0.78272049],
       [ 0.01332011, -0.39414891, -1.39659129, -1.29869099, -2.54452436,
         1.17207669, -2.28898212, -1.39410058, -0.02525089,  0.20643019,
         0.41804047,  1.25936338, -1.04878003, -0.45618808,  0.0698209 ,
        -0.24423307, -1.89553337,  0.49548205, -1.44472999, -1.45497256,
        -1.01013032, -1.56431487,  0.32348322,  0.62577625, -0.92979541,
        -1.02471932, -0.72892061, -2.69704102, -1.15073989, -1.15519141,
        -0.29350742, -0.37241281, -0.83882847, -1.15041391, -0.37172476,
        -2.20690594, -1.72565549, -2.26666198, -1.11686224, -0.47130361,
        -0.480054  , -1.0750419 ,  0.03752794,  0.31247733,  0.19431051,
        -0.20525639, -0.70776651,  0.57660546,  0.28558667, -1.23462089]])

    # FEE coefficient
    dict['Beta_FEE'] = np.array([[-22.6236475 , -43.00454565, -39.11370316, -45.49861751,
        -20.11535774, -19.08835857, -48.04821526, -22.98543805,
          1.85117145, -34.82556198, -16.86865758, -33.5800067 ,
        -20.41748145, -35.82423759, -41.28877901, -28.85455974,
         -6.19885286, -45.1514732 , -42.31462805, -36.95232346,
        -29.4063889 , -51.17297604, -36.78489049, -27.35104079,
        -21.97681961, -16.31153958, -36.32942854, -33.23980653,
        -37.18850144, -39.7993733 , -29.06545687, -31.4463881 ,
        -36.68733186, -41.04704817, -22.22059123, -30.52906579,
        -34.9514993 , -39.78612764, -34.62648034, -32.75975696,
        -25.24164412, -25.58386597, -23.4814719 , -22.02101056,
        -24.86408201,  -7.81718205, -39.57333076, -51.26835771,
        -37.79537518, -56.58464301],
       [-34.3405435 , -25.73073391, -12.28106537, -26.85811176,
        -24.5465459 , -37.99559723, -23.24566977, -35.6434403 ,
        -53.35038353, -21.7955554 , -34.81282008, -65.10520542,
        -28.94116974, -39.7132417 , -11.05869562, -20.69291431,
        -35.7513125 , -50.62591074, -35.78442114, -13.62824923,
        -25.63593508, -23.79258163, -32.76751174,  -8.62048933,
        -37.14986106, -27.68169593, -38.01652157, -35.56435643,
        -41.82428228, -36.54147797, -26.24783633, -28.75434363,
        -27.94900793, -23.18557832, -21.15352252, -18.44894132,
        -22.06052381, -43.85326216, -32.56588967, -38.83132689,
        -16.95025046, -43.39865136, -31.28431733, -11.44748113,
        -24.70283031, -17.51279077, -34.2285512 , -34.98591407,
        -33.15162709, -35.89032181],
       [-39.7476621 , -44.33605813, -56.24315827, -33.98749568,
        -27.0414748 , -34.62268228, -50.36401487, -41.45165912,
        -41.99356656, -39.5735969 , -33.18260427, -30.98169215,
        -36.46547566, -44.17792565, -12.10999996, -37.30960384,
        -34.02008203, -28.37454085, -54.32982353, -42.9779827 ,
         -7.06834384,  -4.03714933, -26.86324926, -36.82006981,
        -41.55870105, -52.34376584, -26.04686581, -27.40299077,
        -19.36194351, -16.23518806, -11.59502198, -50.15123662,
        -52.7256158 , -43.62179326, -34.44216346, -31.16057896,
        -42.03507506, -39.5509352 , -53.78964591, -18.61180642,
        -31.50629839, -27.84449571, -38.42476432, -34.96731668,
        -27.59459285, -30.14458215, -17.9926274 , -26.76914731,
        -21.61132442, -11.62711851],
       [-47.82520718, -28.28374536, -26.15499674,  -6.63416771,
        -38.13319595, -27.48486859,  19.62745819, -27.81050893,
         -6.36352708, -22.79187624, -36.62251391, -50.64697698,
        -20.5348645 , -55.33183138, -41.36284027, -30.04786052,
        -18.59165633, -38.25180739, -49.9832114 , -52.37373581,
        -37.98631303, -28.9129132 , -35.94013038, -22.36517442,
        -47.63726666,  -5.28188025, -30.62201315, -35.43702488,
         -7.10418456,  -7.97440107, -21.69231142, -21.14889479,
        -40.51166998, -56.22629827, -31.26717917, -22.26155105,
        -53.36330782,  -6.78896298,  -4.6871116 , -21.14343934,
        -66.31755728, -18.85609712, -12.61439534, -34.38378438,
        -23.88792801, -17.5703054 , -32.33357786, -25.49444485,
        -41.32778855, -51.70280191],
       [ -9.05264578, -28.36884937,  -9.43471349, -40.39317329,
        -55.43534848, -31.91202803, -49.5541363 , -18.33931984,
        -28.10403533, -31.22097292,  -5.26718567, -35.80034025,
        -24.11908916, -15.19077543, -34.05076715, -49.8337674 ,
        -16.68184465, -49.93392814, -75.05470643, -15.67592421,
        -50.927001  ,  -8.18500064, -38.19652289, -48.55347381,
        -44.27373242, -16.67645018, -35.64632794, -35.31542247,
        -19.47995741, -41.1306548 , -24.99109061, -22.71589872,
        -43.14245157, -32.62179483, -34.85441346, -34.99733255,
        -24.41324286, -35.06107476, -17.26903192, -51.84192151,
        -23.19857993, -20.33542437, -37.18054507, -22.03852943,
        -18.92672191, -34.65988957, -13.90088995, -23.72908795,
        -40.28907347, -26.96403626]])

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
                max -= 1
    dict['priority_list'] = priority_list

    # Exogene utility
    exo_utility = np.empty([dict['I'] + 1, dict['N'], dict['R_coef']])
    for n in range(dict['N']):
        for i in range(dict['I'] + 1):
            for r in range(dict['R_coef']):
                if i == 0:
                    # Opt-Out
                    exo_utility[i, n, r] = (dict['Beta_AT'][n, r] * dict['AT_FSP'] +
                                           dict['Beta_TD'] * dict['TD_FSP'] +
                                           dict['Beta_Origin'] * dict['Origin'][n])
                elif i == 1:
                    # PSP
                    exo_utility[i, n, r] = (dict['ASC_PSP'] +
                                           dict['Beta_AT'][n, r] * dict['AT_PSP'] +
                                           dict['Beta_TD'] * dict['TD_PSP'])
                else:
                    # PUP
                    exo_utility[i, n, r] = (dict['ASC_PUP'] +
                                           dict['Beta_AT'][n, r] * dict['AT_PUP'] +
                                           dict['Beta_TD'] * dict['TD_PUP'] +
                                           dict['Beta_Age_Veh'] * dict['Age_veh'][n])
    dict['exo_utility'] = exo_utility

    # Beta coefficients for endogenous variables
    beta_FEE_PSP = np.empty([dict['N'], dict['R_coef']])
    beta_FEE_PUP = np.empty([dict['N'], dict['R_coef']])
    for n in range(dict['N']):
        for r in range(dict['R_coef']):
            beta_FEE_PSP[n, r] = (dict['Beta_FEE'][n, r] +
                                 dict['Beta_FEE_INC_PSP'] * dict['Low_inc'][n] +
                                 dict['Beta_FEE_RES_PSP'] * dict['Res'][n])
            beta_FEE_PUP[n, r] = (dict['Beta_FEE'][n, r] +
                                 dict['Beta_FEE_INC_PUP'] * dict['Low_inc'][n] +
                                 dict['Beta_FEE_RES_PUP'] * dict['Res'][n])
    dict['endo_coef'] = np.array([np.zeros([dict['N'], dict['R_coef']]), beta_FEE_PSP, beta_FEE_PUP])

    # Calculate bounds on the utility
    lb_U = np.empty([dict['I'] + 1, dict['N'], dict['R_coef']])
    ub_U = np.empty([dict['I'] + 1, dict['N'], dict['R_coef']])
    for n in range(dict['N']):
        for i in range(dict['I'] + 1):
            for r in range(dict['R_coef']):
                if dict['endo_coef'][i, n, r] > 0:
                    lb_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['lb_p'][i] +
                                    dict['exo_utility'][i, n, r])
                    ub_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['ub_p'][i] +
                                    dict['exo_utility'][i, n, r])
                else:
                    lb_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['ub_p'][i] +
                                    dict['exo_utility'][i, n, r])
                    ub_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['lb_p'][i] +
                                    dict['exo_utility'][i, n, r])

    dict['lb_U'] = lb_U
    dict['ub_U'] = ub_U


if __name__ == '__main__':
    dict = getData()
    preprocess(dict)
