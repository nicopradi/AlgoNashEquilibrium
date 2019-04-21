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
    dict['N'] = 5

    # Number of draws
    dict['R'] = 50

    # Lower and upper bound on prices
    dict['lb_p'] = np.array([0, 0.00, 0.00]) # lower bound (FSP, PSP, PUP)
    dict['ub_p'] = np.array([0, 1.00, 1.00]) # upper bound (FSP, PSP, PUP)

    dict['capacity'] = np.array([60.0, 3.0, 3.0]) # Availability for each alternative (opt-out always available)
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

    # Random term (Gumbel distribution (0,1)) - 3 alternatives x 10 customers x 50 draws
    dict['xi'] = np.array([[[  8.88287700e-01,  -7.50947600e-01,  -1.97780310e+00,
          -1.52697500e+00,  -2.28565400e-01,   3.69475400e-01,
          -3.11329200e-01,  -1.86434340e+00,   2.49625300e-01,
          -9.74791600e-01,  -6.61659400e-01,  -1.94760910e+00,
          -1.79748070e+00,   1.00720370e+00,  -7.64116500e-01,
          -7.27799800e-01,  -6.00411600e-01,   3.73260600e-01,
           9.71175000e-01,   5.87978700e-01,   9.19574500e-01,
          -4.84287100e-01,  -5.87997200e-01,   7.72983300e-01,
          -9.13005200e-01,  -2.34323060e+00,  -4.11420300e-01,
          -5.11232400e-01,  -3.21986320e+00,  -2.07697040e+00,
          -1.19174400e-01,  -1.50678270e+00,  -1.62198470e+00,
          -2.36050740e+00,   4.36499000e-02,  -1.03294360e+00,
          -1.14472700e-01,   2.04545300e-01,  -3.31371200e-01,
          -1.34757150e+00,   3.20861000e-01,   6.99485000e-02,
          -1.84671300e+00,  -9.72856000e-02,   7.64780000e-03,
          -2.89460200e-01,  -2.26403420e+00,   5.27071900e-01,
           1.22395040e+00,   6.65196800e-01],
        [  7.87100900e-01,  -1.17984050e+00,  -1.25745560e+00,
          -4.08293730e+00,  -2.67520760e+00,  -3.54100300e-01,
          -1.18916660e+00,  -2.30427340e+00,   4.00527800e-01,
           6.72005400e-01,   1.79706660e+00,  -2.69629740e+00,
          -2.06943890e+00,   5.06786700e-01,  -1.73733180e+00,
           2.91308500e-01,  -1.07421180e+00,   1.08868540e+00,
          -1.30801090e+00,  -1.46972630e+00,   2.18924000e-02,
           4.85919800e-01,   4.04956100e-01,  -1.04064870e+00,
           5.67263400e-01,  -5.84336400e-01,  -5.73781500e-01,
          -7.12508000e-01,  -4.42968100e-01,  -6.51742500e-01,
          -4.72708500e-01,  -5.76108000e-01,   1.29122410e+00,
          -1.91527300e-01,  -1.24848990e+00,   8.07285600e-01,
          -2.62050000e-02,   8.57243400e-01,   7.06841500e-01,
          -1.77881600e-01,   3.94622900e-01,  -7.68070800e-01,
          -1.75668300e-01,  -4.22454000e-02,  -3.47792110e+00,
          -1.70007210e+00,   7.82520700e-01,  -1.70007210e+00,
           6.02634500e-01,   5.95364000e-01],
        [ -6.79779200e-01,  -2.45967080e+00,  -4.59924600e-01,
          -6.70447900e-01,  -2.53461080e+00,  -1.65552600e+00,
          -3.37124000e-01,  -1.65185480e+00,   3.69086200e-01,
           3.26979800e-01,  -3.13036940e+00,  -6.51742500e-01,
           1.41282630e+00,   2.49625300e-01,  -5.12858700e-01,
           3.11729300e-01,  -4.33058770e+00,   4.63287000e-02,
          -1.12157270e+00,   4.17219100e-01,  -6.35385000e-02,
          -1.65113970e+00,   7.48818000e-02,  -4.74488100e-01,
           1.30481020e+00,  -1.47949860e+00,  -1.02472550e+00,
          -1.64914710e+00,  -2.01715300e-01,  -1.67176960e+00,
          -1.73733180e+00,  -1.15275800e+00,   1.31618070e+00,
          -2.18870010e+00,   3.49999800e-01,  -2.78396000e-01,
           2.26240900e-01,  -2.39793920e+00,   1.02463600e-01,
          -3.60170600e-01,  -3.04913610e+00,   5.44577700e-01,
          -2.86737500e-01,  -1.27901610e+00,  -1.40683000e-02,
          -8.34070000e-02,  -2.82204740e+00,   5.20350000e-02,
          -1.82628970e+00,   6.72484900e-01],
        [  7.38013900e-01,   6.84739000e-01,  -7.00699200e-01,
          -5.59940000e-02,  -1.94849720e+00,  -3.54297500e-01,
          -2.78866000e-02,   1.76091200e+00,  -6.73836500e-01,
          -2.24024090e+00,  -1.97497600e-01,  -1.80567790e+00,
          -6.68603900e-01,  -1.83764660e+00,   4.00527800e-01,
           1.46302880e+00,   2.28233400e-01,  -1.96958200e-01,
          -6.50664200e-01,   4.76963100e-01,  -3.60865300e-01,
          -1.13118900e-01,   3.73260600e-01,  -1.83997560e+00,
          -1.72296200e-01,  -4.01843600e-01,  -3.12022500e-01,
          -6.24288900e-01,  -4.36111890e+00,  -9.51234100e-01,
          -1.78511000e-01,   5.87978700e-01,   6.15831500e-01,
          -1.31017720e+00,   7.25556000e-01,  -8.84671100e-01,
           1.01577100e-01,   1.04733800e-01,  -1.62366990e+00,
          -1.20224910e+00,  -1.13016130e+00,  -1.81079270e+00,
          -1.72365570e+00,   2.53159200e-01,  -4.93626400e-01,
          -8.44977300e-01,  -6.99294000e-02,   9.61240100e-01,
          -4.80618100e-01,  -7.57942500e-01],
        [  6.96066300e-01,  -1.96835730e+00,  -8.14588000e-01,
          -2.81923500e-01,  -8.50216400e-01,   1.10445850e+00,
          -4.17171000e-02,  -1.18916660e+00,  -1.27406760e+00,
           7.07207300e-01,  -1.23086970e+00,  -3.22006040e+00,
           1.09996970e+00,   6.97872500e-01,   1.20109360e+00,
           3.99451000e-01,   1.05328760e+00,  -2.17443380e+00,
          -1.05032900e-01,  -9.23874000e-01,  -3.09155100e-01,
          -8.26739300e-01,  -3.96294500e-01,   1.08747280e+00,
          -1.99878550e+00,   5.60186200e-01,  -3.54691460e+00,
          -4.44063400e-01,   1.95969000e-01,  -2.16912170e+00,
          -4.47014300e-01,  -2.10218530e+00,  -6.41615000e-02,
           7.87100900e-01,  -1.00944720e+00,  -1.84840730e+00,
          -1.19191650e+00,  -2.28197000e-02,  -3.56476690e+00,
           3.91066500e-01,  -1.71147090e+00,   1.12774600e-01,
          -2.17423100e-01,  -8.24998800e-01,  -9.93352800e-01,
          -3.67598900e-01,  -1.91275930e+00,   1.04442970e+00,
          -1.32531870e+00,  -1.32889850e+00]],

       [[ -1.59205810e+00,  -9.67603200e-01,  -3.46953000e-02,
          -2.00417500e-01,  -1.18792280e+00,  -2.42472810e+00,
           3.14348000e-02,  -7.33809000e+00,  -2.68996800e+00,
          -3.08301260e+00,  -4.62562000e-02,  -7.68070800e-01,
          -1.04003370e+00,  -5.77349000e-01,  -4.44063400e-01,
          -1.83435750e+00,   2.69856400e-01,  -9.25771700e-01,
           1.29598770e+00,   9.92217300e-01,   2.31057700e-01,
          -1.29879520e+00,  -3.76101540e+00,   1.29938530e+00,
          -2.35378780e+00,   2.38658000e-01,  -1.64914710e+00,
          -1.10035300e-01,  -4.52870700e-01,   1.20109360e+00,
          -1.06292800e-01,  -2.63403600e-01,   4.49999700e-01,
           1.19061300e+00,  -8.59489500e-01,  -3.42244000e-01,
          -8.58154700e-01,  -3.31371200e-01,  -6.03403900e-01,
          -1.12529380e+00,   2.53715600e-01,   6.10019700e-01,
           1.91087000e-01,  -3.85637100e-01,  -1.22217550e+00,
          -1.95068120e+00,  -2.14733490e+00,  -5.85119900e-01,
          -8.47355500e-01,  -9.79459300e-01],
        [ -1.28199060e+00,   1.20811900e-01,  -1.54092350e+00,
          -1.72598790e+00,  -9.74563700e-01,  -3.70346400e-01,
          -7.41189600e-01,  -7.50126700e-01,  -1.81216030e+00,
           4.91239500e-01,  -2.32962350e+00,  -2.46623430e+00,
          -3.40758700e-01,  -4.39576200e-01,   6.37400100e-01,
           7.71144400e-01,  -8.96302200e-01,  -1.23877900e-01,
          -1.03116000e+00,   1.01917950e+00,   6.48057600e-01,
          -8.42242100e-01,  -5.08729000e-01,   1.15782660e+00,
          -2.34323060e+00,   4.44566900e-01,  -9.34687400e-01,
          -5.01903000e-02,  -1.86966780e+00,   2.46951600e-01,
           7.53497900e-01,  -2.94312320e+00,  -1.82730510e+00,
           3.16193000e-01,   5.46405200e-01,   1.77208900e-01,
          -7.94137000e-02,   9.64200000e-03,  -2.78566310e+00,
          -6.03403900e-01,   5.76780100e-01,  -1.59133100e-01,
          -6.72166100e-01,  -2.35378780e+00,  -1.75017200e-01,
          -7.19764300e-01,  -2.08057520e+00,   7.66202200e-01,
          -2.39659100e+00,   9.01635800e-01],
        [  9.71175000e-01,  -7.22018000e-01,   2.75999000e-02,
           8.89937600e-01,   6.88292100e-01,  -3.13797400e-01,
          -5.73781500e-01,  -1.45473300e-01,   9.41801600e-01,
          -4.30634600e-01,  -9.84352000e-02,  -2.26136100e-01,
          -1.03621600e+00,  -6.75447920e+00,  -1.61152460e+00,
          -4.36776810e+00,  -1.17778900e-01,  -5.91529500e-01,
          -1.30854730e+00,  -2.26482110e+00,  -1.67016680e+00,
          -8.14718200e-01,   3.82450900e-01,   9.01635800e-01,
          -1.19880500e-01,  -9.37662100e-01,  -6.45676500e-01,
          -3.46543200e-01,  -4.78394800e-01,  -1.86694500e-01,
          -5.06705000e-02,  -1.83997560e+00,   6.73733500e-01,
          -2.23572500e-01,  -1.59903370e+00,  -9.39731800e-01,
          -1.01185390e+00,   4.64187000e-01,   8.94875100e-01,
           1.30803440e+00,   1.32171200e-01,  -1.22728660e+00,
          -8.78502600e-01,  -6.43728100e-01,  -1.86334630e+00,
          -1.79733480e+00,  -5.86973900e-01,  -3.31005010e+00,
           1.45962930e+00,  -1.66505150e+00],
        [ -7.93710000e-03,  -6.02862000e-02,  -1.04483110e+00,
           2.91308500e-01,   8.73817700e-01,   6.76412000e-02,
           1.27849290e+00,  -2.34395900e-01,   1.13211000e+00,
           2.52433700e-01,  -2.74753300e-01,  -1.80776500e-01,
          -1.16862240e+00,   7.66202200e-01,  -1.84930020e+00,
          -3.52001420e+00,   9.08193000e-02,   5.46405200e-01,
          -5.31728200e-01,   1.24534900e-01,  -7.75839400e-01,
          -9.91780600e-01,   6.62270900e-01,   1.14667500e-01,
           2.74034000e-01,  -8.26739300e-01,  -7.45566600e-01,
           2.19851900e-01,  -2.30427340e+00,  -8.32127100e-01,
          -7.19128700e-01,   1.02900400e+00,  -1.03977100e-01,
          -1.59903370e+00,   3.35979600e-01,  -6.83327900e-01,
          -1.24900990e+00,  -4.03118760e+00,  -2.53913070e+00,
          -2.17443380e+00,  -1.51413660e+00,  -1.84723700e-01,
          -1.55568680e+00,  -7.64884200e-01,  -1.45114870e+00,
          -9.79623200e-01,  -8.78502600e-01,  -5.06705000e-02,
          -5.33983930e+00,  -2.36735100e-01],
        [ -1.82415800e-01,  -2.20235610e+00,   1.09762060e+00,
           2.68110200e-01,  -4.18352800e-01,  -5.24858500e-01,
          -1.25765600e-01,  -1.03253160e+00,  -7.94815400e-01,
          -1.04418160e+00,   7.28711100e-01,  -2.62050000e-02,
          -3.19119100e-01,  -1.25460400e-01,   4.58604300e-01,
           6.24981600e-01,  -2.57393300e-01,   2.32883100e-01,
           5.22521000e-02,  -7.85195600e-01,   1.15315290e+00,
           7.96335900e-01,   3.35979600e-01,   3.70449600e-01,
          -1.82415800e-01,   6.43113100e-01,   4.73345500e-01,
           1.04419220e+00,   9.36578100e-01,  -1.04483110e+00,
          -5.80931000e-02,  -2.55194400e-01,   3.74592700e-01,
           4.19781100e-01,  -1.04064870e+00,  -6.84642800e-01,
           4.33388900e-01,  -3.95452700e+00,  -2.60491800e-01,
          -2.10710910e+00,  -1.59903370e+00,  -2.55183370e+00,
          -3.47564620e+00,  -9.72010100e-01,   1.50038500e-01,
          -4.24479000e-02,   5.36067400e-01,  -8.92509500e-01,
           1.39881300e-01,  -6.54566200e-01]],

       [[ -4.62562000e-02,   1.47979700e-01,  -3.43605000e-02,
          -9.38623700e-01,  -6.40906000e-02,  -1.30582930e+00,
           5.61495500e-01,  -1.83997560e+00,  -6.30410000e-02,
          -2.71730000e-02,  -1.84840730e+00,  -2.11183250e+00,
          -2.29516460e+00,   7.38068800e-01,  -1.22217550e+00,
          -3.98857710e+00,   5.53731400e-01,   3.11385400e-01,
          -1.87181600e+00,   3.58959800e-01,  -4.01322600e-01,
          -1.41589060e+00,  -2.73557670e+00,  -2.01713830e+00,
           5.22521000e-02,  -2.88082600e-01,  -5.89593600e+00,
          -7.89493900e-01,  -2.82890600e-01,  -4.68413350e+00,
          -3.33905400e-01,   1.42153870e+00,  -1.20079170e+00,
           1.35792810e+00,  -2.44847930e+00,  -5.89554100e-01,
          -1.96721200e-01,   8.37761500e-01,  -2.70119000e-01,
           7.57884300e-01,  -6.39962900e-01,   1.29544510e+00,
          -1.03621600e+00,  -1.49007920e+00,   8.76656100e-01,
           2.87457500e-01,   4.76203000e-01,  -3.15793100e-01,
          -2.61807800e-01,  -2.92704330e+00],
        [  1.11562690e+00,  -1.21512000e-02,   6.44475600e-01,
          -8.40612500e-01,   1.12211520e+00,  -1.73891980e+00,
           3.94622900e-01,  -1.23838620e+00,  -1.12551470e+00,
          -1.47949860e+00,  -8.89654700e-01,  -1.16475830e+00,
          -1.62577910e+00,   4.56974600e-01,  -4.40398000e-02,
           6.94112000e-02,   6.91456800e-01,   5.69848400e-01,
          -9.75858000e-01,   7.25556000e-01,  -1.29888590e+00,
          -1.43645900e+00,   6.17938200e-01,   9.67380700e-01,
          -4.77052480e+00,   1.09645200e+00,  -1.06817750e+00,
          -4.93626400e-01,  -1.75203890e+00,   1.50353000e-01,
           6.00998300e-01,  -2.53521960e+00,  -4.92193400e-01,
           1.96316100e-01,  -5.01903000e-02,   1.96316100e-01,
           1.04856970e+00,  -1.07260100e+00,  -1.69865200e+00,
           1.17264200e-01,   5.69848400e-01,   4.27740000e-03,
          -7.13078800e-01,  -4.55030100e-01,  -2.73350650e+00,
           3.45681300e-01,   6.12679800e-01,   6.23678800e-01,
           2.74327800e-01,   1.53264620e+00],
        [ -6.32415100e-01,   2.25943100e-01,  -7.40090400e-01,
          -2.13168700e-01,  -1.70538960e+00,   1.23521680e+00,
          -7.00699200e-01,  -1.11981760e+00,  -1.45215700e-01,
          -7.78995000e-01,   5.08734500e-01,  -5.59483600e-01,
          -4.67554940e+00,   1.01892330e+00,  -2.74186800e-01,
          -1.93156800e-01,   1.10445850e+00,  -1.85939090e+00,
           9.67380700e-01,  -1.31017720e+00,  -1.19048320e+00,
          -1.41193030e+00,   1.62245030e+00,  -3.32001030e+00,
           8.81990400e-01,   8.59682800e-01,   1.62245030e+00,
          -2.66099270e+00,  -2.30349360e+00,   1.97204200e-01,
           1.61647830e+00,   4.91110000e-01,  -1.84493290e+00,
           5.15978300e-01,   6.43209800e-01,   9.32221100e-01,
           1.17152950e+00,  -1.03278990e+00,  -2.55600100e-01,
          -4.36819790e+00,   5.67564600e-01,  -8.93237400e-01,
           9.70896800e-01,  -2.75180000e-03,   4.20503200e-01,
           6.79001000e-01,  -5.12790000e-02,   7.17271000e-02,
           1.01620000e-01,   5.04776500e-01],
        [ -1.41763100e-01,   6.34473800e-01,  -2.34471800e-01,
          -1.02896670e+00,  -7.02813000e-02,  -5.51661000e-01,
          -8.66525200e-01,  -2.60387500e-01,  -1.39214200e+00,
           6.33698800e-01,   4.24838100e-01,  -1.49580310e+00,
          -3.04913610e+00,  -2.19220650e+00,   1.02809980e+00,
          -1.89564500e-01,   7.86724400e-01,  -5.18078000e-02,
           2.41604200e-01,  -6.86838200e-01,  -1.36845000e+00,
          -2.08057520e+00,  -5.11232400e-01,   6.86006000e-01,
          -1.51145030e+00,   1.87748900e-01,   4.46380500e-01,
          -2.86737500e-01,   1.72610000e-01,  -8.88650500e-01,
          -4.92400800e-01,   1.29775260e+00,   1.56630700e-01,
           1.52078530e+00,   7.14006000e-02,  -9.56497200e-01,
          -2.52989100e-01,   6.01681400e-01,  -3.64554000e-01,
          -1.15173350e+00,   6.94560000e-02,  -5.76349560e+00,
          -4.92400800e-01,   3.19561100e-01,   7.50342000e-02,
          -1.09422600e-01,   1.21643910e+00,  -1.57872890e+00,
          -5.45301230e+00,  -5.82003500e-01],
        [ -1.03796570e+00,  -2.82204740e+00,  -2.39050610e+00,
           1.20426070e+00,  -2.17527750e+00,  -7.98768500e-01,
           6.01065800e-01,   4.97720200e-01,  -1.00460950e+00,
          -1.26898610e+00,   1.40359880e+00,  -1.35535680e+00,
           2.59450100e-01,   2.66454000e-02,  -2.65395700e-01,
           6.36863000e-02,   2.94193000e-01,  -3.83398000e-01,
          -2.01892730e+00,  -1.02524530e+00,  -5.97179200e-01,
           9.17723100e-01,  -4.95338600e-01,  -2.07370300e-01,
          -5.50657000e-02,  -5.97925100e-01,   1.03008390e+00,
           2.22956100e-01,  -1.12306840e+00,   5.49934100e-01,
           1.01632080e+00,   4.87782100e-01,  -2.52989100e-01,
           2.79672300e-01,  -1.49336100e-01,   4.73345500e-01,
          -7.21654000e-02,  -1.27402390e+00,  -6.85492100e-01,
          -4.36819790e+00,  -3.46929300e-01,  -3.80331600e-01,
          -6.34030900e-01,  -1.28582310e+00,  -3.14421100e-01,
          -7.55009300e-01,  -1.17932600e-01,   9.89473000e-01,
          -1.54592180e+00,  -6.38959000e-02]]])

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
    exo_utility = np.empty([dict['I'] + 1, dict['N'], dict['R']])
    for n in range(dict['N']):
        for i in range(dict['I'] + 1):
            for r in range(dict['R']):
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

    # Beta coefficient for endogenous variables
    beta_FEE_PSP = np.empty([dict['N'], dict['R']])
    beta_FEE_PUP = np.empty([dict['N'], dict['R']])
    for n in range(dict['N']):
        for r in range(dict['R']):
            beta_FEE_PSP[n, r] = (dict['Beta_FEE'][n, r] +
                                 dict['Beta_FEE_INC_PSP'] * dict['Low_inc'][n] +
                                 dict['Beta_FEE_RES_PSP'] * dict['Res'][n])
            beta_FEE_PUP[n, r] = (dict['Beta_FEE'][n, r] +
                                 dict['Beta_FEE_INC_PUP'] * dict['Low_inc'][n] +
                                 dict['Beta_FEE_RES_PUP'] * dict['Res'][n])
    dict['endo_coef'] = np.array([np.zeros([dict['N'], dict['R']]), beta_FEE_PSP, beta_FEE_PUP])

    # Calculate bounds on the utility
    lb_U = np.empty([dict['I'] + 1, dict['N'], dict['R']])
    ub_U = np.empty([dict['I'] + 1, dict['N'], dict['R']])
    lb_Umin = np.full((dict['N'], dict['R']), np.inf)
    ub_Umax = np.full((dict['N'], dict['R']), -np.inf)
    M = np.empty([dict['N'], dict['R']])
    for n in range(dict['N']):
        for r in range(dict['R']):
            for i in range(dict['I'] + 1):
                    if dict['endo_coef'][i, n, r] > 0:
                        lb_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['lb_p'][i] +
                                        dict['exo_utility'][i, n, r] + dict['xi'][i, n, r])
                        ub_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['ub_p'][i] +
                                        dict['exo_utility'][i, n, r] + dict['xi'][i, n, r])
                    else:
                        lb_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['ub_p'][i] +
                                        dict['exo_utility'][i, n, r] + dict['xi'][i, n, r])
                        ub_U[i, n, r] = (dict['endo_coef'][i, n, r] * dict['lb_p'][i] +
                                        dict['exo_utility'][i, n, r] + dict['xi'][i, n, r])
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

if __name__ == '__main__':
    dict = getData()
    preprocess(dict)
