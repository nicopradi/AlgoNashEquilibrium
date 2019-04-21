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

    # Number of coef draws
    dict['R_coef'] = 100

    # Lower and upper bound on prices
    dict['lb_p'] = np.array([0, 0.0, 0.0]) # lower bound (FSP, PSP, PUP)
    dict['ub_p'] = np.array([0, 1.0, 1.0]) # upper bound (FSP, PSP, PUP)

    #dict['capacity'] = np.array([60.0, 4.0, 4.0]) # Availability for each alternative (opt-out always available)
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
    dict['Beta_AT'] = np.array([[-1.04651716e+00, -1.43355942e+00,  1.05182280e-02,
        -1.77672189e+00, -8.68817239e-01, -1.82034347e+00,
         5.14431041e-01,  9.08433360e-01, -1.51215475e+00,
        -1.47320759e+00, -1.35704393e+00, -2.33442000e+00,
        -1.08662441e+00, -5.29191186e-01, -4.57403187e-01,
        -8.61298623e-01,  1.87025192e-01,  1.60723723e+00,
        -5.19362601e-01,  7.44357328e-01, -1.62132336e+00,
        -2.67529405e+00, -1.76574989e+00, -3.04180428e-01,
        -1.51108263e+00,  2.41332913e-01, -1.51277300e+00,
        -9.37836055e-02, -2.02435246e+00,  1.18983693e+00,
        -1.38613989e+00, -1.85359034e+00, -1.31939429e-01,
        -7.97227970e-01, -2.47422885e+00, -8.32650135e-01,
        -4.28166497e-01, -3.79488742e+00, -2.36412974e+00,
        -1.41412182e+00, -5.54764075e-01,  3.97061254e-01,
         2.57103748e-02, -6.74004798e-01,  2.96020936e-01,
        -1.44478268e+00, -1.06318955e+00, -1.90116757e+00,
        -2.57280543e+00,  8.30027165e-01, -3.04792740e-01,
        -8.56497267e-01, -7.09456113e-02, -2.02698969e+00,
         1.44945881e+00, -1.07722732e+00,  1.04723338e+00,
        -3.21559924e+00, -1.61362763e+00, -3.10889493e+00,
        -5.66907264e-01, -9.80431304e-02, -7.36220660e-02,
        -7.64977684e-01, -4.56449291e-01, -2.83115518e-01,
        -5.48533594e-01,  1.20906085e-01, -6.39459437e-02,
         4.42422617e-01, -2.06260307e+00, -5.37722327e-02,
        -1.15727203e+00, -6.33064270e-01,  1.38295976e-01,
         4.50987001e-01, -1.49191306e+00, -2.54343885e+00,
        -2.23258726e-01, -4.10200246e-01, -9.52133665e-01,
        -1.10490952e+00, -2.20601792e+00, -2.84005707e+00,
        -6.86097727e-01,  3.91565455e-01, -3.88891868e-01,
         3.75040932e-01, -6.38058520e-01,  3.73278395e-01,
         6.15690309e-01, -2.14413690e-01, -9.64280096e-01,
         1.01565616e+00, -2.44078889e+00, -2.13163415e-01,
        -5.58795928e-01,  6.51395442e-01, -1.07204787e+00,
        -1.37543250e+00],
       [-1.84911247e+00,  3.20461442e-01, -2.54631709e+00,
         2.85370066e-01, -3.21639012e-01,  1.87984706e-01,
        -7.44138260e-01, -2.13775430e+00, -5.83271826e-01,
        -4.28056625e-01,  1.49519414e-01, -2.33711207e+00,
         1.52367738e+00, -1.38957142e+00,  5.83394865e-01,
        -9.63845965e-01,  1.49961709e-02, -1.90784774e+00,
        -6.55301187e-02, -2.63412145e+00, -5.28963512e-01,
        -2.35483371e-01, -1.72561889e+00, -4.08048967e-01,
        -2.15245899e+00, -5.91638959e-01, -1.68853963e+00,
        -7.64085673e-01, -1.08809540e+00, -2.16523003e+00,
        -4.33584523e-01,  6.09493943e-01,  9.38121701e-01,
         1.35694122e-01, -2.12573043e-01, -1.82382518e+00,
         2.46171649e-01, -1.40805313e+00,  1.67595025e-01,
        -1.76455415e+00, -2.83332848e+00, -5.53958765e-01,
        -1.87893410e+00, -9.15498522e-01, -8.58603583e-01,
        -2.55762830e+00, -2.65265091e-01,  2.94680257e-01,
         1.44533867e+00,  3.96298939e-01,  1.05495543e-01,
        -8.24342738e-01, -4.09821664e+00, -2.46867817e+00,
         2.21743719e-01, -1.65569208e+00, -9.90224334e-01,
        -6.94451828e-01, -1.79021754e+00, -1.51152371e+00,
         5.42552999e-01, -1.61314756e+00,  1.56139549e-01,
        -1.58619688e+00, -1.48233639e+00, -2.98539728e+00,
         3.38818697e-01,  6.01518505e-01, -2.52870186e+00,
        -5.20319175e-01, -1.12134344e-01, -3.61852605e-02,
        -1.46869963e+00, -2.14069699e+00, -1.27692666e+00,
        -1.54933860e+00, -4.96156209e-01, -2.47499133e-01,
        -3.95602763e-01, -1.45125468e+00, -1.19811200e+00,
        -1.17649884e+00, -1.74822310e+00,  8.98534132e-01,
        -2.85846646e+00,  2.08741490e-01,  8.21600039e-01,
         9.22902847e-01,  8.60325446e-01, -3.01767241e+00,
        -1.49814386e+00, -1.99893069e+00, -1.15349088e+00,
        -1.28793174e+00, -1.59444777e+00, -1.36373903e+00,
         4.64386598e-01, -2.15367684e+00, -9.58222932e-02,
         9.71054957e-02],
       [-5.29953375e-01, -3.14285035e+00,  7.92497688e-01,
        -9.64340282e-01, -6.97705509e-01, -2.86632944e+00,
         1.54329427e+00, -6.62531334e-01, -3.26200436e-01,
        -1.73220287e+00,  1.04298131e+00, -1.40831374e+00,
        -1.88688744e+00, -2.67979268e+00, -1.60324271e+00,
        -6.70394862e-01,  1.81383170e-01, -4.81458802e-01,
         3.70867633e-01,  3.92884029e-01, -2.36045976e-01,
         8.63150078e-01, -5.45681554e-02, -5.92096669e-01,
        -9.34157280e-01, -6.21624737e-01, -5.74291575e-01,
        -2.17583796e+00, -1.42278683e+00,  2.12954795e+00,
        -1.17080065e+00, -2.19685742e-01, -1.32817525e+00,
        -2.38986663e+00,  6.84573444e-01, -9.31658173e-01,
        -7.24918614e-01, -2.15967172e-01, -5.40996297e-01,
        -1.54364776e+00,  4.89985182e-01, -1.12388328e+00,
        -1.27766469e+00, -1.67377057e-01, -2.62569101e+00,
        -6.78141030e-02, -1.84491624e+00, -1.13095793e+00,
        -3.27384521e-01,  3.98200006e-01, -9.38629852e-02,
        -1.04645855e+00, -4.88538084e-01,  4.21869725e-02,
        -8.28511826e-01, -1.85720844e+00, -8.39596898e-01,
        -8.10695236e-01, -2.18470910e+00, -5.77104779e-02,
         1.48216375e+00, -1.27466699e+00, -3.86932401e-01,
        -1.62858985e+00,  7.10401301e-01, -1.27051693e+00,
         1.79251799e-02, -2.07336760e+00, -1.73742079e-02,
         1.60123629e+00,  6.82001813e-01, -1.03479669e+00,
        -1.54411508e+00, -1.05858070e+00,  2.81090979e-01,
        -1.19385737e+00, -1.18926163e+00,  4.92899989e-02,
         5.02388283e-01, -5.37105600e-01, -4.67273975e-01,
        -1.21684507e+00, -2.34319179e-02, -1.86551848e+00,
        -5.23859957e-01,  5.62081241e-01,  1.47560225e-01,
        -2.14709142e+00, -6.07905290e-01,  3.27514591e-01,
        -1.51653225e+00, -1.28705762e+00, -5.17754606e-01,
        -2.63292474e+00, -2.46805768e+00, -9.67320671e-01,
        -1.48157682e-01, -1.60722953e+00, -9.53020850e-01,
        -4.40268482e-01],
       [-7.86307316e-01, -3.89425335e-02,  1.87210043e+00,
        -2.25515908e+00,  1.44176223e-01, -7.73022592e-02,
         3.15066182e-01, -1.69409463e+00, -5.91190828e-01,
         1.48054921e+00, -1.21935258e+00, -7.99866005e-01,
        -1.02465853e+00, -9.72340686e-01, -1.79980613e-01,
        -2.63112492e+00,  1.37682157e+00, -2.58085877e+00,
        -2.00730548e+00, -1.60579618e+00,  1.67564550e+00,
        -1.66849416e+00, -4.91838355e-02,  1.82047385e+00,
        -1.36646302e+00,  8.89627469e-02, -8.34839665e-01,
         8.47992486e-01, -2.47299355e+00, -1.53668539e+00,
        -1.42985162e+00, -2.67397637e-01,  3.67296013e-01,
        -6.07168380e-01, -8.80769926e-01, -8.98835783e-01,
        -1.16767403e+00,  1.30789942e+00, -2.24594161e-01,
         6.59886916e-01,  4.14963738e-01,  7.77845116e-02,
        -3.59439684e-01, -4.39838687e-01, -2.42448440e+00,
        -3.46189267e-01,  1.52393408e-01, -2.98633752e-02,
         1.30572518e+00, -7.40628459e-01, -7.33382295e-01,
         2.94562690e-03, -1.75881538e+00, -1.15337958e+00,
        -1.89840674e+00, -1.72306040e+00, -4.32305091e-01,
        -4.63925945e-01, -6.84932814e-01,  1.92200059e-01,
        -1.67208704e+00,  1.73459701e-01, -6.88976440e-01,
         6.23996844e-01,  4.53852602e-01, -1.13230056e+00,
         7.36311279e-01, -1.39068815e-01, -6.50367023e-01,
        -1.90146695e+00, -1.08627758e+00,  2.11669495e-01,
         5.35994466e-01, -1.33177323e-01, -4.25148276e-01,
         7.44737260e-01, -9.30948387e-01, -1.49769961e+00,
         1.26395935e+00, -9.54992174e-01,  6.95768686e-01,
         7.97170436e-02, -1.78888786e+00, -9.64327236e-01,
         3.55243239e-01, -1.71508216e+00, -1.15807612e+00,
        -1.21332394e+00,  3.26741068e-01, -2.95387214e+00,
        -2.27046341e+00, -7.16485375e-01,  1.49945833e-02,
        -1.12459281e+00,  3.64352756e-02, -1.16970616e+00,
        -3.27845261e+00, -1.55337011e+00, -8.85639560e-03,
        -2.08361789e+00],
       [-1.12165159e+00,  9.19110211e-01,  4.74954120e-01,
        -1.31648569e+00,  3.45459923e-01, -3.48267153e-02,
        -1.21728433e+00, -2.34838063e-02, -1.09023662e+00,
         3.12454477e-01, -7.95885348e-01,  8.84429570e-02,
        -7.87099250e-01, -2.13337315e+00, -7.36720889e-01,
        -9.03419515e-01,  9.30617651e-01, -2.83391459e+00,
        -4.95638578e-01, -1.10447921e+00, -8.52776400e-04,
        -5.96293806e-01,  4.61250868e-02, -1.03570964e+00,
        -2.22278126e+00, -1.65122749e+00, -5.10603614e-02,
         3.68955813e-01, -1.74417620e+00,  1.25256114e+00,
         2.71272672e-01, -1.52052974e+00, -1.74605993e+00,
         2.05841221e-01, -3.56210401e-01,  2.11551951e-02,
        -1.06392287e+00,  6.00055664e-01, -1.05599039e+00,
        -1.77129668e-02,  5.57494296e-01, -4.37794873e-02,
        -1.65916417e+00, -2.07385111e+00, -1.15691790e+00,
        -3.99042569e-01, -1.57579374e+00, -4.09929642e-01,
         5.26414402e-01, -9.26245772e-01, -2.20102175e+00,
        -3.53630445e-01,  6.04147018e-01, -3.51612238e-01,
        -7.29666163e-01, -1.47947251e-02, -1.35488547e+00,
        -1.20831004e-01,  2.12701586e+00, -2.23312678e+00,
        -4.98089478e-01, -5.01100448e-01, -2.36009631e+00,
        -3.45547043e+00, -3.34506912e-01,  5.57468170e-01,
        -1.30886487e+00, -1.29162319e+00, -7.48695567e-01,
        -2.44380926e+00, -1.20584315e+00, -1.19951764e+00,
        -7.97684565e-01,  8.37373439e-01, -1.50418547e+00,
        -2.75436919e-01,  4.77892141e-01,  1.36289523e+00,
         3.42069056e-01, -7.85962520e-02, -1.18773183e-02,
        -2.01349171e+00, -1.39490543e+00, -1.51747004e+00,
         5.98014962e-01, -3.54271607e-01,  2.05719340e-01,
        -1.94577827e+00, -1.59359423e+00,  4.12099336e-01,
         5.51757220e-01, -1.06284978e+00, -8.52318810e-01,
        -1.21870985e+00, -1.04327823e+00,  6.99003821e-02,
        -1.10293355e+00, -1.34522685e+00, -1.01034102e+00,
        -1.13894476e-01]])

    # FEE coefficient
    dict['Beta_FEE'] = np.array([[-16.72076372, -30.52968169, -11.36375956, -33.27107748,
        -47.07708929, -39.63335947, -28.63756694, -20.05463694,
        -47.30359923, -39.82017407, -22.30518034, -14.12460446,
        -34.01247327, -31.54089895, -44.37636148, -26.43791041,
        -33.56190634, -14.27810587, -50.27476673, -30.46474093,
        -39.74907519, -43.70100216, -13.0187184 , -13.34387561,
        -32.52601199, -27.17934972, -38.46999886, -27.76410284,
        -43.85476291, -35.86894772, -36.42286589, -22.26395817,
        -21.77125725, -38.26169879, -28.37567423, -35.72300207,
        -33.19004574, -32.75898102, -33.26088363, -44.65136212,
        -51.86169357, -27.55350836, -25.62177461, -58.8994035 ,
        -16.38731948, -65.13977463, -25.46307253, -18.4679285 ,
        -29.43728026, -42.56516709, -25.09519218, -20.34004891,
        -23.70811098, -20.86977146, -33.23489487, -42.09944949,
        -19.49367048, -39.17233588, -58.90646344, -48.51639277,
        -62.40210234, -40.23612449, -14.34836728, -12.68684703,
        -16.46789021, -31.25337361, -29.88027789, -24.01888113,
        -36.43172859, -39.56939299, -23.81331323, -14.87463119,
        -32.0660981 , -23.18252514, -27.37809728, -26.28504988,
        -20.01029154,  -5.51871175, -18.78760663, -26.99266928,
        -41.15989799, -30.99490778, -42.72538601, -41.2995425 ,
        -43.9925115 , -40.24257179, -29.59904832, -42.7746798 ,
        -46.35215131, -35.26087515, -41.18364355, -39.96070656,
        -25.89903906, -38.11763677, -42.81651857, -58.13777049,
        -32.11344963, -55.49540767, -24.33979206,  -8.92566598],
       [-17.54021331, -15.12316287, -35.79341496, -49.37672178,
        -48.61317635, -24.21725348, -53.55406835, -31.93989687,
        -23.78142171, -43.19436195, -38.77776461, -26.86162043,
        -34.95851546, -14.31477732, -41.63857595,  -9.02355808,
        -39.76808735, -40.41752186, -13.3792003 , -43.2965329 ,
        -32.05595443,   2.09245715, -19.10559549, -12.13281385,
        -50.19449047, -31.26317714, -47.76486979, -35.86674197,
        -42.97130888, -43.37547485, -23.3910537 , -27.28901409,
        -30.99000132, -39.43595654, -55.12599224, -20.31185884,
        -56.90602033, -25.48448232, -43.26788486, -20.09141451,
        -36.15404727, -49.46727186, -42.76316388, -29.5212917 ,
        -46.40921817,  -9.58613268, -64.98508483, -33.54604363,
        -30.19243145, -20.04911425, -22.54530975, -49.21921934,
        -38.6940044 , -19.86662798, -13.71625612, -35.57447802,
        -17.81120594,  -9.13001831, -47.33488069, -40.31452175,
          3.56618814, -17.05317892, -39.42657848, -34.38283092,
        -23.98884247, -43.82497623, -54.66062341, -13.08230297,
        -36.44181779, -36.31444152, -28.87351494, -55.5523404 ,
        -46.75192337, -44.65745156, -51.45087569, -41.15020229,
         -8.58588808, -18.00224429,  -9.69322999, -54.08329253,
        -32.17479288, -48.62744858, -33.64868783, -40.15758001,
        -22.45493341, -29.60354142, -26.51442823, -29.29815419,
        -41.83239305, -39.84031097, -19.08442954, -27.8031057 ,
        -32.14619091, -24.17093315, -36.14322682, -23.73344759,
        -51.69005549, -37.82515077, -50.9303476 , -33.37427755],
       [-49.75079566, -12.11171083, -25.31315991,   1.41786225,
        -55.51286933, -26.17086557, -34.71771429, -26.3121658 ,
        -38.06358476, -29.95877878, -37.16195001, -14.79885787,
        -47.85890978, -45.7968104 , -31.0667196 , -72.70307771,
        -43.47172987, -24.53588615, -75.08419668, -39.00037403,
        -39.19098562, -28.70057977, -26.42431976, -51.27332018,
        -27.2152268 , -17.34015805, -16.93177416, -24.11184894,
        -24.62413498, -48.81271361, -18.14374348, -39.76407884,
        -25.52499681, -19.03536004, -32.69648736, -30.56802884,
        -22.88134776, -63.62144665, -33.46423836, -29.79501129,
        -57.85817525, -10.99290905, -39.87636249, -17.65720006,
        -23.94888463, -49.66653491, -25.52653846, -18.22930585,
        -35.68641265, -26.44527578, -23.93205916, -34.60891881,
        -22.10237103, -53.02338045, -15.84438805, -11.48260281,
        -28.69518439, -17.47066888, -26.6901612 , -49.98823682,
        -18.71022123, -68.8946728 ,  -3.49442445, -34.1434001 ,
        -32.59799472, -31.93279945, -26.185376  , -39.26922304,
        -45.31708036, -52.01962169, -58.85835188, -41.94382298,
        -23.15277647, -24.24896048, -43.13152549, -24.11050532,
        -30.21788838,  -1.22805956, -35.10860692, -19.98280632,
        -17.5691471 , -41.95961304,  -2.47630606, -42.6242205 ,
        -35.77440743, -40.05069296, -23.53966638, -11.77136395,
        -42.34872922, -39.35147121, -32.55705023,  -8.56716193,
        -28.23791421, -37.59139327, -11.64209971, -34.57150357,
        -30.53413652, -14.13982242, -46.28580916, -41.60494771],
       [-41.9492828 , -41.52008614, -58.76346507, -48.69766056,
        -27.14115953, -53.84407476, -30.44679758, -29.02423251,
        -17.63293468, -23.81392096,  -9.74559687, -64.64396043,
        -15.82433649, -33.99671368, -17.17221455, -33.60446454,
        -45.5456763 , -28.92324495, -14.74397356, -18.24853524,
        -27.62755764, -29.53678207, -37.64101289,   0.77827596,
        -33.62210524, -22.63431479, -41.89152139, -39.11326563,
        -61.25066125, -33.13778818,  -8.82283849, -16.29095747,
        -18.16450234, -21.26962427, -39.11605636, -40.73142379,
        -64.68442174, -44.94299721, -12.27371548,   0.22323249,
        -20.88778773, -16.55793712, -13.81576507, -27.90574019,
        -38.4199395 , -42.2892478 , -26.57122766, -22.06845887,
         -3.39115728, -33.31791203, -43.34117226, -55.72152417,
        -38.79295377, -23.49184591,   3.33863731, -18.40849118,
        -35.25824536, -19.97291127, -36.85836955, -23.59549891,
         -9.67712447, -13.67881793, -26.80350467, -15.59581061,
        -41.30528689, -17.57446058, -39.32026672, -60.84004278,
        -28.83107473, -39.82642436, -25.82247633, -27.33896661,
        -65.4265305 , -37.08123023, -34.05007386, -42.30792297,
        -47.98641291, -16.19964672, -29.62927449, -34.1561229 ,
        -39.25568244, -38.70859109, -42.18638154, -33.34721341,
        -37.56339183, -19.62886761, -46.70537364, -58.64859596,
        -52.31183774, -37.45913319, -11.12544965, -26.72391215,
        -18.9294663 , -27.54635371, -37.82614507, -34.89566384,
        -24.27382843, -45.54950483, -38.99790683, -34.10696554],
       [-33.00331742, -42.18741328, -43.7705777 , -56.40541515,
        -26.5750357 , -34.88633087, -47.1368845 , -37.61062043,
        -18.79632019,  -8.58588808, -53.16486926, -36.01771935,
        -38.26330636, -19.91006778, -33.38645666, -17.23725211,
        -32.49500521, -11.82440456, -44.48441461, -48.89163047,
        -17.86669973, -64.76045163, -37.85398831, -34.080128  ,
        -23.5615519 , -30.57256916, -30.9379882 , -67.07055596,
        -42.51266242, -20.80552148, -30.99115297, -12.55886089,
        -48.80208715, -13.18677681, -27.07287841, -50.17045051,
        -51.6450038 , -67.75476877, -32.27056665, -17.74844783,
        -17.91983876, -60.5546943 , -17.04815049, -62.8310541 ,
        -15.65463029, -21.03928342, -24.14904332, -29.96899226,
        -36.87431468, -22.1105859 , -47.88999024,  -8.51018673,
        -39.11957158, -31.87576712, -31.8448157 , -25.61368049,
        -23.89666474, -48.91934674, -27.9329219 , -37.37614262,
        -53.39784156, -23.22130411, -17.05577334, -41.3250611 ,
        -27.87492857, -27.71276151, -26.70954249, -38.04881479,
        -59.73897735, -43.32280538, -12.82001572, -27.7107322 ,
        -46.68410863, -52.10838205, -15.35303724, -20.13704727,
        -48.97029543, -51.42643192, -41.48279124, -13.86068908,
        -18.61200549, -59.35274752, -47.86174852, -46.0653862 ,
        -24.12528954, -47.95833703, -38.98584678, -44.38040429,
        -25.43162973, -36.51292809, -18.91606703, -31.13400436,
        -48.76246495, -50.26340628, -67.99770816, -23.90194112,
        -41.20717176, -48.82994486, -21.56979181, -47.82764693]])

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
