from sklearn.cluster import KMeans
import numpy as np
import Data.Non_linear_Stackelberg.ProbLogit_n10 as data_file

def getData():
    ''' Construct a dictionary containing all the data_file
        Returns:
            dict          dictionarry containing all the data
    '''
    # Get the customers socio-economic characteristics
    dict = data_file.getData()

    # Data
    customers = np.column_stack((dict['Origin'], dict['Age_veh'], dict['Low_inc'], dict['Res']))
    import IPython
    IPython.embed()

    # Compute the clusters
    kmeans = KMeans(n_clusters=10, random_state=0).fit(customers)

    # Update the data
    dict['N'] = 10
    dict['Origin'] = kmeans.cluster_centers_[:,0]
    dict['Age_veh'] = kmeans.cluster_centers_[:,1]
    dict['Low_inc'] = kmeans.cluster_centers_[:,2]
    dict['Res'] = kmeans.cluster_centers_[:,3]
    dict['N_size'] = []
    for k in range(10):
        dict['N_size'].append((kmeans.labels_ == k).sum())

    import IPython
    IPython.embed()

    return dict

def preprocess(dict):
    data_file.preprocess(dict)
