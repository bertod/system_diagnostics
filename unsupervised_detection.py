#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import shelve
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

import data_manager
import data_viewer
import data_labeler
import signal_processor


#%%
def import_data():
    fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
            '2019-02-04 00:00:01', '2019-02-10 00:59:59', 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)

    df = fabbri1905_fabax6pdb.measure_pd_joined_dataframe
    #df.head()
    return df

def import_data_event():
    fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
            '2019-02-04 00:00:01', '2019-02-10 00:59:59', 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)

    df_event = fabbri1905_fabax6pdb.measure_pd_dataevent_samples[0]
    df_event.head()
    return df_event

def import_data_events():
    fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
            '2019-02-04 00:00:01', '2019-02-10 00:59:59', 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)

    list_event = fabbri1905_fabax6pdb.measure_pd_dataevent_samples
    return list_event


def get_event_microfeatures(list_events,feature_engineering):
    if feature_engineering == True:
        ## work on all events - shape: n.events x 36features (mean + variance)
        events_dict = {}
        for j, event in enumerate(list_events):
            if j in [250,500,750,900,1000,len(list_events)-1]:
                print('Dataframe n',j)
            microfeatures_list = []
            event_timespan = []
            count = 0
            for i in event.columns:
                microfeatures_list.append(event.loc[:,i].mean())
                microfeatures_list.append(event.loc[:,i].std())
            event_timespan.append(str(event.index[0]))
            event_timespan.append(str(event.index[len(event)-1]))
            events_dict[' - '.join(event_timespan)] = microfeatures_list
        df_events_microfeatures = pd.DataFrame(events_dict)
        #return df_events_microfeatures
    else:
        ## work on all events - shape: n.events x 16200 (micro)features
        events_dict = {}
        for j, event in enumerate(list_events):
            if j in [250,500,750,900,1000,len(list_events)-1]:
                print('Dataframe n',j)
            microfeatures_list = []
            event_timespan = []
            count = 0
            for i in event.index:
                microfeatures_list = microfeatures_list + event.loc[i,:].values.tolist()
                if count == 0 or count == len(event)-1:
                    event_timespan.append(str(i))
                count += 1
            events_dict['-'.join(event_timespan)] = microfeatures_list
        df_events_microfeatures = pd.DataFrame(events_dict)
    return df_events_microfeatures.T


#%%
def shelve_distance(data,load_shelve,nclusters):
    shelve_filename = 'distances_kmeans_'
    shelve_filename += '{0}_'.format(nclusters)
    shelve_message = ''
    shelve_message += '{0} '.format(shelve_filename)
    if load_shelve:
        if os.path.isfile('./{0}.dat'.format(shelve_filename)):
            df_temp = pd.DataFrame()
            shelve_file = shelve.open(shelve_filename)
            df_temp['timestamp'] = shelve_file['timestamp']
            df_temp['distance'] = shelve_file['distance']
            shelve_file.close()
            shelve_message += 'has been LOADED from the shelve file.'
            return df_temp
        else:
            shelve_message += 'has NOT been found.'
    else:
        shelve_file = shelve.open(shelve_filename)
        shelve_file['timestamp'] = data['timestamp']
        shelve_file['distance'] = data['distance']
        shelve_file.close()
        shelve_message += 'has been SAVED in the shelve file.'
        print(shelve_message)
        return True

def kmeans_model(data,elbow,ncluster):
    if elbow == True:
        n_cluster = range(1, 20)
        distorsions = []
        for i in n_cluster:
            print('\nCluster',i)
            km = KMeans(n_clusters=i).fit(data)
            distorsions.append(km.inertia_)
            print('\nCluster',i,' finito')
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(n_cluster, distorsions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plt.show() 
        return None
    
    else:
        print('\nCluster number:',ncluster)
        km = KMeans(n_clusters=ncluster).fit(data)
        km.fit(data)
        km.predict(data)
        distorsion = km.inertia_
        print('\nCluster',ncluster,' finito - distorsion:',distorsion)
        print(km.labels_)

        df_temp = data.copy()
        df_temp = df_temp.reset_index()
        for i in df_temp.index:
            df_temp.loc[i,'cluster'] = km.labels_[i]
        df_temp.loc[:,['index','cluster']].to_csv('kmeans_clusters_event.csv',sep=';')

        return km

def kmeans_detection(data,model,outliers_fraction,compute_distances):
    df1 = df.copy()
    timestamp = list(df1.index)
    timestamp_series = pd.Series(timestamp)
    df1 = df1.reset_index(drop=True)

    if compute_distances == True:
        distance = pd.Series()
        for i in range(0,len(df1)):
            if i > 1000:
                break
            print('sono al index:',i)
            Xa = np.array(df1.loc[i])
            Xb = model.cluster_centers_[model.labels_[i]-1]
            distance.set_value(i, np.linalg.norm(Xa-Xb))

        frame = { 'timestamp': timestamp_series, 'distance': distance } 
        df_distance = pd.DataFrame(frame)
        shelve_distance(df_distance,False,13)
    else:
        df_distance = shelve_distance(None,True,13)
        distance = pd.Series(df_distance['distance'].values)
        
    #outliers_fraction = 0.01
    # get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
    number_of_outliers = int(outliers_fraction*len(distance))
    threshold = distance.nlargest(number_of_outliers).min()
    # anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
    df1['anomaly'] = (distance >= threshold).astype(int)
    
    fig, ax = plt.subplots(figsize=(10,6))
    a = df1.loc[df1['anomaly'] == 1, ['Processor_Percent_Processor_Time__Total']] #anomaly
    ax.plot(df1.index, df1['Processor_Percent_Processor_Time__Total'], color='blue', label = 'Normal')
    ax.scatter(a.index,a['Processor_Percent_Processor_Time__Total'], color='red', label = 'Anomaly')
    plt.legend()
    plt.show();   

    df1 = df1.join(df.reset_index(),rsuffix='_duplicate') 
    df1 = df1.set_index('index') 
    df1.to_csv('kmeans_detection.csv',sep=';')


#%%
##isolation forest model
def forest_model(data,outliers_fraction):
    # train isolation forest
    #forest =  IsolationForest(contamination=outliers_fraction)
    #forest.fit(data)
    #forest = IsolationForest(n_estimators=100, max_samples='auto', contamination=outliers_fraction, \
           # max_features=18, bootstrap=False, n_jobs=-1, random_state=None,behaviour='new', verbose=1)
    forest =  IsolationForest(contamination=outliers_fraction,behaviour='new',n_jobs=-1,verbose=1,max_features=18)
    forest.fit(data)
    print('\nFinished the forest model fitting')
    return forest
    
#%%
##isolation forest
def forest_detection(data,model,outliers_fraction):
    print('\nStarting with anomaly detection via forest model')

    df_temp = data.copy()
    df_temp = df_temp.reset_index()
    df_temp['anomaly'] = pd.Series(model.predict(data))
    # df['anomaly2'] = df['anomaly2'].map( {1: 0, -1: 1} )
    print('\nafter prediction')
    """ fig, ax = plt.subplots(figsize=(10,6))
    a = df_temp.loc[df_temp['anomaly'] == -1, ['Processor_Percent_Processor_Time__Total']] #anomaly
    ax.plot(df_temp.index, df_temp['Processor_Percent_Processor_Time__Total'], color='blue', label = 'Normal')
    ax.scatter(a.index,a[' '], color='red', label = 'Anomaly')
    plt.legend()
    plt.savefig('forest_anomalydetected_ts.png')
    plt.show() """
    print('\nWriting csv..')
    df_temp.loc[:,['index','anomaly']].to_csv('forest_detection_event.csv',sep=';')
    

    print('\nStarting with PCA')
    pca = PCA(n_components=2)  # Reduce to k=3 dimensions
    X_reduce = pca.fit_transform(df_temp.set_index('index'))
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], s=4, lw=1, label="inliers",c="green")
    # Plot x's for the ground truth outliers
    ind = df_temp.loc[df_temp['anomaly'] == -1].index.tolist()
    ax.scatter(X_reduce[ind,0],X_reduce[ind,1], lw=2, s=60, marker="x", c="red", label="outliers")
    plt.legend()
    plt.savefig('forest_anomalydetected_pca_event.png')
    plt.show() 
    


    #print('Anomaly found: ',len(a.index))
    ''' pca = PCA(n_components=3)  # Reduce to k=3 dimensions
    X_reduce = pca.fit_transform(df_temp.set_index('index'))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel("x_composite_3")
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
    # Plot x's for the ground truth outliers
    ax.scatter(X_reduce[a.index,0],X_reduce[a.index,1], X_reduce[a.index,2],
            lw=2, s=60, marker="x", c="red", label="outliers")
    ax.legend()
    plt.show() '''

#%%
def svm_model(data,outliers_fraction):
    # train oneclassSVM 
    model_svm = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.008)
    model_svm.fit(data)
    print('\nSVM OneClass model fitting terminated')
    return model_svm
#%%
def svm_detection(data,model,outliers_fraction):
    # train oneclassSVM 
    print('\nStarting with anomaly detection via svm one class model')
    df_temp = data.copy()
    df_temp = df_temp.reset_index()
    df_temp['anomaly'] = pd.Series(model.predict(data))

    fig, ax = plt.subplots(figsize=(10,6))
    """ print(df_temp.columns)
    a = df_temp.loc[df_temp['anomaly'] == -1, ['Processor_Percent_Processor_Time__Total']] #anomaly

    ax.plot(df_temp.index, df_temp['Processor_Percent_Processor_Time__Total'], color='blue', label = 'Normal')
    ax.scatter(a.index,a['Processor_Percent_Processor_Time__Total'], color='red', label = 'Anomaly')
    plt.xticks(df_temp['index'],rotation=45)
    plt.legend()
    plt.savefig('svm_anomalydetected_ts_g0005_event.png')
    plt.show() """
    print('\nWriting csv..')
    df_temp.loc[:,['index','anomaly']].to_csv('svm_detection_0001_g0005_event.csv',sep=';')
    
    print('\nStarting with PCA')
  
    pca = PCA(n_components=2)  # Reduce to k=3 dimensions
    X_reduce = pca.fit_transform(df_temp.set_index('index'))
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], s=4, lw=1, label="inliers",c="green")
    # Plot x's for the ground truth outliers
    ind = df_temp.loc[df_temp['anomaly'] == -1].index.tolist()
    ax.scatter(X_reduce[ind,0],X_reduce[ind,1], lw=2, s=60, marker="x", c="red", label="outliers")
    plt.legend()
    plt.savefig('svm_anomalydetected_pca2_g0005.png')
    plt.show() 
    ''' pca = PCA(n_components=3)  # Reduce to k=3 dimensions
    X_reduce = pca.fit_transform(df_temp.set_index('index'))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = fig.gca(projection='3d')
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1],X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
    ind = df_temp.loc[df_temp['anomaly'] == -1].index.tolist()
    ax.scatter(X_reduce[ind,0],X_reduce[ind,1],X_reduce[ind,2], lw=2, s=60, marker="x", c="red", label="outliers")
    plt.legend()
    plt.savefig('svm_anomalydetected_pca3.png')
    plt.show() '''

    #print('Anomaly found: ',len(a.index))


 
#%%
def dbscan_model(data):
    print('DBSCAN Clustering is running\n')
    model = DBSCAN(eps=1.5, min_samples = 3)
    clusters = model.fit_predict(data)
    print('Clustering is terminated\n')

    print(clusters)

    print('Writing csv..')
    df_temp = data.copy()
    df_temp = df_temp.reset_index()
    for i in df_temp.index:
        df_temp.loc[i,'Cluster'] = clusters[i]
    print('\nWriting csv..')
    df_temp.loc[:,['index','Cluster']].to_csv('dbscan_detection_labelOnly.csv',sep=';')
    print('Writing csv DONE..')

    print('PCA is running..\n')
    pca = PCA(n_components=2)  # Reduce to k=3 dimensions
    X_reduce = pca.fit_transform(data)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], c=clusters,cmap="plasma")
    # Plot x's for the ground truth outliers
    #ind = df_temp.loc[df_temp['anomaly'] == -1].index.tolist()
    #ax.scatter(X_reduce[ind,0],X_reduce[ind,1], lw=2, s=60, marker="x", c="red", label="outliers")
    plt.legend()
    plt.savefig('dbscan_anomalydetected_pca2.png')
    plt.show() 
    print('PCA is terminated..\n')


#%%
def main():
    outliers_fraction = 0.001

    # Import data from influxdb - calling the private_data_manager
    #df is the joint dataframes of all the measurements (or units) - Event not handled
    #df = import_data()

    list_events = import_data_events()
    df_events_microfeatures = get_event_microfeatures(list_events,True)

    ## K-MEANS ##
    #km = kmeans_model(data=df_events_microfeatures,elbow=False,ncluster=10)
    #kmeans_detection(data=df_events_microfeatures,model=km,outliers_fraction=outliers_fraction,compute_distances = False)

    ## ISOLATION FOREST ##
    #forest = forest_model(df_events_microfeatures,outliers_fraction)
    #forest_detection(data=df_events_microfeatures,model=forest,outliers_fraction=outliers_fraction)


    ## SVM ONE CLASS ##
    #model_svm = svm_model(data=df_events_microfeatures,outliers_fraction=outliers_fraction)
    #svm_detection(data=df_events_microfeatures,model=model_svm,outliers_fraction=outliers_fraction)
    
    ## DBSCAN ##
    #dbscan_model(df_events_microfeatures)
    

if __name__ == '__main__':
    main()