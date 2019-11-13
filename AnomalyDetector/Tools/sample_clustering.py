import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class Modeler():

    def __init__(self,df_samples,n_interactions):

        self.df_samples = df_samples
        self.n_interactions = n_interactions

    def model_kmeans(self, elbow=False, ncluster=10):
        
        if elbow == True:
            n_cluster = range(1, 20)
            distorsions = []
            for i in n_cluster:
                print('\nCluster',i)
                km = KMeans(n_clusters=i).fit(self.df_samples)
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
            km = KMeans(n_clusters=ncluster,max_iter=500).fit(self.df_samples)
            km.fit(self.df_samples)
            km.predict(self.df_samples)
            distorsion = km.inertia_
            print('\nCluster',ncluster,' finito - distorsion:',distorsion)
            print(km.labels_)

            df_temp = self.df_samples.copy()
            df_temp = df_temp.reset_index()
            for i in df_temp.index:
                df_temp.loc[i,'cluster'] = km.labels_[i]
            df_temp.loc[:,['index','cluster']].to_csv('kmeans_clusters_event.csv',sep=';')

            return km,df_temp
