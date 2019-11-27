import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Modeler:

    def __init__(self, df_samples, random_state=1):

        self.df_samples = df_samples
        self.random_state = random_state
        # self.n_interactions = n_interactions

    def model_kmeans(self, elbow=False, ncluster=18, print_clusters=False):
        X = self.df_samples
            
        if elbow == True:
            n_cluster = range(1, 20)
            distorsions = []
            for i in n_cluster:
                print('\nCluster', i)
                km = KMeans(n_clusters=i, random_state=self.random_state).fit(X)
                distorsions.append(km.inertia_)
                print('\nCluster', i, ' finito, - inertia: ', km.inertia_)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(n_cluster, distorsions, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum of squared distances')
            plt.title('Elbow Method For Optimal k')
            plt.show() 

            if print_clusters:
                plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='rainbow')
                plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker="*")
                plt.show()
            return None
        
        else:
            print('\nCluster number:', ncluster)
            km = KMeans(n_clusters=ncluster, max_iter=600, random_state=self.random_state).fit(X)
            # km = KMeans(n_clusters=ncluster,max_iter=600, random_state=0).fit(X)
            km.fit(X)
            km.predict(X)
            distorsion = km.inertia_
            print('\nCluster', ncluster, ' finito - distorsion:', distorsion)

            df_temp = X.copy()
            df_temp = df_temp.reset_index()
            for i in df_temp.index:
                df_temp.loc[i, 'cluster'] = km.labels_[i]
            df_temp.loc[:, ['index', 'cluster']].to_csv('kmeans_clusters_event.csv', sep=';')
            if print_clusters:
                plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=km.labels_, cmap='rainbow')
                plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker="*")
                plt.show()
            df_temp = df_temp.set_index('index')
            return km, df_temp
