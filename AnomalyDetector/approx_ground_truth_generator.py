from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd

class ApproxGroudTruthGenerator:
    
    def __init__(self, clustering_model=None, df_samples_cluster=None, n_interactions=''):
        self.clustering_model = clustering_model
        self.df_samples_cluster = df_samples_cluster
        self.n_interactions = n_interactions
        self.event_interactions = []
        self.df_samples_approx_labels = pd.Series()

    """  def take_interactions(self):
        df_sample_tmp = self.df_samples_cluster.iloc[:,0:len(self.df_samples_cluster.columns)-1]
        closest, _ = pairwise_distances_argmin_min(self.clustering_model.cluster_centers_, df_sample_tmp)
        closest_index_list = []
        for c in closest:
            closest_index_list.append(df_sample_tmp.index[c])
        print(closest_index_list) 
        self.event_interactions = closest_index_list """
    
    def take_interactions(self):
        print("--- Exctracting ",str(self.n_interactions), "Interactions (i.e 1 event per each cluster): START")
        closest_index_list = []
        for i,centroid in enumerate(self.clustering_model.cluster_centers_):
            df_sample_tmp = self.df_samples_cluster[self.df_samples_cluster['cluster']==i]
            df_sample_tmp = df_sample_tmp.iloc[:,0:len(df_sample_tmp.columns)-1]
            closest, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), df_sample_tmp)
            for c in closest:
                closest_index_list.append(df_sample_tmp.index[c])
        self.event_interactions = closest_index_list
        print("These are the extracted interactions as required:\n",self.event_interactions)
        print("--- Exctracting ",str(self.n_interactions), "Interactions (i.e 1 event per each cluster): DONE")


    #def propagate_labels(self,series_events_label,df_samples_cluster):
    def propagate_labels(self,series_events_label):
        print("--- Generating approximate labels based on given interactions: START")
        self.df_samples_cluster = self.df_samples_cluster.reset_index()
        for i,closest in enumerate(self.event_interactions):
            closest_label = series_events_label[closest]#pd translate the time zone by itself
            df_cluster = self.df_samples_cluster[self.df_samples_cluster['cluster']==i]
            self.df_samples_cluster.loc[df_cluster.index.to_list(),'label'] = closest_label
        self.df_samples_cluster = self.df_samples_cluster.set_index('index')
        self.df_samples_approx_labels = pd.Series(self.df_samples_cluster['label'],index=self.df_samples_cluster.index)
        print("--- Generating approximate labels based on given interactions: DONE")