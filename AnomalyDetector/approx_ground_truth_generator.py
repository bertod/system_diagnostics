from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
from collections import Counter

class ApproxGroudTruthGenerator:
    
    def __init__(self, clustering_model=None, df_samples_cluster=None, n_interactions=''):
        self.clustering_model = clustering_model
        self.df_samples_cluster = df_samples_cluster
        self.n_interactions = n_interactions #n interactions per cluster
        self.event_interactions = {}
        self.df_samples_approx_labels = pd.Series()
   
    def take_interactions_v1(self):
        """
            This function extract interactions (event to be labeled) taking just the first event
            closest to the cluster centroid 
        """
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

    def take_interactions(self):
        print("--- Exctracting ",str(self.n_interactions), "Interactions per Cluster (i.e events per each cluster): START")
        for i,centroid in enumerate(self.clustering_model.cluster_centers_):
            closest_index_list = []
            df_sample_tmp = self.df_samples_cluster[self.df_samples_cluster['cluster']==i]
            df_sample_tmp = df_sample_tmp.iloc[:,0:len(df_sample_tmp.columns)-1]
            closest, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), df_sample_tmp)
            #print("closest1 ! : ",closest)

            n = self.n_interactions
            if len(df_sample_tmp.index) < n:
                n = len(df_sample_tmp.index)
            while len(closest) < n:
                closest_, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), df_sample_tmp.drop(df_sample_tmp.index[closest]))
                closest = np.append(closest,closest_)
                #print("closest",len(closest)," : ",closest_)

            for c in closest:
                closest_index_list.append(df_sample_tmp.index[c])
            #print("two closest!! : ",closest)
            self.event_interactions[i] = closest_index_list
            
        #event_interactions = closest_index_list
        #print("These are the extracted interactions as required:\n",event_interactions)
        print("--- Exctracting ",str(self.n_interactions), "Interactions per Cluster (i.e events per each cluster): DONE")

    #def propagate_labels(self,series_events_label,df_samples_cluster):
    def propagate_labels_v1(self,series_events_label):
        """
            This function generate approximate labels, using just the first event closest 
            to cluster centroid 
        """
        print("--- Generating approximate labels based on given interactions: START")
        self.df_samples_cluster = self.df_samples_cluster.reset_index()
        for i,closest in enumerate(self.event_interactions):
            closest_label = series_events_label[closest]#pd translate the time zone by itself
            df_cluster = self.df_samples_cluster[self.df_samples_cluster['cluster']==i]
            self.df_samples_cluster.loc[df_cluster.index.to_list(),'label'] = closest_label
        self.df_samples_cluster = self.df_samples_cluster.set_index('index')
        self.df_samples_approx_labels = pd.Series(self.df_samples_cluster['label'],index=self.df_samples_cluster.index)
        print("--- Generating approximate labels based on given interactions: DONE")


    def propagate_labels(self,series_events_label):
        """
            This function generate approximate labels, which generalize the previous one
            for using more than the first closest event
        """
        print("--- Generating approximate labels based on given interactions: START")
        self.df_samples_cluster = self.df_samples_cluster.reset_index()
        for i,closest_list in self.event_interactions.items():
            closest_label_list = [series_events_label[closest] for closest in closest_list]#pd translate the time zone by itself
            closest_label = self.find_majority(closest_label_list)
            df_cluster = self.df_samples_cluster[self.df_samples_cluster['cluster']==i]
            self.df_samples_cluster.loc[df_cluster.index.to_list(),'label'] = closest_label
        self.df_samples_cluster = self.df_samples_cluster.set_index('index')
        self.df_samples_approx_labels = pd.Series(self.df_samples_cluster['label'],index=self.df_samples_cluster.index)
        print("--- Generating approximate labels based on given interactions: DONE")

    def find_majority(self,votes):
        """
            helper function for the propagate_labels method. 
            It looks for the most common value in an event period
        """
        vote_count = Counter(votes)
        top = vote_count.most_common(1)
        if len(top)>1:
            # It is a tie and it maight be ambiguous
            return -1
        return top[0][0]