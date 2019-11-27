from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
from collections import Counter


class ApproxGroundTruthGenerator:
    """
    This class provides methods for building an approximation
    of the ground truth in the following manner:
    - take clustering model applied at the previous stage (i.e. clustering part)
    - extract a number of interactions (samples) from each cluster
    - spread  the label for those interactions to the whole cluster where it belongs
    Note. The label for the interactions could be set into diagnostic map (i.e. design phase)
          or taken as input (i.e. training phase)
    """
    def __init__(self, clustering_model=None, df_samples_cluster=None, n_interactions=''):
        self.clustering_model = clustering_model
        self.df_samples_cluster = df_samples_cluster
        self.n_interactions = n_interactions
        self.event_interactions = {}
        self.df_samples_approx_labels = pd.Series()
   
    def take_interactions_v1(self):
        """
            This function extract interactions (event to be labeled) taking just the first event
            closest to the cluster centroid 
        """
        print("--- Exctracting ", str(self.n_interactions),
              "Interactions (i.e 1 event per each cluster): START")
        closest_index_list = []
        for i, centroid in enumerate(self.clustering_model.cluster_centers_):
            df_sample_tmp = self.df_samples_cluster[self.df_samples_cluster['cluster'] == i]
            df_sample_tmp = df_sample_tmp.iloc[:, 0:len(df_sample_tmp.columns)-1]
            closest, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), df_sample_tmp)
            for c in closest:
                closest_index_list.append(df_sample_tmp.index[c])
        self.event_interactions = closest_index_list
        print("These are the extracted interactions as required:\n", self.event_interactions)
        print("--- Exctracting ", str(self.n_interactions),
              "Interactions (i.e 1 event per each cluster): DONE")

    def take_interactions(self):

        print("--- Exctracting ", str(self.n_interactions),
              "Interactions per Cluster (i.e events per each cluster): START")
        for i, centroid in enumerate(self.clustering_model.cluster_centers_):
            # print("\ncluster n. ",i)
            closest_index_list = []
            df_sample_tmp = self.df_samples_cluster[self.df_samples_cluster['cluster'] == i]
            df_sample_tmp = df_sample_tmp.iloc[:, 0:len(df_sample_tmp.columns)-1]
            closest, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), df_sample_tmp)
            closest_index_list.append(df_sample_tmp.index[closest[0]])
            # print("closest1 ! : ",closest)
            # print("df di closest111 ",df_sample_tmp.index[closest])

            n = self.n_interactions
            if len(df_sample_tmp.index) < n:
                n = len(df_sample_tmp.index)

            # print("len of not dropped ",len(df_sample_tmp.index))
            df_sample_tmp = df_sample_tmp.drop(df_sample_tmp.index[closest[0]])
            
            while len(closest) < n:
                closest_ = None                
                closest_, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), df_sample_tmp)
                closest_index_list.append(df_sample_tmp.index[closest_[0]])
                df_sample_tmp = df_sample_tmp.drop(df_sample_tmp.index[closest_[0]])
                closest = np.append(closest, closest_)
            self.event_interactions[i] = closest_index_list
            
        print("--- Exctracting ", str(self.n_interactions),
              "Interactions per Cluster (i.e events per each cluster): DONE")

    def propagate_labels_v1(self, series_events_label):
        """
            This function generate approximate labels, using just the first event closest 
            to cluster centroid 
        """
        print("--- Generating approximate labels based on given interactions: START")
        self.df_samples_cluster = self.df_samples_cluster.reset_index()
        for i, closest in enumerate(self.event_interactions):
            closest_label = series_events_label[closest]
            df_cluster = self.df_samples_cluster[self.df_samples_cluster['cluster'] == i]
            self.df_samples_cluster.loc[df_cluster.index.to_list(), 'label'] = closest_label
        self.df_samples_cluster = self.df_samples_cluster.set_index('index')
        self.df_samples_approx_labels = pd.Series(self.df_samples_cluster['label'], index=self.df_samples_cluster.index)
        print("--- Generating approximate labels based on given interactions: DONE")

    def propagate_labels(self, series_events_label):
        """
            This function generate approximate labels, which generalize the previous one
            for using more than the first closest event
        """
        print("--- Generating approximate labels based on given interactions: START")
        self.df_samples_cluster = self.df_samples_cluster.reset_index()
        for i, closest_list in self.event_interactions.items():
            closest_label_list = [series_events_label[closest] for closest in closest_list]
            closest_label = self.find_majority(closest_label_list)
            # in case of tie
            if closest_label == -1:
                closest_label = closest_label_list[0]  # label from the first closest point
            df_cluster = self.df_samples_cluster[self.df_samples_cluster['cluster'] == i]
            self.df_samples_cluster.loc[df_cluster.index.to_list(), 'label'] = closest_label
        self.df_samples_cluster = self.df_samples_cluster.set_index('index')
        self.df_samples_approx_labels = pd.Series(self.df_samples_cluster['label'], index=self.df_samples_cluster.index)
        print("--- Generating approximate labels based on given interactions: DONE")

    def find_majority(self, votes):
        """
            helper function for the propagate_labels method. 
            It looks for the most common value in an event period
        """
        vote_count = Counter(votes)
        top = vote_count.most_common(1)
        if len(top) > 1:
            # It is a tie and it maight be ambiguous
            return -1
        return top[0][0]
