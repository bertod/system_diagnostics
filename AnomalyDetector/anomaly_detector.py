import pandas as pd
import numpy as np
import AnomalyDetector.data_manager as data_manager
import AnomalyDetector.ground_truth_generator as gtg
import AnomalyDetector.approx_ground_truth_generator as agtg
import AnomalyDetector.Tools.sample_extractor as sample_extractor
from AnomalyDetector.Tools.sample_clustering import Modeler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


class CustomerHostTrainer:
    
    def __init__(self, customer='', network='', source='', db='', host='', labeler='', labeling_model='',
                 experiment_start='', experiment_end='', time_zone='', n_ago='',
                 event_period='', guardperiod='', json_path=''):

        self.time_zone = time_zone
        self.customer = customer
        self.network = network
        self.source = source
        self.db = db
        self.host = host        
        self.labeler = labeler
        self.labeling_model = labeling_model
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.n_ago = n_ago
        self.event_period = ''
        self.guardperiod = ''
        self.json_path = json_path
        self.groundtruths_dict = {}
        self.df_events_index = None

        self.import_data()

    def import_data(self):
        """import via data_manager the events dataframe"""

        experiment_start = pd.to_datetime('today', format='%Y-%m-%d %H:%M:%S')
        experiment_start = pd.to_datetime(str(experiment_start).split('.')[0])
        if self.n_ago != '' and (not self.experiment_start and not self.experiment_end):
            delta = [s for s in self.n_ago if s.isdigit()]
            delta = ''.join(delta)
            if 'day' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta), 'days')
            elif 'hour' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta), 'hours')
        else:
            experiment_start = self.experiment_start 
            experiment_end = self.experiment_end

        if self.event_period == '':
            self.event_period = '15m'

        print(experiment_start)
        print(experiment_end)
        print('\n---Importing the list of Events Dataframes via data manager: STARTED')

        fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
                self.customer, self.network, self.source, self.db, self.host,
                str(experiment_start), str(experiment_end), self.labeler, self.labeling_model, self.time_zone,
                'diagnostics_map.json', event_minimum_period=self.event_period, local_data=True,
                database_queries=True, preprocess_data=True)

        self.df_events_index = fabbri1905_fabax6pdb.df_events_index.copy()
        self.groundtruths_dict = fabbri1905_fabax6pdb.groundtruths_dict
        print('---Importing data via data manager: DONE')

    def generate_ground_truth(self):
        gtg_instance = gtg.GroundTruthGenerator(self.labeler, self.labeling_model, self.experiment_start,
                                                self.experiment_end, self.event_period, self.guardperiod,
                                                self.time_zone, self.df_events_index, self.groundtruths_dict)
        
        for serie, data in gtg_instance.class_dict.items():
            if not serie == 'default_class_value' and not serie == 'invalid_class_value' and not serie == 'period' \
                    and not data['type'] == 'default':
                print('Periods of class', serie, ' manually labeled by a human labeler and reported in json file')
                print(data['periods'])
                print('\n')

        # generate signals for each class
        gtg_instance.get_class_continuous_signal()
        
        # generate the ground truth signal - gtg_instance.gt_series attribute
        gtg_instance.get_ground_truth_continuous_signal()

        # perform event labeling (v1 - using inclusion criterion) - gtg_instance.series_events_label
        gtg_instance.perform_event_labeling()

        # perform event labeling (v2 - using gt signal event extraction) - gtg_instance.gt_series_events_label
        gtg_instance.perform_event_labeling_from_gt_signal()

        return gtg_instance


class CustomerHostDesigner:
    """
    An object of this class is supposed to be used just in the Design Phase of the whole semi-supervised system.
    It will perform the following task:
    - import data via data_manager.py
    - feature extraction + dimensionality reduction (implemented in sample_extractor.py)
    - unsupervised clustering via sample_clustering (implemented in  sample_clustering.py)
    - generate approximated ground truth (implemented in approx_ground_truth.py)
    - generate (real) ground truth using data from diagnostic_map (implemented in ground_truth_generator.py)
    - assess the quality of the approximation
    """
    def __init__(self, n_interactions=15, n_clusters=18, clustering_algo_name='kmeans',
                 customer='', network='', source='', db='',
                 host='', labeler='', labeling_model='', experiment_start='',
                 experiment_end='', time_zone='', n_ago='', event_period='', guardperiod=''):

        self.n_interactions = n_interactions
        self.n_clusters = n_clusters
        self.clustering_algo_name = clustering_algo_name
        self.time_zone = time_zone
        self.customer = customer
        self.network = network
        self.source = source
        self.db = db
        self.host = host        
        self.labeler = labeler
        self.labeling_model = labeling_model
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.n_ago = n_ago
        self.event_period = event_period
        self.guardperiod = guardperiod

        self.list_events = []
        self.groundtruths_dict = {}
        self.model = None
        self.df_events_index = None
        self.df_samples_cluster = None
        self.df_samples_reduce = None
        self.df_samples = None

        self.import_data()
    
    def import_data(self):
        """import via data_manager the events dataframe"""

        experiment_start = pd.to_datetime('today', format='%Y-%m-%d %H:%M:%S')
        experiment_start = pd.to_datetime(str(experiment_start).split('.')[0])
        if self.n_ago != '' and (not self.experiment_start and not self.experiment_end):
            delta = [s for s in self.n_ago if s.isdigit()]
            delta = ''.join(delta)
            if 'day' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta), 'days')
            elif 'hour' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta), 'hours')
        else:
            experiment_start = self.experiment_start 
            experiment_end = self.experiment_end

        if self.event_period == '':
            self.event_period = '15m'

        print(experiment_start)
        print(experiment_end)
        print('\n---Importing the list of Events Dataframes via data manager: STARTED')

        fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
                self.customer, self.network, self.source, self.db, self.host,
                str(experiment_start), str(experiment_end), self.labeler, self.labeling_model, self.time_zone,
                'diagnostics_map.json', event_minimum_period=self.event_period, local_data=True,
                database_queries=True, preprocess_data=True)

        self.df_events_index = fabbri1905_fabax6pdb.df_events_index.copy()
        self.groundtruths_dict = fabbri1905_fabax6pdb.groundtruths_dict
        self.list_events = fabbri1905_fabax6pdb.measure_pd_dataevent_samples

        print('---Importing data via data manager: DONE')
        
    def get_samples(self, reduction=False, n_components=2):
        extractor = sample_extractor.SampleExtractor(self.list_events)
        extractor.generate_samples(feature_engineering=True)
        self.df_samples = extractor.df_samples.copy()
        if reduction:
            extractor.apply_pca(n_components)
            self.df_samples_reduce = extractor.df_samples_reduce.copy()

    def get_clustering_model(self, elbow=False, print_clusters=False, random_state=1, df_subset_sample=None):
        # if not df_subset_sample.items() == None:
        if not df_subset_sample.empty:
            clusterizer = Modeler(df_subset_sample)
            # clusterizer = Modeler(df_subset_sample,self.n_interactions)
        elif not self.df_samples_reduce.empty:
            clusterizer = Modeler(self.df_samples_reduce, random_state)
        else:
            clusterizer = Modeler(self.df_samples, random_state)

        if self.clustering_algo_name.lower().strip() == 'kmeans':
            if elbow == True:
                clusterizer.model_kmeans(elbow=True, print_clusters=print_clusters)
            else:
                self.model, self.df_samples_cluster = clusterizer.model_kmeans(elbow=False,
                                                                               ncluster=self.n_clusters,
                                                                               print_clusters=print_clusters)

    def instantiate_approx_gtg(self, n_interactions_optional=None):
        """ if not self.df_samples_reduce.empty:
            agtg_instance = agtg.ApproxGroundTruthGenerator(clustering_model=self.model, \
                            df_samples=self.df_samples_reduce, n_interactions=self.n_interactions)
        else:
            agtg_instance = agtg.ApproxGroundTruthGenerator(clustering_model=self.model, \
                            df_samples=self.df_samples, n_interactions=self.n_interactions) """
        if n_interactions_optional:
            n_interactions = n_interactions_optional
        else:
            n_interactions = self.n_interactions
        agtg_instance = agtg.ApproxGroundTruthGenerator(clustering_model=self.model,
                                                        df_samples_cluster=self.df_samples_cluster,
                                                        n_interactions=n_interactions)
        return agtg_instance

    def generate_ground_truth(self):

        experiment_start = pd.to_datetime('today', format='%Y-%m-%d %H:%M:%S')
        experiment_start = pd.to_datetime(str(experiment_start).split('.')[0])
        if self.n_ago != '' and (not self.experiment_start and not self.experiment_end):
            delta = [s for s in self.n_ago if s.isdigit()]
            delta = ''.join(delta)
            if 'day' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta), 'days')
            elif 'hour' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta), 'hours')
        else:
            experiment_start = self.experiment_start 
            experiment_end = self.experiment_end

        if self.event_period == '':
            self.event_period = '15m'
        if not self.df_samples_reduce.empty:
            df_event_index = pd.DatetimeIndex(self.df_samples_reduce.index)
        else:
            df_event_index = pd.DatetimeIndex(self.df_samples.index)

        gtg_instance = gtg.GroundTruthGenerator(self.labeler, self.labeling_model, experiment_start, experiment_end,
                                                self.event_period, self.guardperiod, self.time_zone, df_event_index,
                                                self.groundtruths_dict)
        
        for serie, data in gtg_instance.class_dict.items():
            if not serie == 'default_class_value' and not serie == 'invalid_class_value' and not serie == 'period' \
                    and not data['type'] == 'default':
                print('Periods of class', serie, ' manually labeled by a human labeler and reported in json file')
                print(data['periods'])
                print('\n')

        # generate signals for each class
        gtg_instance.get_class_continuous_signal()
        
        # generate the ground truth signal - gtg_instance.gt_series attribute
        gtg_instance.get_ground_truth_continuous_signal()

        # perform event labeling (v1 - using inclusion criterion) - gtg_instance.series_events_label
        gtg_instance.perform_event_labeling()

        # perform event labeling (v2 - using gt signal event extraction) - gtg_instance.gt_series_events_label
        gtg_instance.perform_event_labeling_from_gt_signal()

        return gtg_instance
    
    def assess_agtg(self, ground_truth_labels=None, approximate_labels=None, confusion_mtx=True):

        if confusion_mtx:
            cm = confusion_matrix(ground_truth_labels, approximate_labels)
            # Plot confusion matrix
            plt.figure(figsize=(7, 7))
            plt.imshow(cm, interpolation='none', cmap='Greys')
            for (i, j), z in np.ndenumerate(cm):
                plt.text(j, i, z, ha='center', va='center')
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            # plt.rcParams.update({'font.size': 32})
            plt.show()

        print('accuracy: ', accuracy_score(ground_truth_labels, approximate_labels))
        accuracy = accuracy_score(ground_truth_labels, approximate_labels)

        print('f1 measure (weighted): ', f1_score(ground_truth_labels, approximate_labels, average='weighted'))
        f1_weighted = f1_score(ground_truth_labels, approximate_labels, average='weighted')

        print('f1 measure (none): ', f1_score(ground_truth_labels, approximate_labels, average=None))
        f1_none = f1_score(ground_truth_labels, approximate_labels, average=None)

        print('f1 measure (binary): ', f1_score(ground_truth_labels, approximate_labels, average='binary'))
        f1_binary = f1_score(ground_truth_labels, approximate_labels, average='binary')

        return accuracy, f1_weighted, f1_binary
