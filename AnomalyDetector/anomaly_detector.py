
import pandas as pd
import numpy as np
import AnomalyDetector.data_manager as data_manager
import AnomalyDetector.ground_truth_generator as gtg
#import AnomalyDetector.Data.feature_extractor as feature_extractor

class CustomerHostTrainer():
    
    def __init__(self,customer='' , network='', source='', db='', host='', labeler='', labeling_model='', experiment_start='', \
        experiment_end='', time_zone='', n_ago='', event_period='', guardperiod='', json_path=''):

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
        self.gtg_instance = None

        self.import_data()
        

    def import_data(self):
        """import via data_manager the events dataframe"""

        experiment_start = pd.to_datetime('today',format='%Y-%m-%d %H:%M:%S')
        experiment_start = pd.to_datetime(str(experiment_start).split('.')[0])
        if self.n_ago != '' and (not self.experiment_start and not self.experiment_end) :
            delta = [s for s in self.n_ago if s.isdigit()]
            delta = ''.join(delta)
            if 'day' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta),'days')
            elif 'hour' in self.n_ago:
                experiment_end = experiment_start - pd.to_timedelta(int(delta),'hours')
        else:
            experiment_start = self.experiment_start 
            experiment_end = self.experiment_end

        if self.event_period == '':
            self.event_period = '15m'
        #else:
            #event_period = self.event_period

        print(experiment_start)
        print(experiment_end)
        print('\n---Importing the list of Events Dataframes via data manager: STARTED')
        fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
                self.customer, self.network, self.source, self.db, self.host,
                str(experiment_start), str(experiment_end), self.labeler, self.labeling_model, 'Europe/Rome',
                'diagnostics_map.json', event_minimum_period=self.event_period, local_data=True,
                database_queries=True, preprocess_data=True)

        self.df_events_index = fabbri1905_fabax6pdb.df_events_index.copy()
        self.groundtruths_dict = fabbri1905_fabax6pdb.groundtruths_dict
        print('---Importing data via data manager: DONE')


    def generate_ground_truth(self):
        #print(self.df_events_index.index)
        gtg_instance = gtg.GroudTruthGenerator(self.labeler, self.labeling_model, self.experiment_start,self.experiment_end, \
                                        self.event_period, self.guardperiod, self.time_zone, self.df_events_index, \
                                        self.groundtruths_dict)
        
        for serie,data in gtg_instance.class_dict.items():
            if not serie == 'default_class_value' and not serie == 'invalid_class_value' and not serie == 'period' and not data['type'] == 'default':
                print('Periods of class',serie,' manually labeled by a human labeler and reported in json file')
                print(data['periods'])
                print('\n')

        #generate signals for each class
        gtg_instance.get_class_continuous_signal()
        
        #generate the ground truth signal - gtg_instance.gt_series attribute
        gtg_instance.get_ground_truth_continuous_signal()

        #perform event labeling (v1 - using inclusion criterion) - gtg_instance.gt_series_events_label
        gtg_instance.perform_event_labeling()

        #perform event labeling (v2 - using gt signal event extraction) - gtg_instance.series_events_label
        gtg_instance.perform_event_labeling_from_gt_signal()

        return gtg_instance

