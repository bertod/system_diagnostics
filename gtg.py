##Ground Truth Generator

#%%
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import shelve
import argparse
import json

import data_manager
import data_viewer
import data_labeler
import signal_processor

#%%
## class definition
class GroudTruthGenerator:

    def __init__(self,user = '' ,session = '', observation_start = '', observation_end ='' , json_path = ''):
        self.user = user
        self.session = session
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.json_path = json_path
        self.json_obj = ''
        self.load_json()
        self.anomaly_periods = []
        self.anomaly_periods_resample = []
        self.gt_series = pd.Series()
        self.gt_series_resample = pd.Series()
        print("\nGTG instantiated\n")

    def load_json(self):
        try:
            json_file = open(self.json_path)
        except IOError:
            print('error in json file opening - check if it acutally exists')
            return None
        try:
            json_object = json.load(json_file)
        except ValueError:
            print('error in json file reading')
            return None
        self.json_obj = json_object
        #return json_object
    
    ########################################################################
    #load period of anomaly (Actual GroundTruth), recorded in the json file#
    ########################################################################
    def get_ground_truth_periods(self):
        gt_map = self.json_obj
        #print(gt_map['users'])
        print('\n---Loading periods manually labeled as Anomaly from the selected Json: STARTED')
        for user in gt_map['users']:#filter the user
            if user['user_name'].lower().strip() == self.user.lower().strip() or \
                 user['user_nickname'].lower().strip() == self.user.lower().strip(): 

                 for session in user['user_sessions']: #filter the session
                     if session['session_name'].lower().strip() == self.session.lower().strip():
                        for period in session['analyzed_periods']: #filter period of observation
                            if period['observation_start'] == self.observation_start and \
                                period['observation_end'] == self.observation_end:
                                flag = False #it is True when the class_name 'anomaly' is found in the json
                                for class_ in period['classes']: #filter class - e.g anomaly or normal
                                    if class_['class_name'] == 'anomaly':
                                        flag = True
                                        records = class_['class_records']
                                        for ts,te in records: #get start and end timestamp of an anomaly period
                                            self.anomaly_periods.append((ts,te)) #list of tuples
                                if flag == False:
                                    print('---Loading periods manually labeled as Anomaly from the selected Json: FAILED\n \
                                        the class_name "anomaly" does not exist in the selected json file')
                                else:
                                    if len(self.anomaly_periods) > 0:
                                        print('---Loading periods manually labeled as Anomaly from the selected Json: DONE')
                                    else:
                                        print('---Loading periods manually labeled as Anomaly from the selected Json: FAILED\n \
                                            The class_name "anomaly" is empty (no anomalous period time recorded) in the selected Json')
                                    


    ########################################################################
    #generate a continuous signal from the anomaly periods recorded in Json#
    ########################################################################
    def get_ground_truth_continuous_signal(self):
        print('Generaton of the continuous signal: STARTED')
        series_list = []
        #time_zone = 'Europe/Rome'
        time_zone = 'GMT' #corresponds to UTC
        prev = ''
        #next = ''
        obs_start = pd.to_datetime(self.observation_start)
        obs_end = pd.to_datetime(self.observation_end)
        for start,end in self.anomaly_periods:
            if prev == '':
                timezone_index = pd.date_range(str(obs_start.tz_localize(time_zone)),start, freq='1T')
                utc_index = pd.to_datetime(timezone_index, utc=True)
                series_list.append(pd.Series([0] * len(utc_index), index=utc_index))
            else: 
                timezone_index = pd.date_range(prev,start, freq='1T')
                utc_index = pd.to_datetime(timezone_index, utc=True)
                series_list.append(pd.Series([0] * len(utc_index), index=utc_index))

            timezone_index = pd.date_range(start,end, freq='1T')
            utc_index = pd.to_datetime(timezone_index, utc=True)
            series_list.append(pd.Series([1] * len(utc_index), index=utc_index))
            prev = end
        if not prev == '':
            timezone_index = pd.date_range(prev,str(obs_end.tz_localize(time_zone)), freq='1T')
            utc_index = pd.to_datetime(timezone_index, utc=True)
            series_list.append(pd.Series([0] * len(utc_index), index=utc_index))

        #print(series_list)
        #gt_series = pd.Series()
        for s in series_list:
            self.gt_series = self.gt_series.append(s)
        #print(gt_series)
        print('Generaton of the continuous signal: DONE')
    ########################################################################
    #resample the continuous signal with the resampling period used on the #
    #original series                                                       #                                      
    ########################################################################
    def resample_ground_truth_continuous_signal(self,resampling_period):
        self.gt_series_resample = self.gt_series.resample(resampling_period).pad()

    def get_anomaly_periods_after_resample(self):
        print('Extracting the anomaly periods after resampling: STARTED')
        prev = None
        start = None
        end = None
        for i in self.gt_series_resample.index:
            if self.gt_series_resample[i] == 0 :
                if prev == None:
                    #print('Sto qua 4')
                    pass
                elif self.gt_series_resample[prev] == 1:
                    #print('Sto qua 0')
                    end = prev
                    print(start,end)
                    self.anomaly_periods_resample.append((start,end))
                    start = None
                    end = None
            else:
                #print(gt_series_resample[i])
                if prev == None:
                    #print('Sto qua 3')
                    start = i
                elif self.gt_series_resample[prev] == 1:
                    #print('Sto qua 2')
                    pass
                else:
                    #print('Sto qua 1')
                    start = i
            prev = i
        #print(len(gt_series_resample[gt_series_resample==1]))
        #print('start',start)
        #print('end',end)
        #print(gt_series_resample[len(gt_series_resample.index)-1])
        if start and end == None:
            if self.gt_series_resample[len(self.gt_series_resample.index)-1] == 1:
                end = self.gt_series_resample.index[len(self.gt_series_resample.index)]
                self.anomaly_periods_resample.append((start,end))
        print('Extracting the anomaly periods after resampling: DONE')
        


#%%
###### helper functions ######

########################################################################
#import via data_manager the events dataframe                          #
########################################################################
def import_data_events(observation_start,observation_end):
    print('\n---Importing the list of Events Dataframes via data manager: STARTED')
    fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
             observation_start, observation_end, 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)
    list_event = fabbri1905_fabax6pdb.measure_pd_dataevent_samples
    global _resampling_period_string
    _resampling_period_string = fabbri1905_fabax6pdb._resampling_period_string
    #print(list_event)
    print('---Importing the list of Events Dataframes via data manager: DONE')
    return list_event

def get_resampling_period_string(): 
    return _resampling_period_string

########################################################################
#combine all the events and generate a dataframe with microfeatures    #
#it can also apply feature engineering (mean and std method)           #
########################################################################
def get_event_microfeatures(list_events,feature_engineering):
    if feature_engineering == True:
        ## work on all events - shape: n.events x 36features (mean + variance)
        print('\n---Bulding the Event Dataframe with microfeatures (mean and std method): STARTED')
        events_dict = {}
        for j, event in enumerate(list_events):
            if j in [250,500,750,900,1000,len(list_events)-1]:
                print('Event modelling: event dataframe n',j)
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
        print('---Bulding the Event Dataframe with microfeatures: DONE')
        return df_events_microfeatures
    else:
        ## work on all events - shape: n.events x 16200 (micro)features
        print('\n---Bulding the Event Dataframe with microfeatures (without feature engineering): STARTED')
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
        print('\n---Bulding the Event Dataframe with microfeatures (without feature engineering): DONE')
    return df_events_microfeatures.T

#%%
def main():
       
    json_path = "classes.json" # default value
    parser = argparse.ArgumentParser(description='Welcome to Ground Truth Generator by WurthPhoenix')
    parser.add_argument('-u','--user', help='Human Labeler name or nickname',required=True)
    parser.add_argument('-s','--session',help='Session name, e.g. anomaly_normal_20191016', required=True)
    parser.add_argument('-b','--begin', help='Observation period start timestamp, e.g 2019-02-04 00:00:01',required=True)
    parser.add_argument('-e','--end', help='Observation period end timestamp',required=True)
    parser.add_argument('-j','--path', help='json file path',required=False)
    args = parser.parse_args()
    
    print("\n---Params you typed")
    print ("Human Labeler name (or nickname): %s" % args.user )
    print ("Session name: %s" % args.session )
    print ("Observation period start: %s" % args.begin )
    print ("Observation period end: %s" % args.end )
    if args.path:
        json_path = args.path
    print ("json file path: %s" % json_path  )


    ### Import data from influxdb - calling the private_data_manager
    #df is the joint dataframes of all the measurements (or units) - Event not handled
    #df = import_data()

    #decomment for importing data as Events
    list_events = import_data_events(args.begin,args.end)
    df_events_microfeatures = get_event_microfeatures(list_events,True)

    gtg_obj = GroudTruthGenerator(args.user, args.session, args.begin, args.end, json_path)
    gtg_obj.get_ground_truth_periods()
    print("Anomalous time periods labeled by the human ",args.user)
    print(gtg_obj.anomaly_periods)
    gtg_obj.get_ground_truth_continuous_signal()
    #print('\n GT Signal')
    #print(gtg_obj.gt_series)
    #print(gtg_obj.gt_series[gtg_obj.gt_series==1])
    resampling_period = get_resampling_period_string()
    gtg_obj.resample_ground_truth_continuous_signal(resampling_period)
    print("Anomalous time periods labeled by the human ",args.user, " (after resampling)")
    gtg_obj.get_anomaly_periods_after_resample()
    #print(gtg_obj.anomaly_periods_resample)


if __name__ == '__main__':
    main()


