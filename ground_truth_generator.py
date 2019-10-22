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

    def __init__(self,labeler = '' ,model = '', observation_start = '', observation_end ='' , json_path = '',groundtruths_json = [] ):
        self.labeler = labeler
        self.model = model
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.json_path = json_path
        self.json_obj = ''
        self.load_json()
        self.anomaly_periods = []
        self.anomaly_periods_resample = []
        self.gt_series = pd.Series()
        self.gt_series_resample = pd.Series()
        self.groundtruths_json = groundtruths_json
        self.invalid_class_value = ''
        self.periods_list = []
        self.class_dict = {}
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
        print('\n---Loading periods manually labeled as Anomaly from the selected Json: STARTED')
        for truth in self.groundtruths_json:#filter the user
            if truth['labeler_alias'].lower().strip() == self.labeler.lower().strip(): 
                for model in truth['labeling_models']: #filter the session
                    if model['model_name'].lower().strip() == self.model.lower().strip():
                        self.invalid_class_value = model['invalid_class_value']
                        
                        for period in model['periods']: #filter period of observation
                            self.periods_list.append((period['period_start'],period['period_end']))
                
                        flag = False #it is True when the class_name 'anomaly' is found in the json
                        for class_ in model['classes']: #filter class - e.g anomaly or normal
                            self.class_dict[class_['class_label']] = class_['class_value'].lower().strip()
                            if class_['class_label'] == 'anomaly':
                                flag = True
                                for class_period in class_['class_periods']:
                                    self.anomaly_periods.append((class_period['period_start'],class_period['period_end']))

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
        print('\n---Generation of the continuous signal: STARTED')
        series_list = []
        time_zone = 'Europe/Rome'
        #time_zone = 'GMT' #corresponds to UTC
        prev = ''
        obs_start = pd.to_datetime(self.observation_start).tz_localize(time_zone)
        obs_end = pd.to_datetime(self.observation_end).tz_localize(time_zone)

        for ps,pe in self.periods_list: #labeling period start, period end
            ps = str(pd.to_datetime(ps).tz_convert(time_zone))
            pe = str(pd.to_datetime(pe).tz_convert(time_zone))
            if prev == '':
                timezone_index = pd.date_range(str(obs_start),ps, freq='1T',closed = 'left')
                utc_index = pd.to_datetime(timezone_index, utc=True)
                series_list.append(pd.Series([int(self.invalid_class_value)] * len(utc_index), index=utc_index))
            else: 
                timezone_index = pd.date_range(prev,ps, freq='1T',closed = 'left')
                utc_index = pd.to_datetime(timezone_index, utc=True)
                series_list.append(pd.Series([int(self.invalid_class_value)] * (len(utc_index)-1), index=utc_index[1:]))
            prev = pe
            
            prev_anomaly = ''
            flag = False
            for anomaly_start,anomaly_end in self.anomaly_periods: #anomaly start, anomaly end
                anomaly_start = str(pd.to_datetime(anomaly_start).tz_convert(time_zone))
                anomaly_end = str(pd.to_datetime(anomaly_end).tz_convert(time_zone))
                if anomaly_start >= ps and anomaly_end <= pe: #this anomaly is in the current period
                    flag = True
                    if prev_anomaly == '':
                        timezone_index = pd.date_range(ps,anomaly_start, freq='1T',closed = 'left')
                        utc_index = pd.to_datetime(timezone_index, utc=True)
                        series_list.append(pd.Series([self.class_dict['normal']] * len(utc_index), index=utc_index))
                    else:
                        timezone_index = pd.date_range(prev_anomaly,anomaly_start, freq='1T',closed = 'left')
                        utc_index = pd.to_datetime(timezone_index, utc=True)
                        series_list.append(pd.Series([self.class_dict['normal']] * (len(utc_index)-1), index=utc_index[1:]))#skip the end timestamp of the previous anomaly
                    prev_anomaly = anomaly_end
                   
                    timezone_index = pd.date_range(anomaly_start,anomaly_end, freq='1T')
                    utc_index = pd.to_datetime(timezone_index, utc=True)
                    series_list.append(pd.Series([1] * len(utc_index), index=utc_index))
            if flag == False: # normale period - period without anomalies
                timezone_index = pd.date_range(ps,pe, freq='1T')
                utc_index = pd.to_datetime(timezone_index, utc=True)
                series_list.append(pd.Series([self.class_dict['normal']] * len(utc_index), index=utc_index))
            elif not prev_anomaly == '':
                timezone_index = pd.date_range(prev_anomaly,pe, freq='1T',closed = 'right')
                utc_index = pd.to_datetime(timezone_index, utc=True)
                series_list.append(pd.Series([0] * len(utc_index), index=utc_index))

        if not prev == '':
            timezone_index = pd.date_range(prev,str(obs_end), freq='1T', closed = 'right')
            utc_index = pd.to_datetime(timezone_index, utc=True)
            series_list.append(pd.Series([int(self.invalid_class_value)] * len(utc_index), index=utc_index))


        for s in series_list:
            self.gt_series = self.gt_series.append(s)
        #print(gt_series)
        print('---Generaton of the continuous signal: DONE')


    ########################################################################
    #resample the continuous signal with the resampling period used on the #
    #original series                                                       #                                      
    ########################################################################
    def resample_ground_truth_continuous_signal(self,resampling_period):
        print('\n---Resampling the GroundTruth continuous signal: STARTED')
        self.gt_series_resample = self.gt_series.resample(resampling_period).pad()
        print('---Resampling the GroundTruth continuous signal: DONE')

    def get_anomaly_periods_after_resample(self):
        print('---Extracting the anomaly periods after resampling: STARTED')
        prev = None
        start = None
        end = None
        for i in self.gt_series_resample.index:
            if self.gt_series_resample[i] == 0 :
                if prev == None:
                    #print('Here 4')
                    pass
                elif self.gt_series_resample[prev] == 1:
                    #print('Here 0')
                    end = prev
                    print(start,end)
                    self.anomaly_periods_resample.append((start,end))
                    start = None
                    end = None
            else:
                #print(gt_series_resample[i])
                if prev == None:
                    #print('Here 3')
                    start = i
                elif self.gt_series_resample[prev] == 1:
                    #print('Here 2')
                    pass
                else:
                    #print('Here 1')
                    start = i
            prev = i
        if start and end == None:
            if self.gt_series_resample[len(self.gt_series_resample.index)-1] == 1:
                end = self.gt_series_resample.index[len(self.gt_series_resample.index)]
                self.anomaly_periods_resample.append((start,end))
        print('---Extracting the anomaly periods after resampling: DONE')
        
    
    def check_eventi_anomaly_inclusion(self,df_events):
        print("\n---Checking for anomalous period fully/partially included in our events: STARTED")
        #anomalies = gtg_obj.anomaly_periods_resample
        for ind in df_events.index:
            period = ind.split(" - ")
            for a in self.anomaly_periods_resample:
                # t(event) start < t(anomaly) start and t(event) end > t(anomaly) end
                if pd.to_datetime(period[0]) < a[0] and pd.to_datetime(period[1]) > a[1]:
                    print(a[0]," - ",a[1]," is fully included in the event: ",period[0]," - ",period[1])
                    df_events.loc[ind,'class'] = "A" #anomaly
                elif pd.to_datetime(period[0]) < a[0] and pd.to_datetime(period[1])+pd.Timedelta(1,'m') > a[1]:
                    print(a[0]," - ",a[1]," is partially included (using a safety band on the right) in ",period[0]," - ",period[1])
                    df_events.loc[ind,'class'] = "A" #anomaly
                elif pd.to_datetime(period[0])-pd.Timedelta(1,'m') < a[0] and pd.to_datetime(period[1]) > a[1]:
                    print(a[0]," - ",a[1]," is partially included (using a safety band on the left) in ",period[0]," - ",period[1])
                    df_events.loc[ind,'class'] = "A" #anomaly
                elif pd.to_datetime(period[0]) > a[0] and pd.to_datetime(period[1]) < a[1]:
                    print("The anomaly ",a[0]," - ",a[1]," includes the event ",period[0]," - ",period[1])
                    df_events.loc[ind,'class'] = "A" #anomaly
                elif pd.to_datetime(period[0]) > a[0]-pd.Timedelta(1,'m') and pd.to_datetime(period[1]) < a[1]:
                    print("The anomaly ",a[0]," - ",a[1]," partially includes (using a safety band on the left) the event ",period[0]," - ",period[1])
                    df_events.loc[ind,'class'] = "A" #anomaly
                elif pd.to_datetime(period[0]) > a[0] and pd.to_datetime(period[1]) < a[1]+pd.Timedelta(1,'m'):
                    print("The anomaly ",a[0]," - ",a[1]," partially includes (using a safety band on the right) the event ",period[0]," - ",period[1])
                    df_events.loc[ind,'class'] = "A" #anomaly
                else:
                    df_events.loc[ind,'class'] = "N" #normal - not an anomaly
        print("---Checking for anomalous period fully/partially included in our events: DONE")
        return df_events

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

def import_data(observation_start,observation_end):
    print('\n---Importing the list of Events Dataframes via data manager: STARTED')
    fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
             observation_start, observation_end, 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)

    list_event = fabbri1905_fabax6pdb.measure_pd_dataevent_samples

    _resampling_period_string = fabbri1905_fabax6pdb._resampling_period_string

    ground_truths = fabbri1905_fabax6pdb.groundtruths
    #print(list_event)
    print('---Importing the list of Events Dataframes via data manager: DONE')
    return (list_event,_resampling_period_string,ground_truths)

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
        return df_events_microfeatures.T
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
    parser.add_argument('-l','--labeler', help='Human Labeler name or nickname',required=True)
    parser.add_argument('-s','--session',help='Session name, e.g. anomaly_normal_20191016', required=True)
    parser.add_argument('-b','--begin', help='Observation period start timestamp, e.g 2019-02-04 00:00:01',required=True)
    parser.add_argument('-e','--end', help='Observation period end timestamp',required=True)
    parser.add_argument('-j','--path', help='json file path',required=False)
    args = parser.parse_args()
    
    print("\n---Params you typed")
    print ("Human Labeler name (or nickname): %s" % args.labeler )
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
    #list_events = import_data_events(args.begin,args.end)
    list_events,resampling_period,ground_truths  = import_data(args.begin,args.end)
    df_events_microfeatures = get_event_microfeatures(list_events,True)

    gtg_obj = GroudTruthGenerator(args.labeler, args.session, args.begin, args.end, json_path,ground_truths)
    gtg_obj.get_ground_truth_periods()
    print("\nAnomalous time periods labeled by the human ",args.labeler)
    print(gtg_obj.anomaly_periods)
    gtg_obj.get_ground_truth_continuous_signal()
    #resampling_period = get_resampling_period_string()
    gtg_obj.resample_ground_truth_continuous_signal(resampling_period)
    print("\nAnomalous time periods labeled by the human ",args.labeler, " (after resampling)")
    gtg_obj.get_anomaly_periods_after_resample()
    
    df_event_label = gtg_obj.check_eventi_anomaly_inclusion(df_events_microfeatures)
    event_label_series = pd.Series(df_event_label['class'], index= df_event_label.index)


if __name__ == '__main__':
    main()
