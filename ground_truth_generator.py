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
import seaborn as sns

import data_manager
import data_viewer
import data_labeler
import signal_processor

#%%
## class definition
class GroudTruthGenerator:

    def __init__(self,customer ='' ,network='',source='',db='',host='',labeler = '' ,model = '', experiment_start = '', experiment_end ='', n_ago='', event_period ='',guardperiod = '' , json_path = '',groundtruths_json = [] ):
        
        self.customer = customer
        self.network = network
        self.source = source
        self.db = db
        self.host = host        
        self.labeler = labeler
        self.model = model
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.n_ago = n_ago
        self.json_path = json_path
        self.json_obj = ''
        self.anomaly_periods = []
        self.anomaly_periods_resample = []
        self.gt_series = pd.Series()
        self.gt_series_resample = pd.Series()
        self.groundtruths_json = []
        self.invalid_class_value = ''
        self.periods_list = []
        self.class_dict = {}
        self.list_events = []
        self.event_period = ''
        self.guardperiod = ''
        self._resampling_period_string = ''
        self.df_events_microfeatures = None
        self.df_events_label = pd.DataFrame(columns=['label'])

        #self.load_json()
        self.import_data()

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
    #import via data_manager the events dataframe                          #
    ########################################################################
    def import_data(self):
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
            event_period = '15m'
        else:
            event_period = self.event_period

        print(experiment_start)
        print(experiment_end)
        print('\n---Importing the list of Events Dataframes via data manager: STARTED')
        fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
                self.customer, self.network, self.source, self.db, self.host,
                str(experiment_start), str(experiment_end), 'Europe/Rome',
                'diagnostics_map.json', event_minimum_period=event_period, local_data=True,
                database_queries=True, preprocess_data=True)

        self.list_events = fabbri1905_fabax6pdb.measure_pd_dataevent_samples
        self.get_event_microfeatures(True)
        self._resampling_period_string = fabbri1905_fabax6pdb._resampling_period_string
        self.groundtruths_json = fabbri1905_fabax6pdb.groundtruths
        #print(list_event)
        print('---Importing data via data manager: DONE')
        #return (list_event,_resampling_period_string,ground_truths)
   
   
    ########################################################################
    #combine all the events and generate a dataframe with microfeatures    #
    #it can also apply feature engineering (mean and std method)           #
    ########################################################################
    def get_event_microfeatures(self,feature_engineering):
        if feature_engineering == True:
            ## work on all events - shape: n.events x 36features (mean + variance)
            print('\n---Bulding the Event Dataframe with microfeatures (mean and std method): STARTED')
            events_dict = {}
            for j, event in enumerate(self.list_events):
                if j in [250,500,750,900,1000,len(self.list_events)-1]:
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
            self.df_events_microfeatures = df_events_microfeatures.T
            print('---Bulding the Event Dataframe with microfeatures: DONE')
            #return df_events_microfeatures.T
        else:
            ## work on all events - shape: n.events x 16200 (micro)features
            print('\n---Bulding the Event Dataframe with microfeatures (without feature engineering): STARTED')
            events_dict = {}
            for j, event in enumerate(self.list_events):
                if j in [250,500,750,900,1000,len(self.list_events)-1]:
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
            self.df_events_microfeatures = df_events_microfeatures.T
            print('\n---Bulding the Event Dataframe with microfeatures (without feature engineering): DONE')
        #return df_events_microfeatures.T

    
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
                        
                        for period in model['periods']: #filter period of Experiment
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
        exp_start = pd.to_datetime(self.experiment_start).tz_localize(time_zone)
        exp_end = pd.to_datetime(self.experiment_end).tz_localize(time_zone)

        for ps,pe in self.periods_list: #labeling period start, period end
            ps = str(pd.to_datetime(ps).tz_convert(time_zone))
            pe = str(pd.to_datetime(pe).tz_convert(time_zone))
            if prev == '':
                timezone_index = pd.date_range(str(exp_start),ps, freq='1T',closed = 'left')
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
            timezone_index = pd.date_range(prev,str(exp_end), freq='1T', closed = 'right')
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
    def resample_ground_truth_continuous_signal(self):
        print('\n---Resampling the GroundTruth continuous signal: STARTED')
        self.gt_series_resample = self.gt_series.resample(self._resampling_period_string).pad()
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
        
    
    def check_event_anomaly_inclusion(self):
        
        print("\n---Checking for anomalous period fully/partially included in our events: STARTED")
        #anomalies = gtg_obj.anomaly_periods_resample
        if self.guardperiod == '':
            self.guardperiod = '60'
        for ind in self.df_events_microfeatures.index:
            period = ind.split(" - ")
            event_labels = []
            event_msg = []
            for a in self.anomaly_periods_resample:
                # t(event) start < t(anomaly) start and t(event) end > t(anomaly) end
                if pd.to_datetime(period[0]) < a[0] and pd.to_datetime(period[1]) > a[1]:
                    #print(a[0]," - ",a[1]," is fully included in the event: ",period[0]," - ",period[1])
                    #self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 1
                    event_labels.append(1)
                    msg = str(a[0])+" - "+str(a[1])+" is fully included in the event: "+period[0]+" - "+period[1]
                    event_msg.append(a[0]," - ",a[1]," is fully included in the event: ",period[0]," - ",period[1])
                    #break
                elif pd.to_datetime(period[0]) < a[0] and pd.to_datetime(period[1])+pd.Timedelta(self.guardperiod,'s') > a[1]:
                    #print(a[0]," - ",a[1]," is partially included (using a safety band on the right) in ",period[0]," - ",period[1])
                    #self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = "1"
                    event_labels.append(1)
                    msg = str(a[0])+" - "+str(a[1])+" is partially included (using a safety band on the right) in "+period[0]+" - "+period[1]
                    event_msg.append(msg)

                    #break
                elif pd.to_datetime(period[0])-pd.Timedelta(self.guardperiod,'s') < a[0] and pd.to_datetime(period[1]) > a[1]:
                    #print(a[0]," - ",a[1]," is partially included (using a safety band on the left) in ",period[0]," - ",period[1])
                    #self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 1
                    event_labels.append(1)
                    msg = str(a[0])+" - "+str(a[1])+" is partially included (using a safety band on the left) in "+period[0]+" - "+period[1]
                    event_msg.append(msg)
                    #break
                elif pd.to_datetime(period[0]) > a[0] and pd.to_datetime(period[1]) < a[1]:
                    #print("The anomaly ",a[0]," - ",a[1]," includes the event ",period[0]," - ",period[1])
                    #self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 1
                    event_labels.append(1)
                    msg = "The anomaly "+str(a[0])+" - "+str(a[1])+" includes the event "+period[0]+" - "+period[1]
                    event_msg.append(msg)
                    #break
                elif pd.to_datetime(period[0]) > a[0]-pd.Timedelta(self.guardperiod,'s') and pd.to_datetime(period[1]) < a[1]:
                    #print("The anomaly ",a[0]," - ",a[1]," partially includes (using a safety band on the left) the event ",period[0]," - ",period[1])
                    #self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 1
                    event_labels.append(1)
                    msg = "The anomaly "+str(a[0])+" - "+str(a[1])+" partially includes (using a safety band on the left) the event "+period[0]+" - "+period[1]
                    event_msg.append(msg)
                    #break
                elif pd.to_datetime(period[0]) > a[0] and pd.to_datetime(period[1]) < a[1]+pd.Timedelta(self.guardperiod,'s'):
                    #print("The anomaly ",a[0]," - ",a[1]," partially includes (using a safety band on the right) the event ",period[0]," - ",period[1])
                    #self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 1
                    event_labels.append(1)
                    msg = "The anomaly "+str(a[0])+" - "+str(a[1])+" partially includes (using a safety band on the right) the event "+period[0]+" - "+period[1]
                    event_msg.append(msg)
                    #break
                else:
                    #self.df_events_microfeatures.loc[ind,'label'] = 0 #normal - not an anomaly
                    #self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 0
                    event_labels.append(0)
                    #event_msg.append()
            if len(set(event_labels)) > 1: #when we have differente human label for the same period
                print("The event is ambiguous. I won't assign neither 1 or 0. I will assign -1")
                self.df_events_microfeatures.loc[ind,'label'] = -1 #anomaly
                self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = -1
            elif event_labels[0] == 1:
                self.df_events_microfeatures.loc[ind,'label'] = 1 #anomaly
                self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 1
                print(event_msg[0])
            elif event_labels[0] == 0:
                self.df_events_microfeatures.loc[ind,'label'] = 0 #normal - not an anomaly
                self.df_events_label.loc[pd.to_datetime(period[0]),'label'] = 0


        print("---Checking for anomalous period fully/partially included in our events: DONE")
        #return (df_events,df_events_label)


#%%
def main():
       
    #json_path = "classes.json" # default value
    json_path = "diagnostics_map.json" # default value

    while True:
        parser = argparse.ArgumentParser(description='Welcome to Ground Truth Generator by WurthPhoenix')
        parser.add_argument('-c','--customer', help='Customer which you are interested in searching for anomalies, e.g fabbri',required=True)
        parser.add_argument('-n','--network', help='Customer\'s network, e.g alyvix',required=True)
        parser.add_argument('-s','--source', help='Customer\'s data source for pulling the data, e.g influx ',required=True)
        parser.add_argument('-d','--db', help='A specific database in the data source, e.g fabbri1905',required=True)
        parser.add_argument('-H','--host', help='A specific host whose data you want to analyze, e.g FABAX6PDB',required=True)
        parser.add_argument('-l','--labeler', help='Human Labeler name or nickname',required=True)
        parser.add_argument('-m','--model',help='Model name, e.g. normal_anomaly', required=True)
        parser.add_argument('-b','--begin', help='Experiment period start timestamp, e.g 2019-02-04 00:00:01',required=False)
        parser.add_argument('-e','--end', help='Experiment period end timestamp',required=False)
        parser.add_argument('-a','--nago', help='You can ask for the last n hours or days,e.g. 7days',required=False)
        parser.add_argument('-p','--eventperiod', help='Set the event duration, e.g 15m (m stands for minutes)',required=False)
        parser.add_argument('-g','--guardperiod', help='Set the guard window (in seconds) used for event-anomaly inclusion, e.g 60',required=False)
        parser.add_argument('-j','--path', help='Diagnostic map json file path',required=False)
        args = parser.parse_args()
        if (args.begin and args.end) or args.nago:
            break
        else: 
            print('You have to insert the experiment period. Try setting --begin (-b) and --end (-e). Alternatively set -nago (-a). \nFor more details, see --help')
    
    print("\n---Params you typed")
    print ("Customer under analysis: %s" % args.customer )
    print ("Customer\'s network: %s" % args.network )
    print ("Customer\'s data source: %s" % args.source )
    print ("Database: %s" % args.db )
    print ("Host: %s" % args.host )
    print ("Human Labeler alias: %s" % args.labeler )
    print ("Labeling model: %s" % args.model )
    print ("Experiment period start: %s" % args.begin )
    print ("Experiment period end: %s" % args.end )
    print ("Experiment period (alternative method): %s ago" % args.nago )
    print ("Event period: %s" % args.eventperiod )
    print ("Guard window: %s" % args.guardperiod )

    if args.path:
        json_path = args.path
    print ("json file path: %s" % json_path  )

    gtg_obj = GroudTruthGenerator(args.customer,args.network,args.source,args.db,args.host,args.labeler, args.model, args.begin, args.end, args.nago, args.eventperiod, args.guardperiod, json_path)
    gtg_obj.get_ground_truth_periods()
    print("\nAnomalous time periods labeled by the human ",args.labeler)
    print(gtg_obj.anomaly_periods)
    gtg_obj.get_ground_truth_continuous_signal()
    gtg_obj.resample_ground_truth_continuous_signal()
    print("\nAnomalous time periods labeled by the human ",args.labeler, " (after resampling)")
    gtg_obj.get_anomaly_periods_after_resample()
    
    gtg_obj.check_event_anomaly_inclusion()
    events_label_series = pd.Series(gtg_obj.df_events_label['label'], index= gtg_obj.df_events_label.index)
    print(events_label_series)
    plt.plot(events_label_series.index,events_label_series.astype(int))
    plt.show()
    #events_label_series.astype(int).plot()
if __name__ == '__main__':
    main()

