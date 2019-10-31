##Ground Truth Generator

#%%
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import shelve
import argparse
import json
from functools import reduce
from collections import Counter

import data_manager
import data_sampler
import data_viewer
import data_labeler
import signal_processor
import feature_extractor

#%%
## class definition
class GroudTruthGenerator:

    def __init__(self,customer='' , network='', source='', db='', host='', labeler='', model='', experiment_start='', \
                experiment_end ='', n_ago='', event_period='', guardperiod='', json_path='', groundtruths_json = [] ):
        
        self.time_zone = 'Europe/Rome'
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
        self.invalid_periods = []
        self.class_series_dict = {}
        self.gt_series = pd.Series()
        self.gt_series_events_label = pd.Series()
        self.groundtruths_json = []
        self.invalid_class_value = ''
        self.default_value = ''
        self.periods_list = []
        self.class_dict = {}
        self.list_events = []
        self.event_period = event_period
        self.guardperiod = pd.to_timedelta(guardperiod,'s')
        self.df_events_microfeatures = None
        self.df_events_label = pd.DataFrame(columns=['label'])

        self.import_data()

        print("\nGTG instantiated\n")


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
                str(experiment_start), str(experiment_end), 'Europe/Rome',
                'diagnostics_map.json', event_minimum_period=self.event_period, local_data=True,
                database_queries=True, preprocess_data=True)

        self.list_events = fabbri1905_fabax6pdb.measure_pd_dataevent_samples

        f_extractor = feature_extractor.FeatureExtractor(self.list_events)
        f_extractor.get_event_microfeatures(True)
        self.df_events_microfeatures = f_extractor.df_events_microfeatures.copy()

        #self._resampling_period_string = fabbri1905_fabax6pdb._resampling_period_string
        self.groundtruths_json = fabbri1905_fabax6pdb.groundtruths
        #print(list_event)
        print('---Importing data via data manager: DONE')
        #return (list_event,_resampling_period_string,ground_truths)
   

    def get_ground_truth_periods(self):
        """load recorded period of each class, recorded in the json file"""

        print('\n---Loading periods manually labeled, for each class, from the selected Json: STARTED')
        for truth in self.groundtruths_json:#filter the user
            if truth['labeler_alias'].lower().strip() == self.labeler.lower().strip(): 
                for model in truth['labeling_models']: #filter the session
                    if model['model_name'].lower().strip() == self.model.lower().strip():
                        self.invalid_class_value = model['invalid_class_value']
                        
                        for period in model['periods']: #filter period of Experiment
                            start_tmp = pd.to_datetime(period['period_start']).tz_convert(self.time_zone)
                            end_tmp = pd.to_datetime(period['period_end']).tz_convert(self.time_zone)
                            self.periods_list.append((start_tmp,end_tmp))
                        self.periods_list.sort(key=lambda tup: tup[0])#sort by period start

                        flag = False #it is True when the class_name 'anomaly' is found in the json
                        for class_ in model['classes']: #filter class - e.g anomaly or normal
                            self.class_dict[class_['class_label']] = {}
                            self.class_dict[class_['class_label']]['value'] = class_['class_value'].lower().strip()
                            self.class_dict[class_['class_label']]['type'] = class_['class_type'].lower().strip()
                            
                            if not class_['class_type'].lower().strip() == 'default':
                                class_period_list = []
                                for class_period in class_['class_periods']:
                                    start_tmp = pd.to_datetime(class_period['period_start']).tz_convert(self.time_zone)
                                    end_tmp = pd.to_datetime(class_period['period_end']).tz_convert(self.time_zone)
                                    class_period_list.append((start_tmp,end_tmp))
                                class_period_list.sort(key=lambda tup: tup[0])#sort by class period start
                                self.class_dict[class_['class_label']]['periods'] = class_period_list
                            else:
                                self.default_value = int(self.class_dict[class_['class_label']]['value'])
                                self.class_dict[class_['class_label']]['periods'] = []

                            if class_['class_label'] == 'anomaly':
                                flag = True
                                #for class_period in class_['class_periods']:
                                    #self.anomaly_periods.append((class_period['period_start'],class_period['period_end']))

                        if flag == False:
                            print('---Loading periods manually labeled as Anomaly from the selected Json: FAILED\n \
                                the class_name "anomaly" does not exist in the selected json file')
                        else:
                            if len(self.class_dict['anomaly']['periods']) > 0:
                                print('---Loading periods manually labeled, for each class, from the selected Json: DONE')
                            else:
                                print('---Loading periods manually labeled, for each class, from the selected Json: FAILED\n \
                                    The class_name "anomaly" is empty (no anomalous period time recorded) in the selected Json')

    
    def get_class_continuous_signal(self):
        """generate a continuous signal from the anomaly periods recorded in Json"""

        print('\n---Generation of the continuous signal for each class: STARTED')
        #self.time_zone = 'Europe/Rome'
        prev = ''
        exp_start = pd.to_datetime(self.experiment_start).tz_localize(self.time_zone)
        exp_end = pd.to_datetime(self.experiment_end).tz_localize(self.time_zone)

        self.class_dict['period'] = {}
        self.class_dict['period']['periods'] = self.periods_list
        self.class_dict['period']['type'] = 'period'
        self.class_dict['period']['value'] = '1'
        sec = pd.to_timedelta(1,'s')
        for serie in self.class_dict.keys():
            #if not serie == 'anomaly':
                #continue
            prev = ''
            
            if self.class_dict[serie]['type'] == 'default':
                timezone_index = pd.date_range(exp_start,exp_end, freq='1S',tz = self.time_zone)
                utc_index = pd.to_datetime(timezone_index, utc=False)
                self.class_dict[serie]['signal'] = pd.Series([1] * (len(utc_index)), index=utc_index)
            else:
                timezone_index = pd.date_range(exp_start,exp_end, freq='1S',tz = self.time_zone)
                utc_index = pd.to_datetime(timezone_index, utc=False)
                #self.class_dict[serie]['signal'] = pd.Series([0] * (len(utc_index)), index=utc_index)
                self.class_dict[serie]['signal'] = pd.Series([self.default_value] * (len(utc_index)), index=utc_index)

            for ps,pe in self.class_dict[serie]['periods']: #labeling period start, period end
                #ps = pd.to_datetime(ps).tz_convert(self.time_zone)
                #pe = pd.to_datetime(pe).tz_convert(self.time_zone)
                overlapping = False
                if serie == 'period':
                    if str(prev) == '':
                        timezone_index = pd.date_range(exp_start,ps, freq='1S',closed = 'left',tz = self.time_zone)
                        utc_index = pd.to_datetime(timezone_index, utc=False)
                        self.class_dict[serie]['signal'][exp_start:ps-sec] = [int(self.invalid_class_value)] * len(utc_index)
                        self.invalid_periods.append((exp_start,ps-sec))
                    else: 
                        if ps > prev:
                            print("not overlapping periods")
                            timezone_index = pd.date_range(prev,ps, freq='1S',closed = 'left',tz = self.time_zone)
                            utc_index = pd.to_datetime(timezone_index, utc=False)
                            self.class_dict[serie]['signal'][prev+sec:ps-sec] = [int(self.invalid_class_value)] * (len(utc_index)-1)
                            self.invalid_periods.append((prev+sec,ps-sec))
                        else:#ps <= prev
                            print("overlapping periods")
                            overlapping = True
                            timezone_index = pd.date_range(prev,pe, freq='1S',closed = 'right',tz = self.time_zone)
                            utc_index = pd.to_datetime(timezone_index, utc=False)
                            #self.class_dict[serie]['signal'][prev+sec:pe] = [1] * (len(utc_index))
                            self.class_dict[serie]['signal'][prev+sec:pe] = [int(self.class_dict[serie]['value'])] * (len(utc_index))

                if not overlapping:#it enters here also when the serie is not 'period'
                    timezone_index = pd.date_range(ps,pe, freq='1S',tz = self.time_zone)
                    utc_index = pd.to_datetime(timezone_index, utc=False)
                    self.class_dict[serie]['signal'][ps:pe] = [int(self.class_dict[serie]['value'])] * len(utc_index)
                prev = pe

            if not prev == '' and serie == 'period':
                timezone_index = pd.date_range(prev,exp_end, freq='1S', closed = 'right',tz = self.time_zone)
                utc_index = pd.to_datetime(timezone_index, utc=False)
                self.class_dict[serie]['signal'][prev+sec:exp_end] = [int(self.invalid_class_value)] * len(utc_index)
                self.invalid_periods.append((prev+sec,exp_end))
            
            self.class_series_dict[serie] = self.class_dict[serie]['signal']

        print('---Generaton of the continuous signal: DONE')


    def get_ground_truth_continuous_signal(self):
        """get the grounf truth cont. signal by combining all the class signals"""

        print('\n---Generation of the GT continuous signal: STARTED')
        dict_indeces = {}
        for serie in self.class_series_dict.keys():
            if self.class_dict[serie]['type'] == 'default':
                #default_value = int(self.class_dict[serie]['value'])
                time_index = self.class_series_dict[serie].index
                n_rows = len(time_index)
                continue
            if not serie == 'period': 
                #dict_indeces[serie] = np.where(self.class_series_dict[serie] == 1)[0]
                dict_indeces[serie] = np.where(self.class_series_dict[serie] == int(self.class_dict[serie]['value']))[0]
            else:
                #invalid_indeces = np.where(self.class_series_dict[serie] == -1)[0]
                invalid_indeces = np.where(self.class_series_dict[serie] == int(self.invalid_class_value))[0]
                continue
        if len(dict_indeces.keys()) > 1:
            intersect_indeces = reduce(np.intersect1d, ([indeces for serie,indeces in dict_indeces.items()]))
            invalid_indeces = np.append(invalid_indeces,intersect_indeces)

        label_signal = np.array([self.default_value]*n_rows)
        for serie,positions in dict_indeces.items():
            value = self.class_dict[serie]['value']
            for pos in positions:
                label_signal[pos] = value
        for invalid_pos in invalid_indeces:
            label_signal[invalid_pos] = self.invalid_class_value

        self.gt_series = pd.Series(label_signal,index=time_index)


    def perform_event_labeling(self):
        """
            (v1 - using inclusion criterion)
            it checks if an event corresponds to a class or not. In case the
            event is included in an invalid period it will be assigned as invalid 
        """
        print("\n---Checking for anomalous period fully/partially included in our events v2: STARTED")
        if self.guardperiod == '':
            self.guardperiod = pd.to_timedelta(60,'s')

        for ind in self.df_events_microfeatures.index:
            period = ind.split(" - ")
            event_labels = []
            event_msg = []
            
            period[0] = pd.to_datetime(period[0]).tz_convert(self.time_zone)
            period[1] = pd.to_datetime(period[1]).tz_convert(self.time_zone)
            
            for serie in self.class_dict.keys():
                if self.class_dict[serie]['type'] == 'default' or serie == 'period':
                    continue
                for a in self.class_dict[serie]['periods']:
                    if period[0] < a[0] and period[1] > a[1]:
                        event_labels.append(1)
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" is fully included in the event: "+str(period[0])+" - "+str(period[1])
                        event_msg.append(msg)
                        #break
                    elif period[0] < a[0] and period[1]+self.guardperiod > a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" is partially included (using a safety band on the right) in "+str(period[0])+" - "+str(period[1])
                        event_msg.append(msg)
                        #break
                    elif period[0]-self.guardperiod < a[0] and period[1] > a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" is partially included (using a safety band on the left) in "+str(period[0])+" - "+str(period[1])
                        event_msg.append(msg)
                        #break
                    elif period[0] > a[0] and period[1] < a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" includes the event "+str(period[0])+" - "+str(period[1])
                        event_msg.append(msg)
                        #break
                    elif period[0] > a[0]-self.guardperiod and period[1] < a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" partially includes (using a safety band on the left) the event "+str(period[0])+" - "+str(period[1])
                        event_msg.append(msg)
                        #break
                    elif period[0] > a[0] and period[1] < a[1]+self.guardperiod:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" partially includes (using a safety band on the right) the event "+str(period[0])+" - "+str(period[1])
                        event_msg.append(msg)
                        #break

            if len(set(event_labels)) > 1: #when we have different label for the same event
                print("The event is ambiguous. I won't assign neither 1 or 0. I will assign -1")
                self.df_events_microfeatures.loc[ind,'label'] = self.invalid_class_value #anomaly
                self.df_events_label.loc[period[0],'label'] = self.invalid_class_value
            elif len(set(event_labels)) == 0:
                invalid = False
                for inv in self.invalid_periods:
                    latest_start = max(period[0],inv[0])
                    earliest_end = min(period[1],inv[1])
                    if latest_start <= earliest_end:
                        invalid = True
                        break
                if invalid == False:
                    event_labels.append(self.default_value)
                    self.df_events_microfeatures.loc[ind,'label'] = self.default_value #anomaly
                    self.df_events_label.loc[period[0],'label'] = self.default_value
                else:
                    event_labels.append(self.invalid_class_value)
                    self.df_events_microfeatures.loc[ind,'label'] = self.invalid_class_value #anomaly
                    self.df_events_label.loc[period[0],'label'] = self.invalid_class_value
            else:
                self.df_events_microfeatures.loc[ind,'label'] = event_labels[0] #anomaly
                self.df_events_label.loc[period[0],'label'] = event_labels[0]

        self.df_events_label = self.df_events_label.sort_index()
        print("---Checking for anomalous period fully/partially included in our events: DONE")        #return (df_events,df_events_label)

    def find_majority(self,votes):
        """
            helper function for the perform_event_labeling_from_gt_signal method. 
            It looks for the most common value in an event period
        """
        vote_count = Counter(votes)
        top = vote_count.most_common(1)
        if len(top)>1:
            # It is a tie and it maight be ambiguous
            return -1
        return top[0][0]

    def perform_event_labeling_from_gt_signal(self):
        """
            (v2 - using gt event extraction)
            This method generates the labels for events, 
            by splitting into events the ground trith signal
        """
        print("\n---Performing Event Labeling GT signal: STARTED")
        gt_events, event_minimum_samples = data_sampler.sample_dataevents(self.gt_series,event_minimum_period=self.event_period)
        #events_dict = {}
        events_dict = {}
        n_events = len(gt_events)
        progress_list = [round(n_events*20/100),round(n_events*40/100),round(n_events*50/100),round(n_events*70/100),round(n_events*90/100),n_events-1]
        for j, event in enumerate(gt_events):
            if j in progress_list:
                print('Event modelling: event dataframe n',j, 'out of ',n_events)
            label_list = []
            event_timespan = []
            count = 0
            for i in event.index:
                label_list.append(event[i])
            event_timespan.append((event.index[0]))
            event_timespan.append((event.index[len(event)-1]))
            value = self.find_majority(label_list)
            #events_dict[' - '.join(event_timespan)] = value
            events_dict[event_timespan[0]] = value

        #df_gt_events = pd.DataFrame(events_dict_single,index=['label'])
        self.gt_series_events_label = pd.Series(events_dict,index=events_dict.keys())
        #self.df_gt_events_label = df_gt_events.T
        print("---Performing Event Labeling from GT signal: DONE")

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
            print('You have to insert the experiment period. Try setting --begin (-b) \
                    and --end (-e). Alternatively set -nago (-a). \nFor more details, see --help')
    
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

    #instantiate the gtg object
    gtg_obj = GroudTruthGenerator(args.customer,args.network,args.source,args.db, \
                                    args.host,args.labeler, args.model, args.begin, args.end, args.nago, \
                                    args.eventperiod, args.guardperiod, json_path)
    
    #read from json, the recorded periods for each class
    gtg_obj.get_ground_truth_periods()
    for serie,data in gtg_obj.class_dict.items():
        if not serie == 'period' and not data['type'] == 'default':
            print('Periods of class',serie,' manually labeled by a human labeler and reported in json file')
            print(data['periods'])
            print('\n')

    #generate signals for each class
    gtg_obj.get_class_continuous_signal()
    
    #generate the ground truth signal
    gtg_obj.get_ground_truth_continuous_signal()

    #perform event labeling (v1 - using inclusion criterion)
    gtg_obj.perform_event_labeling()

    #perform event labeling (v2 - using gt signal event extraction)
    gtg_obj.perform_event_labeling_from_gt_signal()

    
    #print subplots for each generated signal
    fig = plt.figure(figsize=(10,10))
    min_y = min(gtg_obj.gt_series)
    max_y = max(gtg_obj.gt_series)
    yticks = np.arange(min_y,max_y+1)

    plt.subplot(6,1,1)
    plt.title('Period')
    plt.xticks([])
    plt.yticks(yticks)
    plt.ylim(min_y,max_y)
    gtg_obj.class_series_dict['period'].plot(label = 'period signal',c = 'black')

    plt.subplot(6,1,2)
    plt.title('Normal')
    plt.xticks([])
    plt.yticks(yticks)
    plt.ylim(min_y,max_y)
    gtg_obj.class_series_dict['normal'].plot(label = 'normal signal',c = 'black', linestyle = 'dashed')

    plt.subplot(6,1,3)
    plt.title('Anomaly')
    plt.xticks([])
    plt.yticks(yticks)
    plt.ylim(min_y,max_y)
    gtg_obj.class_series_dict['anomaly'].plot(label = 'anomaly signal',c = 'orange')

    plt.subplot(6,1,4)
    plt.title('Control')
    plt.xticks([])
    plt.yticks(yticks)
    plt.ylim(min_y,max_y)
    gtg_obj.class_series_dict['control'].plot(label = 'GT signal',c = 'yellow')

    plt.subplot(6,1,5)
    plt.title('GT')
    plt.xticks([])
    plt.yticks(yticks)
    plt.ylim(min_y,max_y)
    gtg_obj.gt_series.plot(label = 'GT signal',c = 'red',markersize=20)

    plt.subplot(6,1,6)
    plt.title('Events label series')
    plt.yticks(yticks)
    plt.ylim(min_y,max_y)
    gtg_obj.gt_series_events_label.plot(label = 'GT signal label',c = 'red',markersize=20)

    plt.show()

    
    

if __name__ == '__main__':
    #main()
    pass

