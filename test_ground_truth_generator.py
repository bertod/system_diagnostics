from AnomalyDetector import GroudTruthGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import matplotlib.patches as mpatches

global gtg_test_json

gtg_test_json = '{"ground_truths": [ \
                            { \
                                "labeler_alias": "Peter",\
                                "labeling_models": [\
                                    {\
                                        "model_name": "normal_anomaly",\
                                        "date_modified": "2019/10/17 12:30:00+00:00",\
                                        "period_guard_seconds": "60",\
                                        "invalid_class_label": "invalid",\
                                        "invalid_class_value": "-1",\
                                        "periods": [\
                                            {\
                                                "period_start": "2019/10/31 23:00:01+00:00",\
                                                "period_end": "2019/11/11 23:59:59+00:00"\
                                            }\
                                        ],\
                                        "classes": [\
                                            {\
                                                "class_type": "default",\
                                                "class_label": "normal",\
                                                "class_value": "0"\
                                            },\
                                            {\
                                                "class_type": "regular",\
                                                "class_label": "anomaly",\
                                                "class_value": "1",\
                                                "class_periods": [\
                                                    {\
                                                        "period_start": "2019/11/02 01:30:00+00:00",\
                                                        "period_end": "2019/11/02 09:30:00+00:00"\
                                                    }\
                                                ]\
                                            }\
                                        ]\
                                    }\
                                ]\
                            }\
                        ]}'

class Test():

    def __init__(self):
        self.gtg_instance = None
        print('We are going to test ANOMALY DETECTOR Module')
    
    def load_json_file(self,file_path):
        try:
            json_file = open(file_path)
        except IOError:
            print('error | json file opening issue')
            return False
        try:
            json_object = json.load(json_file)
        except ValueError:
            print('error | json file loading issue')
            return False
        return json_object
    
    def load_json_string(self,json_string):
        try:
            json_object = json.loads(json_string)
        except ValueError:
            print('error | json file loading issue')
            return False
        return json_object

    def sample_dataevents(self,pd_dataframe, event_minimum_period='10m'):
        pd_dataframe_sample_amount = pd_dataframe.index.size
        pd_dataframe_sample_period = pd_dataframe.index.freq.delta
        pd_event_minimum_period = pd.to_timedelta(event_minimum_period)
        event_minimum_samples = int(pd_event_minimum_period //
                                    pd_dataframe_sample_period)
        if event_minimum_samples % 2:
            event_minimum_samples += 1
        event_maximum_sampling_period = event_minimum_samples // 2
        serial_event_amount = int(pd_dataframe_sample_amount //
                                event_minimum_samples)
        sampled_event_amount = (serial_event_amount * 2) - 1

        sampled_event_serial_slices = []
        for sampled_event_serial_number in range(sampled_event_amount):
            sampled_event_serial_slice_start = \
                sampled_event_serial_number * event_maximum_sampling_period
            sampled_event_serial_slice_end = \
                sampled_event_serial_slice_start + event_minimum_samples
            sampled_event_serial_slice = slice(sampled_event_serial_slice_start,
                                            sampled_event_serial_slice_end)
            sampled_event_serial_slices.append(sampled_event_serial_slice)

        sampled_events = []
        for sampled_event_serial_slice in sampled_event_serial_slices:
            sampled_events.append(pd_dataframe.iloc[sampled_event_serial_slice])

        return sampled_events, event_minimum_samples

    def get_df_events_index(self,df_events):
        print('\n---Bulding the Event DateTimeIndex: STARTED')
        events_start = []
        n_events = len(df_events)
        progress_list = [round(n_events*20/100),round(n_events*40/100),round(n_events*50/100) \
                        ,round(n_events*70/100),round(n_events*90/100),n_events-1]
        for j, event in enumerate(df_events):
            if j in progress_list:
                print('Event modelling: event dataframe n ',j, 'out of ',n_events)
            events_start.append(event.index[0])
        df_events_index = pd.DatetimeIndex(events_start)
        print('\n---Bulding the Event DateTimeIndex: DONE')
        return df_events_index
        

    def get_ground_truth(self,groundtruths,time_zone,labeler,labeling_model):
        """load recorded period of each class, recorded in the json file"""
        print('\n---Loading periods manually labeled, for each class, from the selected Json: STARTED')
        groundtruths_dict = {}
        print(type(groundtruths))
        for truth in groundtruths:#filter the user
            if truth['labeler_alias'].lower().strip() == labeler.lower().strip():
                groundtruths_dict[truth['labeler_alias']] = {} 
                for model in truth['labeling_models']: #filter the session
                    if model['model_name'].lower().strip() == labeling_model.lower().strip():
                        groundtruths_dict[truth['labeler_alias']][model['model_name']] = {}
                        groundtruths_dict[truth['labeler_alias']][model['model_name']]['invalid_class_value'] = model['invalid_class_value']
                        #self.invalid_class_value = model['invalid_class_value']
                        periods_list = []
                        for period in model['periods']: #filter period of Experiment
                            start_tmp = pd.to_datetime(period['period_start']).tz_convert(time_zone)
                            end_tmp = pd.to_datetime(period['period_end']).tz_convert(time_zone)
                            periods_list.append((start_tmp,end_tmp))
                        periods_list.sort(key=lambda tup: tup[0])#sort by period start

                        groundtruths_dict[truth['labeler_alias']][model['model_name']]['period'] = {}
                        groundtruths_dict[truth['labeler_alias']][model['model_name']]['period']['periods'] = periods_list
                        groundtruths_dict[truth['labeler_alias']][model['model_name']]['period']['type'] = 'period'
                        groundtruths_dict[truth['labeler_alias']][model['model_name']]['period']['value'] = '1'

                        for class_ in model['classes']: #filter class - e.g anomaly or normal
                            groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']] = {}

                            groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']]['value'] = \
                                class_['class_value'].lower().strip()

                            groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']]['type'] = \
                                class_['class_type'].lower().strip()
                            
                            if not class_['class_type'].lower().strip() == 'default':
                                class_period_list = []
                                for class_period in class_['class_periods']:
                                    start_tmp = pd.to_datetime(class_period['period_start']).tz_convert(time_zone)
                                    end_tmp = pd.to_datetime(class_period['period_end']).tz_convert(time_zone)
                                    class_period_list.append((start_tmp,end_tmp))
                                class_period_list.sort(key=lambda tup: tup[0])#sort by class period start
                                groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']]['periods'] = class_period_list
                            else:
                                groundtruths_dict[truth['labeler_alias']][model['model_name']]['default_class_value'] = \
                                    int(groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']]['value'])
                                #self.default_value = int(groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']]['value'])
                                groundtruths_dict[truth['labeler_alias']][model['model_name']][class_['class_label']]['periods'] = []
        return groundtruths_dict

    def test_anomaly_detector(self):
       
        labeler = 'Peter'
        labeling_model = 'normal_anomaly'
        time_zone = "Europe/Rome"
        #exp_start = pd.to_datetime("2019-02-04 00:00:01").tz_localize(time_zone)
        exp_start = pd.to_datetime("2019/11/01 00:00:01").tz_localize(time_zone)
        #exp_end = pd.to_datetime("2019-02-10 00:59:59").tz_localize(time_zone)
        exp_end = pd.to_datetime("2019/11/12 23:59:59").tz_localize(time_zone)
        timezone_index = pd.date_range(exp_start,exp_end, freq='1S',tz = time_zone)
        utc_index = pd.to_datetime(timezone_index, utc=False)
        df_index = pd.DataFrame({},index = utc_index)
        #df_events = df_index.resample("450s").pad() #7 mins and half (half event period)
        df_events,event_minimum_samples = self.sample_dataevents(df_index,'15m')
        df_events_index = self.get_df_events_index(df_events)
        
        gt_json_obj = self.load_json_string(gtg_test_json) #use a json string hard coded here
        #gt_json_obj = self.load_json_file('gtg_test.json') #use a json file

        groundtruths_dict = self.get_ground_truth(gt_json_obj['ground_truths'],time_zone,labeler,labeling_model)
        #self.gtg_instance = GroudTruthGenerator('Peter',labeling_model , "2019-02-04 00:00:01","2019-02-10 00:59:59","15m", \
            #"60", "Europe/Rome", df_events_index, groundtruths_dict)
        
        self.gtg_instance = GroudTruthGenerator(labeler,labeling_model ,"2019/11/01 00:00:01","2019/11/12 23:59:59","15m", \
            "60", "Europe/Rome", df_events_index, groundtruths_dict)


        for serie,data in self.gtg_instance.class_dict.items():
            if not serie == 'period' and not serie == 'default_class_value' and not serie == 'invalid_class_value':
                print('Periods of class',serie,' manually labeled by a human labeler and reported in json file')
                #print(serie)
                print(data['periods'])
                print('\n')


        #generate signals for each class
        self.gtg_instance.get_class_continuous_signal()
        
        #generate the ground truth signal - self.gtg_instance.gt_series attribute
        self.gtg_instance.get_ground_truth_continuous_signal()

        #perform event labeling (v1 - using inclusion criterion) - self.gtg_instance.series_events_label
        self.gtg_instance.perform_event_labeling()

        #perform event labeling (v2 - using gt signal event extraction) - self.gtg_instance.gt_series_events_label
        self.gtg_instance.perform_event_labeling_from_gt_signal()



        print('Checking the two methods used for labeling')
        ind_list = []
        for ind in self.gtg_instance.gt_series_events_label.index:
            if not self.gtg_instance.gt_series_events_label[ind] == self.gtg_instance.series_events_label[ind]:
                #print(ind)
                ind_list.append(ind)

        for ind in ind_list:
            print('GT signal event labeling')
            print(ind, ' -- ', self.gtg_instance.gt_series_events_label[ind])
            print('event labeling')
            print(ind, ' -- ', self.gtg_instance.series_events_label[ind])
            print('\n')

        print('There are ',len(ind_list),' events differently labeled by the two methods')



        fig = plt.figure(figsize=(15,15))
        min_y = min(self.gtg_instance.gt_series)
        max_y = max(self.gtg_instance.gt_series)+1
        yticks = np.arange(min_y,max_y)

        nplots = (len(self.gtg_instance.class_dict.keys())-2) + 3
        pos = 1
        for serie in self.gtg_instance.class_dict.keys():
            if not serie == 'default_class_value' and not serie == 'invalid_class_value':

                plt.subplot(nplots,1,pos)
                plt.title(serie)
                plt.xticks([])
                plt.yticks(yticks)
                plt.ylim(min_y,max_y)
                self.gtg_instance.class_dict[serie]['signal'].plot(label = serie+' signal',c = 'black')
                pos += 1

        plt.subplot(nplots,1,nplots-2)
        plt.title('GT')
        plt.xticks([])
        plt.yticks(yticks)
        plt.ylim(min_y,max_y)
        self.gtg_instance.gt_series.plot(label = 'GT signal',c = 'red',markersize=20)

        plt.subplot(nplots,1,nplots-1)
        plt.title('Events label series - from GT signal')
        plt.xticks([])
        plt.yticks(yticks)
        plt.ylim(min_y,max_y)
        self.gtg_instance.gt_series_events_label.plot(label = 'Event (GT signal method) signal label',c = 'red',markersize=20)

        plt.subplot(nplots,1,nplots)
        plt.title('Events label series -  from fully/partial inclusion')
        plt.yticks(yticks)
        plt.ylim(min_y,max_y)
        self.gtg_instance.series_events_label.plot(label = 'Event (inclusion method) signal label',c = 'red',markersize=20)

        #plt.show()

        #bar chart that clearly shows all the event labels as
        #areas embracing their time validity (every half event period!)
        s_event = self.gtg_instance.gt_series_events_label.copy()
        #x = np.arange(0,60)
        x = np.arange(0,len(s_event.index))
        x_even = x[::2]
        x_odd = x[1::2]
        #0 green , 1 red , 2 yellow , -1 grey
        color_map = {0: '#1EB53A',1:'#EF2B2D',2:'#F9D616',-1:'#AFA593' }

        #color_list_even = [color_map[v] for i,v in s_event[830:890:2].items()]
        color_list_even = [color_map[v] for i,v in s_event[::2].items()]
        #color_list_odd = [color_map[v] for i,v in s_event[831:890:2].items()]
        color_list_odd = [color_map[v] for i,v in s_event[1::2].items()]

        fig = plt.figure(figsize=(15,15))
        #ax = plt.gca()
        ticks = x
        #ticks_label=s_event[830:890].index.astype(str).to_list()
        ticks_label=s_event.index.astype(str).to_list()
        plt.ylim((0,3))
        plt.yticks([0,1,2,3])
        plt.xlim((-1,61))
        plt.xticks(ticks,ticks_label,rotation=90)
        plt.bar(x_even,[1]*len(x_even),width=2, alpha = 0.8,color = color_list_even,align='edge',edgecolor = '#66594C',zorder=10)#even
        plt.bar(x_odd,[2]*len(x_odd),width=2,alpha = 0.5,color = color_list_odd,align='edge',edgecolor = '#66594C',zorder=1)#even
        
        normal = mpatches.Patch(color='#1EB53A', label='normal')
        anomaly = mpatches.Patch(color='#EF2B2D', label='anomaly')
        control = mpatches.Patch(color='#F9D616', label='control')
        invalid = mpatches.Patch(color='#AFA593', label='invalid')
        plt.legend(handles=[normal,anomaly,control,invalid], loc=2,prop={'size': 12})
        plt.title('Event labels - labels as areas')
        plt.show()

def main():
    tester_ad = Test()
    #tester_gtg.run()
    tester_ad.test_anomaly_detector()
    return tester_ad

if __name__ == "__main__":
    tester_ad = main()
