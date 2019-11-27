"""
    GroundTruthGeneratore Module
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import argparse
import json
from collections import Counter
from itertools import combinations

import AnomalyDetector.Tools.data_sampler as data_sampler


class GroundTruthGenerator:

    def __init__(self, labeler='', labeling_model='', experiment_start='', experiment_end='', event_period='',
                 guardperiod='', time_zone='Europe/Rome', df_events_index='', groundtruths_dict={}):
        
        self.labeler = labeler
        self.labeling_model = labeling_model
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.event_period = event_period
        self.guardperiod = pd.to_timedelta(guardperiod, 's')
        self.time_zone = time_zone
        self.df_events_index = df_events_index
        self.groundtruths_dict = groundtruths_dict
        # self.class_dict = self.groundtruths_dict
        self.class_dict = self.groundtruths_dict[self.labeler][self.labeling_model]
        self.invalid_periods = []
        self.gt_series = pd.Series()
        self.gt_series_events_label = pd.Series()
        self.invalid_class_value = self.class_dict['invalid_class_value']
        self.default_value = self.class_dict['default_class_value']
        self.df_events_label = pd.DataFrame(columns=['label'])
        self.series_events_label = pd.Series()

        print("\nGTG instantiated\n")

    def get_class_continuous_signal(self):
        """generate a continuous signal from the class periods recorded in Json"""

        print('\n---Generation of the continuous signal for each class: STARTED')
        prev = ''
        # exp_start = pd.to_datetime(self.experiment_start).tz_convert(self.time_zone)
        exp_start = pd.to_datetime(self.experiment_start).tz_localize(self.time_zone)
        # exp_end = pd.to_datetime(self.experiment_end).tz_convert(self.time_zone)
        exp_end = pd.to_datetime(self.experiment_end).tz_localize(self.time_zone)

        sec = pd.to_timedelta(1, 's')
        for serie in self.class_dict.keys():
            # if not serie == 'anomaly':
            # continue
            if serie == 'invalid_class_value' or serie == 'default_class_value':
                continue
            prev = ''
            if self.class_dict[serie]['type'] == 'default':
                timezone_index = pd.date_range(exp_start, exp_end, freq='1S', tz=self.time_zone)
                utc_index = pd.to_datetime(timezone_index, utc=False)
                self.class_dict[serie]['signal'] = pd.Series([1] * (len(utc_index)), index=utc_index)
            else:
                timezone_index = pd.date_range(exp_start, exp_end, freq='1S', tz=self.time_zone)
                utc_index = pd.to_datetime(timezone_index, utc=False)
                self.class_dict[serie]['signal'] = pd.Series([self.default_value] * (len(utc_index)), index=utc_index)

            for ps, pe in self.class_dict[serie]['periods']:  # labeling period start, period end
                overlapping = False
                if serie == 'period':
                    if str(prev) == '':
                        timezone_index = pd.date_range(exp_start, ps, freq='1S', closed='left', tz=self.time_zone)
                        utc_index = pd.to_datetime(timezone_index, utc=False)
                        self.class_dict[serie]['signal'][exp_start:ps-sec] = \
                            [int(self.invalid_class_value)]*len(utc_index)
                        self.invalid_periods.append((exp_start, ps-sec))
                    else: 
                        if ps > prev:
                            print("not overlapping periods")
                            timezone_index = pd.date_range(prev, ps, freq='1S', closed='left', tz=self.time_zone)
                            utc_index = pd.to_datetime(timezone_index, utc=False)
                            self.class_dict[serie]['signal'][prev+sec:ps-sec] = \
                                [int(self.invalid_class_value)]*(len(utc_index)-1)
                            self.invalid_periods.append((prev+sec, ps-sec))
                        else:  # ps <= prev
                            print("overlapping periods")
                            overlapping = True
                            timezone_index = pd.date_range(prev, pe, freq='1S', closed='right', tz=self.time_zone)
                            utc_index = pd.to_datetime(timezone_index, utc=False)
                            self.class_dict[serie]['signal'][prev+sec:pe] = \
                                [int(self.class_dict[serie]['value'])]*(len(utc_index))

                if not overlapping:  # it enters here also when the serie is not 'period'
                    timezone_index = pd.date_range(ps, pe, freq='1S', tz=self.time_zone)
                    utc_index = pd.to_datetime(timezone_index, utc=False)
                    self.class_dict[serie]['signal'][ps:pe] = [int(self.class_dict[serie]['value'])]*len(utc_index)
                prev = pe

            if not prev == '' and serie == 'period':
                timezone_index = pd.date_range(prev, exp_end, freq='1S', closed='right', tz=self.time_zone)
                utc_index = pd.to_datetime(timezone_index, utc=False)
                self.class_dict[serie]['signal'][prev+sec:exp_end] = [int(self.invalid_class_value)]*len(utc_index)
                self.invalid_periods.append((prev+sec, exp_end))
        print('---Generaton of the continuous signal: DONE')

    def get_ground_truth_continuous_signal(self):
        """get the grounf truth cont. signal by combining all the class signals"""
        print('\n---Generation of the GT continuous signal: STARTED')
        dict_indeces = {}
        for serie in self.class_dict.keys():
            if serie == 'default_class_value' or serie == 'invalid_class_value':
                continue

            if self.class_dict[serie]['type'] == 'default':
                time_index = self.class_dict[serie]['signal'].index
                n_rows = len(time_index)
                continue
            if not serie == 'period': 
                dict_indeces[serie] = np.where(self.class_dict[serie]['signal'] == int(self.class_dict[serie]['value']))[0]
            else:
                invalid_indeces = np.where(self.class_dict[serie]['signal'] == int(self.invalid_class_value))[0]
                continue
        if len(dict_indeces.keys()) > 1:
            intersect_indeces = np.array([])
            comparison_list = combinations(dict_indeces.keys(), 2)
            for s1, s2 in comparison_list:
                intersect_indeces = np.append(intersect_indeces, np.intersect1d(dict_indeces[s1],
                                                                                dict_indeces[s2])).astype(int)
            intersect_indeces = np.unique(intersect_indeces)
            invalid_indeces = np.append(invalid_indeces, intersect_indeces)
        label_signal = np.array([self.default_value]*n_rows)
        for serie, positions in dict_indeces.items():
            value = self.class_dict[serie]['value']
            for pos in positions:
                label_signal[pos] = value
        for invalid_pos in invalid_indeces:
            label_signal[invalid_pos] = self.invalid_class_value

        self.gt_series = pd.Series(label_signal, index=time_index)
        print('\n---Generation of the GT continuous signal: DONE')

    def perform_event_labeling(self):
        """
            (v1 - using inclusion criterion)
            it checks if an event corresponds to a class or not. In case the
            event is included in an invalid period it will be assigned as invalid 
        """
        print("\n---Checking for anomalous period fully/partially included in our events v2: STARTED")
        if self.guardperiod == '':
            self.guardperiod = pd.to_timedelta(60, 's')
        for event_start in self.df_events_index:
            event_end = event_start + pd.to_timedelta(self.event_period, "m")
            event_labels = []
            event_msg = []
            event_start = pd.to_datetime(event_start).tz_convert(self.time_zone)
            event_end = pd.to_datetime(event_end).tz_convert(self.time_zone)
            
            for serie in self.class_dict.keys():
                if serie == 'default_class_value' or serie == 'invalid_class_value':
                    continue
                if self.class_dict[serie]['type'] == 'default' or serie == 'period':
                    continue
                for a in self.class_dict[serie]['periods']:
                    if event_start < a[0] and event_end > a[1]:
                        event_labels.append(1)
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" is fully included in the event: " + \
                              str(event_start)+" - "+str(event_end)
                        event_msg.append(msg)
                    elif event_start < a[0] and event_end+self.guardperiod > a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The " + serie + " period " + str(a[0]) + " - " + str(a[1]) + \
                              " is partially included (using a safety band on the right) in " \
                              + str(event_start) + " - " + str(event_end)
                        event_msg.append(msg)
                    elif event_start-self.guardperiod < a[0] and event_end > a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The " + serie + " period "+str(a[0]) + " - " + str(a[1]) + \
                              " is partially included (using a safety band on the left) in " \
                              + str(event_start) + " - " + str(event_end)
                        event_msg.append(msg)
                    elif event_start > a[0] and event_end < a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0])+" - "+str(a[1])+" includes the event " \
                              + str(event_start) + " - " + str(event_end)
                        event_msg.append(msg)
                    elif event_start > a[0]-self.guardperiod and event_end < a[1]:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The "+serie+" period "+str(a[0]) + " - " + str(a[1]) + \
                              " partially includes (using a safety band on the left) the event " \
                              + str(event_start) + " - " + str(event_end)
                        event_msg.append(msg)
                    elif event_start > a[0] and event_end < a[1]+self.guardperiod:
                        event_labels.append(self.class_dict[serie]['value'])
                        msg = "The " + serie + " period " + str(a[0]) + " - " + str(a[1]) + \
                              " partially includes (using a safety band on the right) the event " \
                              + str(event_start) + " - " + str(event_end)
                        event_msg.append(msg)

            if len(set(event_labels)) > 1:  # when we have different label for the same event
                print("The event is ambiguous. I won't assign neither 1 or 0. I will assign -1")
                self.df_events_label.loc[event_start, 'label'] = self.invalid_class_value
            elif len(set(event_labels)) == 0:
                invalid = False
                for inv in self.invalid_periods:
                    latest_start = max(event_start, inv[0])
                    earliest_end = min(event_end, inv[1])
                    if latest_start <= earliest_end:
                        invalid = True
                        break
                if invalid == False:
                    event_labels.append(self.default_value)
                    self.df_events_label.loc[event_start, 'label'] = self.default_value
                else:
                    event_labels.append(self.invalid_class_value)
                    self.df_events_label.loc[event_start, 'label'] = self.invalid_class_value
            else:
                self.df_events_label.loc[event_start, 'label'] = event_labels[0]

        self.df_events_label = self.df_events_label.sort_index()
        self.series_events_label = pd.Series(self.df_events_label['label'].astype(int), index=self.df_events_label.index)
        print("---Checking for anomalous period fully/partially included in our events: DONE")

    def find_majority(self, votes):
        """
            helper function for the perform_event_labeling_from_gt_signal method. 
            It looks for the most common value in an event period
        """
        vote_count = Counter(votes)
        top = vote_count.most_common(1)
        if len(top) > 1:
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
        gt_events, event_minimum_samples = data_sampler.sample_dataevents(self.gt_series,
                                                                          event_minimum_period=self.event_period)
        events_dict = {}
        n_events = len(gt_events)
        progress_list = [round(n_events*20/100), round(n_events*40/100), round(n_events*50/100),
                         round(n_events*70/100), round(n_events*90/100), n_events-1]
        for j, event in enumerate(gt_events):
            if j in progress_list:
                print('Event modelling: event dataframe n', j, 'out of ', n_events)
            label_list = []
            event_timespan = []
            for i in event.index:
                label_list.append(event[i])
            event_timespan.append((event.index[0]))
            event_timespan.append((event.index[len(event)-1]))
            value = self.find_majority(label_list)
            events_dict[event_timespan[0]] = value

        self.gt_series_events_label = pd.Series(events_dict, index=events_dict.keys())
        print("---Performing Event Labeling from GT signal: DONE")

 
def main():
    pass  


if __name__ == '__main__':
    # main()
    pass

