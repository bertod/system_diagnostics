#!/usr/bin/python3

"""
    System diagnostics: data sampler
    Copyright (C) 2019 Francesco Melchiori
    <https://www.francescomelchiori.com/>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see
    <http://www.gnu.org/licenses/>.
"""


import numpy as np
import pandas as pd

import AnomalyDetector.Tools.signal_processor as signal_processor


def get_pd_dataframe_minimum_sampling_period(pd_dataframe,
                                             sampling_precision='1s'):
    pd_dataframe_sampling_periods = []
    pd_second_delta = pd.to_timedelta(sampling_precision)
    if not pd_dataframe.empty:
        pd_dataframe_range = range(pd_dataframe.shape[0] - 1)
        for current_index in pd_dataframe_range:
            next_index = current_index + 1
            current_timestamp = pd_dataframe.index[current_index]
            next_timestamp = pd_dataframe.index[next_index]
            pd_dataframe_delta = abs(current_timestamp - next_timestamp)
            pd_dataframe_sampling_periods.append(int(
                pd_dataframe_delta/pd_second_delta))
    if pd_dataframe_sampling_periods:
        pd_dataframe_sampling_period = min(pd_dataframe_sampling_periods)
    else:
        pd_dataframe_sampling_period = False
    return pd_dataframe_sampling_period


def get_pd_dataframes_minimum_sampling_period(pd_dataframes,
                                              sampling_precision='1s'):
    pd_dataframes_sampling_periods = []
    for pd_dataframe in pd_dataframes:
        if not pd_dataframe.empty:
            pd_dataframe_sampling_period = \
                get_pd_dataframe_minimum_sampling_period(
                    pd_dataframe, sampling_precision)
            if pd_dataframe_sampling_period:
                pd_dataframes_sampling_periods.append(
                    pd_dataframe_sampling_period)
    if pd_dataframes_sampling_periods:
        pd_dataframes_minimum_sampling_period = min(
            pd_dataframes_sampling_periods)
    else:
        pd_dataframes_minimum_sampling_period = False
    return pd_dataframes_minimum_sampling_period


def get_down_rounded_sampling_period(raw_sampling_period, sampling_unit='s'):
    if sampling_unit == 's':
        hour_max_sampling_period = 3600
        down_rounded_sampling_period = 3600
    elif sampling_unit == 'ms':
        hour_max_sampling_period = 3600000
        down_rounded_sampling_period = 3600000
    else:
        hour_max_sampling_period = 3600
        down_rounded_sampling_period = 3600
    hour_all_sampling_periods = np.arange(1, hour_max_sampling_period+1)
    hour_label_valid_sampling_periods = np.mod(hour_max_sampling_period,
                                               hour_all_sampling_periods) == 0
    hour_valid_sampling_periods = hour_all_sampling_periods[
        hour_label_valid_sampling_periods]
    hour_label_down_rounded_sampling_periods = np.floor_divide(
        hour_valid_sampling_periods, raw_sampling_period+1) == 0
    for label_sampling_period, sampling_period in zip(
            hour_label_down_rounded_sampling_periods,
            hour_valid_sampling_periods):
        if label_sampling_period:
            down_rounded_sampling_period = int(sampling_period)
        else:
            break
    return down_rounded_sampling_period


def get_pd_dataframes_down_rounded_sampling_period(pd_dataframes,
                                                   sampling_precision='1s'):
    sampling_unit = get_sampling_unit(sampling_precision)
    pd_dataframes_down_rounded_sampling_period = False
    if pd_dataframes:
        pd_dataframes_minimum_sampling_period =\
            get_pd_dataframes_minimum_sampling_period(pd_dataframes,
                                                      sampling_precision)
        if pd_dataframes_minimum_sampling_period:
            pd_dataframes_down_rounded_sampling_period = \
                get_down_rounded_sampling_period(
                    pd_dataframes_minimum_sampling_period, sampling_unit)
    return pd_dataframes_down_rounded_sampling_period


def pad_pd_dataframes(pd_dataframes, timestamp_start, timestamp_end,
                      time_zone):
    """
       For each series in each dataframe, pad_pd_dataframes pads an
       entry at the beginning and one at the end respectedly with the
       first and the last available values at the given timestamps.
    """
    pd_timezone_index_start = pd.DatetimeIndex([timestamp_start], tz=time_zone)
    pd_timezone_index_end = pd.DatetimeIndex([timestamp_end], tz=time_zone)
    pd_utc_index_start = pd.to_datetime(pd_timezone_index_start, utc=True)
    pd_utc_index_end = pd.to_datetime(pd_timezone_index_end, utc=True)
    pd_padded_dataframes = []
    if pd_dataframes:
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                padding_start_dict = {}
                for pd_series in pd_dataframe.columns:
                    pd_series_values = pd_dataframe[pd_series]
                    for pd_series_value in pd_series_values:
                        if not pd.isnull(pd_series_value):
                            padding_start_dict['{0}'.format(pd_series)] = \
                                pd.Series(pd_series_value,
                                          index=pd_utc_index_start)
                            break
                padding_end_dict = {}
                for pd_series in pd_dataframe.columns:
                    pd_series_values = pd_dataframe[pd_series]
                    pd_series_size = pd_dataframe[pd_series].size
                    pd_series_back_index = [-pd_series_index
                                            for pd_series_index
                                            in range(1, pd_series_size+1)]
                    pd_series_back_values = pd_series_values[
                        pd_series_back_index]
                    for pd_series_back_value in pd_series_back_values:
                        if not pd.isnull(pd_series_back_value):
                            padding_end_dict['{0}'.format(pd_series)] = \
                                pd.Series(pd_series_back_value,
                                          index=pd_utc_index_end)
                            break
                pd_dataframe_padding_start = pd.DataFrame(padding_start_dict)
                pd_dataframe_padding_end = pd.DataFrame(padding_end_dict)
                pd_padded_dataframe = pd.concat([pd_dataframe,
                                                 pd_dataframe_padding_start,
                                                 pd_dataframe_padding_end])
                pd_padded_dataframes.append(pd_padded_dataframe)
    return pd_padded_dataframes

 
def resample_pd_dataframes(pd_dataframes, sampling_precision='1s'):
    """
       For each series in each dataframe, resample_pd_dataframes
       resamples the time series at the maximum sampling frequency among
       them, so upsampling the others.
    """
    sampling_unit = get_sampling_unit(sampling_precision)
    pd_resampled_dataframes = []
    if pd_dataframes:
        resampling_period = get_pd_dataframes_down_rounded_sampling_period(
            pd_dataframes, sampling_precision)
        resampling_period_string = '{0}{1}'.format(resampling_period,
                                                   sampling_unit)
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                pd_resampled_dataframe = pd_dataframe.resample(
                    resampling_period_string).pad()
                #print("LEN not resampled > ", len(pd_dataframe), "VS LEN resampled > ", len(pd_resampled_dataframe))
                pd_resampled_dataframes.append(pd_resampled_dataframe)
    return (pd_resampled_dataframes,resampling_period_string)

def fill_pd_dataframes(pd_dataframes):
    pd_filled_dataframes = []
    if pd_dataframes:
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                pd_filled_dataframe = pd_dataframe.fillna(
                    method='ffill')
                pd_filled_dataframe = pd_filled_dataframe.fillna(
                    method='bfill')
                pd_filled_dataframes.append(pd_filled_dataframe)
    return pd_filled_dataframes


def join_pd_dataframes(pd_dataframes):
    pd_joined_dataframe = pd.DataFrame()
    if pd_dataframes:
        pd_joined_dataframe = pd_dataframes[0]
        for pd_dataframe in pd_dataframes[1:]:
            if not pd_dataframe.empty:
                pd_joined_dataframe = pd_joined_dataframe.join(pd_dataframe)
    return pd_joined_dataframe


def standardize_pd_dataframes(pd_dataframes):
    pd_standard_dataframes = []
    if pd_dataframes:
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                pd_averaged_dataframe = pd_dataframe - pd_dataframe.mean()
                if pd_dataframe.std()[0] != 0:
                    pd_standard_dataframe = pd_averaged_dataframe /\
                                            pd_dataframe.std()
                else:
                    pd_standard_dataframe = pd_averaged_dataframe
                pd_standard_dataframes.append(pd_standard_dataframe)
    return pd_standard_dataframes


def sample_dataevents(pd_dataframe, event_minimum_period='10m'):
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

    # print('event_maximum_sampling_period: {}'.format(
    #     event_maximum_sampling_period))
    # print('event_minimum_samples: {}'.format(
    #     event_minimum_samples))
    # print('sampled_event_amount: {}'.format(
    #     sampled_event_amount))

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


def filter_low_pass_dataevents(pd_dataevents,
                               lpf_harmonic_amount=10,
                               direct_signal=False):
    pd_dataevents_lpf = []
    for pd_dataevent in pd_dataevents:
        pd_dataframe_lpf = signal_processor.filter_low_pass_pd_dataframe(
            pd_dataevent,
            lpf_harmonic_amount=lpf_harmonic_amount,
            direct_signal=direct_signal)
        pd_dataevents_lpf.append(pd_dataframe_lpf)
    return pd_dataevents_lpf


def transpose_dataevents(pd_dataevents):
    event_feature_amount = pd_dataevents[0].columns.size
    event_feature_range = range(1, event_feature_amount)
    transpose_events = []
    event_timestamps = []
    for pd_dataevent in pd_dataevents:
        transpose_event = pd_dataevent.T.iloc[0]
        event_timestamp = pd_dataevent.index[0]
        for event_feature_serial_number in event_feature_range:
            event_feature = pd_dataevent.T.iloc[event_feature_serial_number]
            transpose_event = pd.concat([transpose_event, event_feature],
                                        ignore_index=True)
        transpose_events.append(transpose_event)
        event_timestamps.append(event_timestamp)
    return transpose_events, event_timestamps


def get_sampling_unit(sampling_precision):
    available_sampling_units = ['m', 's', 'ms']
    available_sampling_units_string = ''.join(available_sampling_units)
    sampling_precision_value = sampling_precision.strip(
        available_sampling_units_string)
    numbers_string = ''.join([str(n) for n in range(10)])
    sampling_unit = sampling_precision.strip(numbers_string)
    if not sampling_precision_value.isdecimal():
        sampling_unit = False
    if sampling_unit not in available_sampling_units:
        sampling_unit = False
    return sampling_unit


def main():
    # print(get_down_rounded_sampling_period(7))
    # print(get_sampling_unit('1321s'))

    timestamp_start_01 = '2019-01-29 08:07:36.910000'
    timestamp_start_02 = '2019-01-29 08:17:36.910000'
    timestamp_start_03 = '2019-01-29 08:03:36.910000'
    timestamp_start_04 = '2019-01-29 08:13:36.910000'
    resampling_timestamp_start = '2019-01-29 08:00:00'
    resampling_timestamp_end = '2019-01-29 09:00:00'
    time_zone = 'Europe/Rome'

    data_test_01 = np.random.normal(0, 1, 10)
    timezone_index_test_01 = pd.date_range(timestamp_start_01, periods=10,
                                           freq='5T', tz=time_zone)
    utc_index_test_01 = pd.to_datetime(timezone_index_test_01, utc=True)
    pd_series_test_01 = pd.Series(data_test_01, index=utc_index_test_01)

    print('data test 01: ',data_test_01)
    print('\ntimezone_index_test_01: ',timezone_index_test_01)
    print('\nutc_index_test_01: ',utc_index_test_01)
    print('\npd_series_test_01: ',pd_series_test_01)


    data_test_02 = np.random.normal(0, 1, 6)
    timezone_index_test_02 = pd.date_range(timestamp_start_02, periods=6,
                                           freq='5T', tz=time_zone)
    utc_index_test_02 = pd.to_datetime(timezone_index_test_02, utc=True)
    pd_series_test_02 = pd.Series(data_test_02, index=utc_index_test_02)
    pd_dataframe_test_01 = pd.DataFrame(
        {'pd_series_test_01': pd_series_test_01,
         'pd_series_test_02': pd_series_test_02})

    data_test_03 = np.random.normal(0, 1, 10)
    timezone_index_test_03 = pd.date_range(timestamp_start_03, periods=10,
                                           freq='5T', tz=time_zone)
    utc_index_test_03 = pd.to_datetime(timezone_index_test_03, utc=True)
    pd_series_test_03 = pd.Series(data_test_03, index=utc_index_test_03)
    data_test_04 = np.random.normal(0, 1, 6)
    timezone_index_test_04 = pd.date_range(timestamp_start_04, periods=6,
                                           freq='5T', tz=time_zone)
    utc_index_test_04 = pd.to_datetime(timezone_index_test_04, utc=True)
    pd_series_test_04 = pd.Series(data_test_04, index=utc_index_test_04)
    pd_dataframe_test_02 = pd.DataFrame(
        {'pd_series_test_03': pd_series_test_03,
         'pd_series_test_04': pd_series_test_04})

    pd_dataframes = [pd_dataframe_test_01, pd_dataframe_test_02]

    pd_dataframes = pad_pd_dataframes(pd_dataframes,
                                      resampling_timestamp_start,
                                      resampling_timestamp_end,
                                      time_zone)
    pd_dataframes = resample_pd_dataframes(pd_dataframes)
    pd_dataframes = fill_pd_dataframes(pd_dataframes)
    pd_dataframes = standardize_pd_dataframes(pd_dataframes)
    pd_joineddataframe = join_pd_dataframes(pd_dataframes)

    event_minimum_period = '30m'
    pd_dataevent_samples, pd_dataevent_sample_length = \
        sample_dataevents(pd_joineddataframe, event_minimum_period)
    pd_dataevent_transposed_samples, pd_dataevent_sample_timestamps = \
        transpose_dataevents(pd_dataevent_samples)
    print('pd_joineddataframe:')
    #print(pd_joineddataframe)
    print('')
    print('pd_dataevent_samples:')
    #print(pd_dataevent_samples)
    print('')
    # print('pd_dataevent_sample_length:')
    # print(pd_dataevent_sample_length)
    # print('')
    # print('pd_dataevent_transposed_samples:')
    # print(pd_dataevent_transposed_samples)
    # print('')
    # print('pd_dataevent_sample_timestamps:')
    # print(pd_dataevent_sample_timestamps)
    # print('')


if __name__ == '__main__':
    main()
