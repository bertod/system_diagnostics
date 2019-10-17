#!/usr/bin/python3

"""
    System diagnostics: data labeler
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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import data_viewer
import data_sampler


def cluster_pd_series(pd_series, cluster_amount=2):
    kmeans = KMeans(n_clusters=cluster_amount)
    kmeans.fit(pd_series)
    pd_series_cluster_labels = kmeans.predict(pd_series)
    pd_series_cluster_centers = kmeans.cluster_centers_
    pd_series_closest_cluster_center_indexes,\
        pd_series_closest_cluster_center_distances\
        = pairwise_distances_argmin_min(
            pd_series_cluster_centers, pd_series)
    return pd_series_cluster_labels,\
        pd_series_cluster_centers,\
        list(pd_series_closest_cluster_center_indexes)


def main():
    timestamp_start = '2019-02-04 00:00:00'
    time_zone = 'Europe/Rome'
    sampling_period = '1S'
    event_minimum_period = '1M'
    series_amount = 10
    sampling_amount = 600
    label_amount = 3  # cluster_amount = 5

    anomaly_start = int(sampling_amount/2)
    anomaly_amount = int(sampling_amount/4)
    anomaly_amplitude = 10
    anomaly_pulse = np.zeros(sampling_amount)
    anomaly_pulse[anomaly_start:anomaly_start+anomaly_amount] += \
        anomaly_amplitude

    pd_series_dictionary_test = {}
    series_counter = range(1, series_amount+1)
    for series_number in series_counter:
        data_test = np.random.normal(0, 1, sampling_amount)
        data_test[:] += anomaly_pulse
        timezone_index_test = pd.date_range(timestamp_start,
                                            periods=sampling_amount,
                                            freq=sampling_period,
                                            tz=time_zone)
        utc_index_test = pd.to_datetime(timezone_index_test, utc=True)
        pd_series_test = pd.Series(data_test, index=utc_index_test)
        pd_series_dictionary_name = 'pd_series_test_' + str(series_number)
        pd_series_dictionary_test[pd_series_dictionary_name] = pd_series_test
    pd_dataframe_test = pd.DataFrame(pd_series_dictionary_test)
    # data_viewer.view_pd_dataframe(pd_dataframe_test)

    pd_dataevent_samples, pd_dataevent_sample_length = \
        data_sampler.sample_dataevents(pd_dataframe_test,
                                       event_minimum_period)
    # pd_dataevent_anomaly_sample_start = int(pd_dataevent_samples.__len__()/2)
    # data_viewer.view_pd_dataframe(
    #     pd_dataevent_samples[pd_dataevent_anomaly_sample_start])

    pd_dataevent_transposed_samples, pd_dataevent_sample_timestamps = \
        data_sampler.transpose_dataevents(pd_dataevent_samples)
    # plt.plot(
    #     pd_dataevent_transposed_samples[pd_dataevent_anomaly_sample_start])
    # plt.show()
    # data_viewer.scatter_pd_series_2d(pd_dataevent_transposed_samples)

    pd_series_cluster_labels,\
        pd_series_cluster_centers,\
        pd_series_closest_cluster_center_indexes = cluster_pd_series(
            pd_series=pd_dataevent_transposed_samples,
            cluster_amount=label_amount)
    # data_viewer.scatter_pd_series_2d(pd_dataevent_transposed_samples,
    #                                  pd_series_cluster_labels,
    #                                  pd_series_cluster_centers,
    #                                  pd_series_closest_cluster_center_indexes)


if __name__ == '__main__':
    main()
