#!/usr/bin/python3

"""
    System diagnostics: data manager
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
 

import sys
import os
import argparse
import json
import shelve
import urllib.parse as urlparse
import urllib.request as urlrequest

import numpy as np
import pandas as pd
import dateutil.parser as dateparse

import data_sampler
import data_exceptions


class CustomerNetworkData:

    def __init__(self, customer_name, json_path=''):
        self.customer_name = customer_name
        self.json_path = json_path
        self.networks = []
        self.load_networks_map()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'JSON path: '
        if not self.json_path.find('\\') >= 0:
            print_message += '{0}\\'.format(os.getcwd())
        print_message += '{0}\n'.format(self.json_path)
        print_message += 'Networks map: {0}\n'.format(self.networks)
        return print_message

    def load_networks_map(self):
        if not self.json_path:
            self.json_path = 'diagnostics_map.json'
        diagnostics_map = load_json(self.json_path)
        for customer_data in diagnostics_map['customers']:
            if customer_data['customer_name'] == self.customer_name:
                self.networks = customer_data['networks']
        if not self.networks:
            raise data_exceptions.DataNotFound(data_name=self.customer_name,
                                               source_name='customers')
        return True


class CustomerSourceData(CustomerNetworkData):

    def __init__(self, customer_name, network_name, json_path=''):
        CustomerNetworkData.__init__(self, customer_name, json_path)
        self.network_name = network_name
        self.data_sources = []
        self.load_data_sources()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'Network name: {0}\n'.format(
            self.network_name)
        print_message += 'Data sources: {0}\n'.format(self.data_sources)
        return print_message

    def load_data_sources(self):
        for network in self.networks:
            if network['network_name'] == self.network_name:
                self.data_sources = network['data_sources']
        if not self.data_sources:
            raise data_exceptions.DataNotFound(data_name=self.network_name,
                                               source_name='networks')
        return True


class CustomerDBData(CustomerSourceData):

    def __init__(self, customer_name, network_name, data_source_name,
                 json_path=''):
        CustomerSourceData.__init__(self, customer_name, network_name,
                                    json_path)
        self.data_source_name = data_source_name
        self.data_source_ip_port = ''
        self.databases = []
        self.load_databases()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'Network name: {0}\n'.format(
            self.network_name)
        print_message += 'Data source name: {0}\n'.format(
            self.data_source_name)
        print_message += 'Data source ip port: {0}\n'.format(
            self.data_source_ip_port)
        print_message += 'Databases: {0}\n'.format(self.databases)
        return print_message

    def load_databases(self):
        for data_source in self.data_sources:
            if data_source['data_source_name'] == self.data_source_name:
                self.data_source_ip_port = data_source['data_source_ip_port']
                self.databases = data_source['databases']
        if not self.databases:
            raise data_exceptions.DataNotFound(data_name=self.data_source_name,
                                               source_name='data_sources')
        return True


class CustomerHostData(CustomerDBData):

    def __init__(self, customer_name, network_name, data_source_name,
                 database_name, json_path=''):
        CustomerDBData.__init__(self, customer_name, network_name,
                                data_source_name, json_path)
        self.database_name = database_name
        self.hosts = []
        self.load_hosts()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'Network name: {0}\n'.format(
            self.network_name)
        print_message += 'Data source name: {0}\n'.format(
            self.data_source_name)
        print_message += 'Data source ip port: {0}\n'.format(
            self.data_source_ip_port)
        print_message += 'Database name: {0}\n'.format(
            self.database_name)
        print_message += 'Hosts: {0}\n'.format(self.hosts)
        return print_message

    def load_hosts(self):
        for database in self.databases:
            if database['database_name'] == self.database_name:
                self.hosts = database['hosts']
        if not self.hosts:
            raise data_exceptions.DataNotFound(data_name=self.database_name,
                                               source_name='databases')
        return True


class CustomerMeasureData(CustomerHostData):

    def __init__(self, customer_name, network_name, data_source_name,
                 database_name, host_name, json_path=''):
        CustomerHostData.__init__(self, customer_name, network_name,
                                  data_source_name, database_name, json_path)
        self.host_name = host_name
        self.measurements = []
        self.load_measurements()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'Network name: {0}\n'.format(
            self.network_name)
        print_message += 'Data source name: {0}\n'.format(
            self.data_source_name)
        print_message += 'Data source ip port: {0}\n'.format(
            self.data_source_ip_port)
        print_message += 'Database name: {0}\n'.format(
            self.database_name)
        print_message += 'Host name: {0}\n'.format(
            self.host_name)
        print_message += 'Measurements: {0}\n'.format(self.measurements)
        return print_message

    def load_measurements(self):
        for host in self.hosts:
            if host['host_name'] == self.host_name:
                self.measurements = host['measurements']
        if not self.measurements:
            raise data_exceptions.DataNotFound(data_name=self.host_name,
                                               source_name='hosts')
        return True


class CustomerUnitData(CustomerMeasureData):

    def __init__(self, customer_name, network_name, data_source_name,
                 database_name, host_name, measurement_name, json_path=''):
        CustomerMeasureData.__init__(self, customer_name, network_name,
                                     data_source_name, database_name,
                                     host_name, json_path)
        self.measurement_name = measurement_name
        self.units = []
        self.load_units()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'Network name: {0}\n'.format(
            self.network_name)
        print_message += 'Data source name: {0}\n'.format(
            self.data_source_name)
        print_message += 'Data source ip port: {0}\n'.format(
            self.data_source_ip_port)
        print_message += 'Database name: {0}\n'.format(
            self.database_name)
        print_message += 'Host name: {0}\n'.format(
            self.host_name)
        print_message += 'Measurement name: {0}\n'.format(
            self.measurement_name)
        print_message += 'Units: {0}\n'.format(self.units)
        return print_message

    def load_units(self):
        for measurement in self.measurements:
            if measurement['measurement_name'] == self.measurement_name:
                self.units = measurement['units']
        if not self.units:
            raise data_exceptions.DataNotFound(data_name=self.measurement_name,
                                               source_name='measurements')
        return True


class CustomerHostDiagnostics(CustomerHostData):
    """
    CustomerHostDiagnostics can manage the data collection and wrangling
    of a given customer host in a given period of time as a first step
    of a diagnostics pipeline.

    Details about the database connection, measurements, units and
    filters to query are properly structured in the input JSON file.
    """
    def __init__(self, customer_name, network_name, data_source_name,
                 database_name, host_name, time_from, time_to,
                 time_zone='Europe/Rome', json_path='',
                 event_minimum_period='15m', local_data=False,
                 database_queries=False, preprocess_data=False):
        CustomerHostData.__init__(self, customer_name, network_name,
                                  data_source_name, database_name, json_path)
        self.host_name = host_name
        self.measurements = []
        self.load_measurements()
        self.time_from = time_from
        self.time_to = time_to
        self.time_zone = time_zone
        self.time_from_code = self.time_from.replace('-', '')
        self.time_from_code = self.time_from_code.replace(' ', '')
        self.time_from_code = self.time_from_code.replace(':', '')
        self.time_to_code = self.time_to.replace('-', '')
        self.time_to_code = self.time_to_code.replace(' ', '')
        self.time_to_code = self.time_to_code.replace(':', '')
        self.time_zone_code = self.time_zone.replace('/', '')
        self.event_minimum_period = event_minimum_period
        self.lpf_harmonic_amount = 10
        self.database_queries = database_queries
        self.measure_pd_dataframes = []
        self.measure_pd_joined_dataframe = pd.DataFrame()
        self.measure_pd_dataevent_samples = []
        self.measure_pd_dataevent_sample_length = 0
        self.measure_pd_dataevent_frequency_samples = []
        self.measure_pd_dataevent_transposed_samples = []
        self.measure_pd_dataevent_sample_timestamps = []

        self._resampling_period_string = '' #berto
 
        if local_data:
            self.shelve_measurements(load_shelve=True)
        else:
            self.get_measurements()
            if preprocess_data:
                self.preprocess_measurements(self.event_minimum_period)
            self.shelve_measurements()

    def __repr__(self):
        print_message = 'Customer name: {0}\n'.format(self.customer_name)
        print_message += 'Network name: {0}\n'.format(
            self.network_name)
        print_message += 'Data source name: {0}\n'.format(
            self.data_source_name)
        print_message += 'Data source ip port: {0}\n'.format(
            self.data_source_ip_port)
        print_message += 'Database name: {0}\n'.format(
            self.database_name)
        print_message += 'Host name: {0}\n'.format(
            self.host_name)
        print_message += 'Measurements: {0}\n'.format(self.measurements)
        print_message += '\n'
        for measure_pd_dataframe in self.measure_pd_dataframes:
            print_message += '{0}\n'.format(measure_pd_dataframe)
            print_message += '\n'
        return print_message

    def load_measurements(self):
        for host in self.hosts:
            if host['host_name'] == self.host_name:
                self.measurements = host['measurements']
        if not self.measurements:
            raise data_exceptions.DataNotFound(data_name=self.host_name,
                                               source_name='hosts')
        return True

    def get_measurements(self):
        for measurement in self.measurements:
            measurement_name = measurement['measurement_name']
            unit_names = measurement['units']
            measurement_filters = [[]]

            if 'filters' in measurement:
                unit_filter_maps = measurement['filters']
                unit_filters = UnitFilters(unit_filter_maps)
                measurement_filters = unit_filters.lists

            measurement_unit_filter_names = []
            for unit_name in unit_names:
                for measurement_filter in measurement_filters:
                    if self.data_source_name == 'influx':
                        source_data = get_influx_data(
                            self.data_source_ip_port,
                            self.database_name,
                            self.host_name,
                            measurement_name,
                            unit_name,
                            self.time_from, self.time_to,
                            measurement_filter,
                            self.time_zone,
                            self.database_queries)
                    else:
                        raise data_exceptions.DataSourceUnknown(
                            self.data_source_name)

                    filter_names = []
                    for filter_rule in measurement_filter:
                        filter_name = filter_rule.rsplit(' = ', maxsplit=1)[1]
                        filter_name = filter_name.strip("'")
                        filter_names.append(filter_name)
                    measurement_unit_filter_name = '_'.join(
                        [measurement_name, unit_name] +
                        filter_names)
                    measurement_unit_filter_names.append(
                        measurement_unit_filter_name)

                    source_np_data = np.array(source_data)
                    source_np_feature_samples = source_np_data.shape[0]
                    if source_np_feature_samples != 0:
                        source_pd_date = pd.to_datetime(source_np_data[:, 0],
                                                        utc=True)
                        source_np_values = source_np_data[:, 1:]
                        source_pd_data = pd.DataFrame(
                            source_np_values, dtype='float64',
                            columns=[measurement_unit_filter_name],
                            index=source_pd_date)
                    else:
                        raise data_exceptions.TimeSeriesMissing(
                            measurement_name,
                            unit_name,
                            "' '".join(filter_names))
                        # influx_pd_data = pd.DataFrame(
                        #     [], dtype='float64',
                        #     columns=[measurement_unit_filter_name],
                        #     index=[])
                    self.measure_pd_dataframes.append(source_pd_data)
        return True

    def shelve_measurements(self, load_shelve=False):
        shelve_filename = ''
        shelve_filename += '{0}_'.format(self.customer_name)
        shelve_filename += '{0}_'.format(self.host_name)
        shelve_filename += '{0}_'.format(self.time_from_code)
        shelve_filename += '{0}_'.format(self.time_to_code)
        shelve_filename += '{0}'.format(self.time_zone_code)
        shelve_message = ''
        shelve_message += '{0} '.format(shelve_filename)
        if load_shelve:
            if os.path.isfile('./{0}.dat'.format(shelve_filename)):
                shelve_file = shelve.open(shelve_filename)
                self.measure_pd_dataframes = shelve_file[
                    'measure_pd_dataframes']
                self.measure_pd_joined_dataframe = \
                    shelve_file['measure_pd_joined_dataframe']
                self.measure_pd_dataevent_samples = \
                    shelve_file['measure_pd_dataevent_samples']
                self.measure_pd_dataevent_sample_length = \
                    shelve_file['measure_pd_dataevent_sample_length']
                self.measure_pd_dataevent_frequency_samples = \
                    shelve_file['measure_pd_dataevent_frequency_samples']
                self.measure_pd_dataevent_transposed_samples = \
                    shelve_file['measure_pd_dataevent_transposed_samples']
                self.measure_pd_dataevent_sample_timestamps = \
                    shelve_file['measure_pd_dataevent_sample_timestamps']
                self._resampling_period_string = shelve_file['_resampling_period_string']  #berto
                shelve_file.close()
                shelve_message += 'has been LOADED from the shelve file.'
            else:
                shelve_message += 'has NOT been found.'
        else:
            shelve_file = shelve.open(shelve_filename)
            shelve_file['customer_name'] = self.customer_name
            shelve_file['network_name'] = self.network_name
            shelve_file['data_source_name'] = self.data_source_name
            shelve_file['data_source_ip_port'] = self.data_source_ip_port
            shelve_file['database_name'] = self.database_name
            shelve_file['host_name'] = self.host_name
            shelve_file['time_from'] = self.time_from
            shelve_file['time_to'] = self.time_to
            shelve_file['time_zone_code'] = self.time_zone_code
            shelve_file['measure_pd_dataframes'] = self.measure_pd_dataframes
            shelve_file['measure_pd_joined_dataframe'] = \
                self.measure_pd_joined_dataframe
            shelve_file['measure_pd_dataevent_samples'] = \
                self.measure_pd_dataevent_samples
            shelve_file['measure_pd_dataevent_sample_length'] = \
                self.measure_pd_dataevent_sample_length
            shelve_file['measure_pd_dataevent_frequency_samples'] = \
                self.measure_pd_dataevent_frequency_samples
            shelve_file['measure_pd_dataevent_transposed_samples'] = \
                self.measure_pd_dataevent_transposed_samples
            shelve_file['measure_pd_dataevent_sample_timestamps'] = \
                self.measure_pd_dataevent_sample_timestamps
            shelve_file['_resampling_period_string'] = self._resampling_period_string #berto
            shelve_file.close()
            shelve_message += 'has been SAVED in the shelve file.'
        print(shelve_message)
        return True

    def preprocess_measurements(self, verbose=False):
        if self.measure_pd_dataframes:
            self.measure_pd_dataframes = data_sampler.pad_pd_dataframes(
                self.measure_pd_dataframes, self.time_from, self.time_to,
                self.time_zone)
            if verbose:
                print('Data sampler | pad_pd_dataframes DONE.')

            self.measure_pd_dataframes, self._resampling_period_string = data_sampler.resample_pd_dataframes(
                self.measure_pd_dataframes)
              
            if verbose:
                print('Data sampler | resample_pd_dataframes DONE.')

            self.measure_pd_dataframes = data_sampler.fill_pd_dataframes(
                self.measure_pd_dataframes)
            if verbose:
                print('Data sampler | fill_pd_dataframes DONE.')

            self.measure_pd_dataframes = \
                data_sampler.standardize_pd_dataframes(
                    self.measure_pd_dataframes)
            if verbose:
                print('Data sampler | standardize_pd_dataframes DONE.')

            self.measure_pd_joined_dataframe = data_sampler.join_pd_dataframes(
                self.measure_pd_dataframes)
            if verbose:
                print('Data sampler | join_pd_dataframes DONE.')

            self.measure_pd_dataevent_samples, \
                self.measure_pd_dataevent_sample_length = \
                data_sampler.sample_dataevents(
                    self.measure_pd_joined_dataframe,
                    self.event_minimum_period)
            if verbose:
                print('Data sampler | sample_dataevents DONE.')

            self.measure_pd_dataevent_frequency_samples = \
                data_sampler.filter_low_pass_dataevents(
                    self.measure_pd_dataevent_samples,
                    self.lpf_harmonic_amount)
            if verbose:
                print('Data sampler | filter_low_pass_dataevents DONE.')

            self.measure_pd_dataevent_transposed_samples, \
                self.measure_pd_dataevent_sample_timestamps = \
                data_sampler.transpose_dataevents(
                    self.measure_pd_dataevent_frequency_samples)
            if verbose:
                print('Data sampler | transpose_dataevents DONE.')
            return True
        else:
            return False


class UnitFilters:

    def __init__(self, unit_filter_maps):
        self.maps = unit_filter_maps
        self.lists = self.get_filter_lists(self.maps, [], [])

    def __repr__(self):
        print_message = ''
        for f in self.lists:
            print_message += '{}\n'.format(f)
        return print_message

    def get_filter_lists(self, filter_maps, previous_filters,
                         filter_lists):
        for filter_map in filter_maps:
            actual_filters = self.get_filters(filter_map)
            if previous_filters:
                actual_filters = [actual_filter + previous_filter
                                  for actual_filter in actual_filters
                                  for previous_filter in previous_filters]
            if 'filters' in filter_map:
                self.get_filter_lists(filter_map['filters'], actual_filters,
                                      filter_lists)
            else:
                filter_lists += actual_filters
        return filter_lists

    def get_filters(self, filter_map):
        filters = []
        filter_name = filter_map['filter_name']
        filter_values = filter_map['filter_values']
        for filter_value in filter_values:
            filter_item = ["{} = '{}'".format(filter_name, filter_value)]
            filters.append(filter_item)
        return filters


def load_json(file_path):
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


def get_influx_data(influx_ip_port, database_name, host_name,
                    measurement_name, unit_names, time_from, time_to,
                    unit_filter, time_zone='Europe/Rome',
                    print_influx_query_request=False):
    influx_base_url = 'http://{}/query'.format(influx_ip_port)
    influx_query_list = []
    influx_query_units = 'SELECT '
    if isinstance(unit_names, str):
        unit_names = [unit_names]
    influx_query_units += '"{}"'.format('", "'.join(unit_names))
    influx_query_list.append(influx_query_units)
    influx_query_measurement = 'FROM {}'.format(measurement_name)
    influx_query_list.append(influx_query_measurement)
    influx_query_time = "WHERE time > '{}' ".format(time_from)
    influx_query_time += "AND time < '{}'".format(time_to)
    influx_query_list.append(influx_query_time)
    influx_query_filters = "AND host = '{}'".format(host_name)
    if unit_filter:
        influx_query_filters += ' AND '
        influx_query_filters += ' AND '.join(unit_filter)
    influx_query_list.append(influx_query_filters)
    influx_query_time_order = 'ORDER BY time DESC'
    influx_query_list.append(influx_query_time_order)
    influx_query_time_zone = "tz('{}')".format(time_zone)
    influx_query_list.append(influx_query_time_zone)
    influx_query = ' '.join(influx_query_list)
    if print_influx_query_request:
        print(influx_query)
    influx_query_url = urlparse.urlencode({'q': influx_query,
                                           'db': database_name})
    influx_request = '{0}?{1}'.format(influx_base_url, influx_query_url)
    if print_influx_query_request:
        print(influx_request)
    influx_response = json.load(urlrequest.urlopen(influx_request))
    influx_data = []
    if 'series' in influx_response['results'][0]:
        if 'values' in influx_response['results'][0]['series'][0]:
            influx_data = influx_response['results'][0]['series'][0]['values']
    return influx_data


def set_to_numpy_datetimes(influx_data):
    for influx_data_row in influx_data:
        influx_datetime = dateparse.parse(influx_data_row[0], ignoretz=True)
        influx_datetime_format = '{}'.format(influx_datetime.isoformat(
            timespec='milliseconds'))
        np_datetime = np.datetime64(influx_datetime_format, 'ms')
        influx_data_row[0] = np_datetime
    return influx_data


def set_to_pandas_datetimes(influx_data):
    for influx_data_row in influx_data:
        pd_datetime = pd.to_datetime(influx_data_row[0])
        influx_data_row[0] = pd_datetime
    return influx_data


def main():
    cli_args = sys.argv[1:]
    if cli_args:
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--json_path',
                            help='set the json path of the diagnostics map')
        parser.add_argument('-c', '--customer_name',
                            help='select a customer from where getting '
                                 'influxdb data')
        parser.add_argument('-v', '--verbose_level',
                            help='verbose the check output')
        args = parser.parse_args()
        customer_name = args.customer_name
        json_path = args.json_path if args.json_path else ''
        verbose_level = int(args.verbose_level) if args.verbose_level else 1
        if customer_name:
            pass
    else:
        pass


if __name__ == '__main__':
    main()
