#!/usr/bin/python3

"""
    System diagnostics: data exceptions
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


class DataNotFound(Exception):

    def __init__(self, data_name='', source_name=''):
        self.data_name = data_name
        self.source_name = source_name

    def __str__(self):
        exception_message = ''
        if self.data_name:
            exception_message += "'{0}'".format(self.data_name)
        if self.source_name:
            exception_message += " from '{0}'".format(self.source_name)
        return exception_message


class DataSourceUnknown(Exception):
    def __init__(self, data_source_name):
        self.data_source_name = data_source_name

    def __str__(self):
        exception_message = 'The data source '
        exception_message += "'{0}' ".format(self.data_source_name)
        exception_message += "is unknown."
        return exception_message


class TimeSeriesMissing(Exception):

    def __init__(self, measurement_name, unit_name, filter_names=''):
        self.measurement_name = measurement_name
        self.unit_name = unit_name
        self.filter_names = filter_names

    def __str__(self):
        exception_message = 'The measurement time series '
        exception_message += "'{0}' ".format(self.measurement_name)
        exception_message += "of unit '{0}' ".format(self.unit_name)
        if self.filter_names:
            exception_message += "filtered by '{0}' ".format(self.filter_names)
        exception_message += "is missing."
        return exception_message
