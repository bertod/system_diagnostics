#!/usr/bin/python3

"""
    System diagnostics: signal processor
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
from numpy.random import default_rng
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.signal import spectrogram, welch
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

plt.style.use('seaborn-dark')
register_matplotlib_converters()


def filter_low_pass(pd_series,
                    lpf_harmonic_amount=10,
                    lpf_cutoff_frequency=0.1):
    sampling_period_s = 1
    pd_series_sampling_unit = pd_series.index.freq.name
    if pd_series_sampling_unit == 'S':
        sampling_period_s = pd_series.index.freq.n
    measures_time = pd_series.values
    sampling_points = measures_time.size
    measures_freq = fft(measures_time)
    measures_power = np.abs(measures_freq)
    measures_frequencies = fftfreq(sampling_points, d=sampling_period_s)
    measures_phases = np.angle(fftshift(measures_freq))
    if lpf_harmonic_amount:
        lpf_cutoff_frequency = measures_frequencies[lpf_harmonic_amount]
    passed_frequencies_mask = np.abs(measures_frequencies) <=\
        lpf_cutoff_frequency
    measures_freq_lpf = measures_freq.copy()
    cutoff_frequencies_mask = np.invert(passed_frequencies_mask)
    measures_freq_lpf[cutoff_frequencies_mask] = 0
    measures_lpf = ifft(measures_freq_lpf)
    return measures_frequencies,\
        measures_power, \
        measures_phases,\
        passed_frequencies_mask,\
        measures_lpf


def filter_low_pass_pd_series(pd_series,
                              lpf_harmonic_amount=10,
                              direct_signal=False):
    measures_frequencies,\
        measures_power,\
        measures_phases,\
        passed_frequencies_mask,\
        measures_lpf = filter_low_pass(
            pd_series,
            lpf_harmonic_amount=lpf_harmonic_amount)
    if not direct_signal:
        passed_frequencies_mask[0] = False
    passed_half_frequencies_mask_size = int(passed_frequencies_mask.size/2)
    passed_half_frequencies_mask = passed_frequencies_mask[:]
    passed_half_frequencies_mask[passed_half_frequencies_mask_size:] = False
    passed_half_frequencies = \
        measures_frequencies[passed_half_frequencies_mask]
    half_measures_power = measures_power[passed_half_frequencies_mask]
    pd_series_lpf = pd.Series(half_measures_power,
                              index=passed_half_frequencies)
    return pd_series_lpf


def filter_low_pass_pd_dataframe(pd_dataframe,
                                 lpf_harmonic_amount=10,
                                 direct_signal=False):
    pd_dataframe_lpf = {}
    pd_series_names = pd_dataframe.columns
    for pd_series_name in pd_series_names:
        pd_series = pd_dataframe[pd_series_name]
        pd_series_lpf = filter_low_pass_pd_series(
            pd_series,
            lpf_harmonic_amount=lpf_harmonic_amount,
            direct_signal=direct_signal)
        pd_dataframe_lpf[pd_series_name] = pd_series_lpf
    pd_dataframe_lpf = pd.DataFrame(pd_dataframe_lpf)
    return pd_dataframe_lpf


def plot_signal_filter(pd_series,
                       lpf_harmonic_amount=10,
                       lpf_cutoff_frequency=0.1,
                       show_direct_signal=False,
                       show_phase_signal=False):
    sampling_period_s = 1
    pd_series_sampling_unit = pd_series.index.freq.name
    if pd_series_sampling_unit == 'S':
        sampling_period_s = pd_series.index.freq.n
    sampling_points = pd_series.values.size
    processing_period_s = (sampling_points - 1) * sampling_period_s
    chart_amount = 3
    if show_phase_signal:
        chart_amount = 4
    fig, ax = plt.subplots(chart_amount)
    sampling_times = np.linspace(0, processing_period_s, sampling_points)
    for sampling_time in sampling_times:
        ax[0].axvline(sampling_time, c='black', alpha=0.02)
    ax[0].set_xlabel('[s]')
    ax[0].scatter(sampling_times, pd_series.values,
                  marker='o', c='green', alpha=0.3)
    measures_frequencies,\
        measures_power,\
        measures_phases,\
        passed_frequencies_mask,\
        measures_lpf = filter_low_pass(
            pd_series,
            lpf_harmonic_amount=lpf_harmonic_amount,
            lpf_cutoff_frequency=lpf_cutoff_frequency)
    cutoff_frequencies_mask = np.invert(passed_frequencies_mask)
    if not show_direct_signal:
        passed_frequencies_mask[0] = False
    for measures_frequency in measures_frequencies:
        ax[1].axvline(measures_frequency, c='black', alpha=0.02)
    ax[1].set_xlabel('[Hz]')
    ax[1].scatter(measures_frequencies[passed_frequencies_mask],
                  measures_power[passed_frequencies_mask],
                  s=10,
                  c='#5dade2')
    ax[1].scatter(measures_frequencies[cutoff_frequencies_mask],
                  measures_power[cutoff_frequencies_mask],
                  s=10,
                  c='red')
    passed_half_frequencies_mask_size = int(passed_frequencies_mask.size/2)
    passed_half_frequencies_mask = passed_frequencies_mask[:]
    passed_half_frequencies_mask[passed_half_frequencies_mask_size:] = False
    passed_half_frequencies_mask_length = sum(passed_half_frequencies_mask)
    passed_half_frequencies = np.linspace(1,
                                          passed_half_frequencies_mask_length,
                                          passed_half_frequencies_mask_length)
    half_measures_power = measures_power[passed_half_frequencies_mask]
    for passed_half_frequency in passed_half_frequencies:
        ax[2].axvline(passed_half_frequency, c='black', alpha=0.02)
    ax[2].scatter(passed_half_frequencies,
                  half_measures_power,
                  s=10,
                  c='#5dade2')
    ax[2].set_xlabel('[feat]')
    if show_phase_signal:
        for measures_frequency in measures_frequencies:
            ax[3].axvline(measures_frequency, c='black', alpha=0.02)
        ax[3].set_xlabel('[rad]')
        ax[3].scatter(measures_frequencies[passed_frequencies_mask],
                      measures_phases[passed_frequencies_mask],
                      s=10,
                      c='#5dade2')
        ax[3].scatter(measures_frequencies[cutoff_frequencies_mask],
                      measures_phases[cutoff_frequencies_mask],
                      s=10,
                      c='red')
    ax[0].plot(sampling_times, measures_lpf, c='#5dade2')
    plt.show()
    return True


def signal_to_fit(x, a, b, c, d):
    return a * np.sin(b * x) + c * np.sin(d * x)


def main():
    timestamp_start = '2019-01-01 00:00:00.000000'
    time_zone = 'Europe/Rome'
    sampling_period_s = 1
    processing_period_s = 60
    sampling_points = int(processing_period_s/sampling_period_s) + 1

    harmonic_base_period = (2 * np.pi) / processing_period_s
    harmonic_base_amplitude = 4
    harmonic_1_period = harmonic_base_period * 4
    harmonic_1_amplitude = 2
    harmonic_2_period = harmonic_base_period * 16
    harmonic_2_amplitude = 1
    phase = (np.pi / 2) * 0.5

    noise_amplitude = 0.3
    interpolation_fraction = 5
    lpf_harmonic_amount = 10
    lpf_cutoff_frequency = 0.1

    plot_lab = False
    plot_phase = True
    scatter_noisy = True
    scatter_interpolation_linear = False
    scatter_interpolation_cubic = False
    scatter_fitting = False
    plot_spectrogram_psd = False

    sampling_times = np.linspace(0, processing_period_s, sampling_points)
    harmonic_base = np.sin(harmonic_base_period * sampling_times + phase) *\
        harmonic_base_amplitude
    harmonic_1 = np.sin(harmonic_1_period * sampling_times + phase) *\
        harmonic_1_amplitude
    harmonic_2 = np.sin(harmonic_2_period * sampling_times + phase) *\
        harmonic_2_amplitude
    noise = default_rng().standard_normal(size=sampling_points) *\
        noise_amplitude
    measures_clean = harmonic_base + harmonic_1 + harmonic_2
    measures_noisy = measures_clean + noise
    timezone_index = pd.date_range(timestamp_start,
                                   periods=sampling_points,
                                   freq='1S',
                                   tz=time_zone)
    utc_index = pd.to_datetime(timezone_index, utc=True)
    measures_clean = pd.Series(measures_clean, index=utc_index)
    measures_noisy = pd.Series(measures_noisy, index=utc_index)
    pd_dataframe = pd.DataFrame({'measures_clean': measures_clean,
                                 'measures_noisy': measures_noisy})
    pd_series = pd_dataframe['measures_noisy']

    if plot_lab:
        chart_amount = 2
        if plot_phase:
            chart_amount = 3
        fig, ax = plt.subplots(chart_amount)

    if plot_lab and scatter_interpolation_linear:
        linear_interpolation = interp1d(
            sampling_times[::interpolation_fraction],
            measures_noisy[::interpolation_fraction])
        linear_results = linear_interpolation(sampling_times)
        ax[0].scatter(sampling_times[::interpolation_fraction],
                      linear_results[::interpolation_fraction],
                      c='purple')
        ax[0].scatter(sampling_times,
                      linear_results,
                      s=10,
                      c='purple',
                      alpha=0.3)
        ax[0].plot(sampling_times, linear_results, c='purple', alpha=0.3)

    if plot_lab and scatter_interpolation_cubic:
        cubic_interpolation = interp1d(
            sampling_times[::interpolation_fraction],
            measures_noisy[::interpolation_fraction],
            kind='cubic')
        cubic_results = cubic_interpolation(sampling_times)
        ax[0].scatter(sampling_times[::interpolation_fraction],
                      cubic_results[::interpolation_fraction],
                      c='purple')
        ax[0].scatter(sampling_times,
                      cubic_results,
                      s=10,
                      c='purple')
        ax[0].plot(sampling_times, cubic_results, c='purple')

    if plot_lab and scatter_fitting:
        params, params_covariance = curve_fit(
            signal_to_fit,
            sampling_times[::interpolation_fraction],
            measures_noisy[::interpolation_fraction],
            p0=[harmonic_base_amplitude, harmonic_base_period,
                harmonic_1_amplitude, harmonic_1_period])
        fitting_results = signal_to_fit(sampling_times, *params)
        ax[0].scatter(sampling_times[::interpolation_fraction],
                      fitting_results[::interpolation_fraction],
                      c='blue')
        ax[0].scatter(sampling_times,
                      fitting_results,
                      s=10,
                      c='blue')
        ax[0].plot(sampling_times, fitting_results, c='blue')

    if plot_lab:
        measures_frequencies,\
            measures_power,\
            measures_phases,\
            passed_frequencies_mask,\
            measures_lpf = filter_low_pass(
                pd_series,
                lpf_harmonic_amount=lpf_harmonic_amount,
                lpf_cutoff_frequency=lpf_cutoff_frequency)
        cutoff_frequencies_mask = np.invert(passed_frequencies_mask)

        for sampling_time in sampling_times:
            ax[0].axvline(sampling_time, c='black', alpha=0.02)
        ax[0].set_xlabel('[s]')
        ax[0].scatter(pd_dataframe['measures_clean'].index.second,
                      pd_dataframe['measures_clean'].values,
                      marker='o',
                      c='green',
                      alpha=0.3)
        if noise_amplitude > 0:
            if scatter_noisy:
                ax[0].scatter(pd_dataframe['measures_noisy'].index.second,
                              pd_dataframe['measures_noisy'].values,
                              marker='o',
                              c='red')

        for measures_frequency in measures_frequencies:
            ax[1].axvline(measures_frequency, c='black', alpha=0.02)
        ax[1].set_xlabel('[Hz]')
        ax[1].bar(measures_frequencies[passed_frequencies_mask],
                  measures_power[passed_frequencies_mask],
                  width=0.01,
                  color='#5dade2')
        ax[1].scatter(measures_frequencies[cutoff_frequencies_mask],
                      measures_power[cutoff_frequencies_mask],
                      s=10,
                      c='red')

        if plot_phase:
            for measures_frequency in measures_frequencies:
                ax[2].axvline(measures_frequency, c='black', alpha=0.02)
            ax[2].set_xlabel('[rad]')
            ax[2].scatter(measures_frequencies[passed_frequencies_mask],
                          measures_phases[passed_frequencies_mask],
                          s=10,
                          c='#5dade2')
            ax[2].scatter(measures_frequencies[cutoff_frequencies_mask],
                          measures_phases[cutoff_frequencies_mask],
                          s=10,
                          c='red')
            ax[0].plot(sampling_times, measures_lpf, c='#5dade2')

        plt.show()

    if plot_spectrogram_psd:
        fig, ax = plt.subplots(2)
        measures_time = pd_series.values
        sampling_points = measures_time.size
        measures_freq_spg, time_segments, measures_powers_spg = spectrogram(
            measures_time, fs=1, nperseg=sampling_points)
        measures_power_spg = [measures_power[0]
                              for measures_power in measures_powers_spg]
        ax[0].bar(measures_freq_spg, measures_power_spg,
                  width=0.01, color='#5dade2')
        ax[0].set_ylabel('spectrogram')
        ax[0].set_xlabel('[Hz]')
        measures_freq_wlc, measures_powers_wlc = welch(
            measures_time, fs=1, nperseg=sampling_points)
        ax[1].bar(measures_freq_wlc, measures_powers_wlc,
                  width=0.01, color='#5dade2')
        ax[1].set_ylabel('welch psd')
        ax[1].set_xlabel('[Hz]')
        plt.show()

    plot_signal_filter(pd_series,
                       lpf_harmonic_amount=lpf_harmonic_amount,
                       lpf_cutoff_frequency=lpf_cutoff_frequency)


if __name__ == '__main__':
    main()
