#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_manager
import data_viewer
import data_labeler
import signal_processor

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# the main library has a small set of functionality
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)

#%%
def import_data():
    fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
            '2019-03-01 00:00:01', '2019-04-30 00:59:59', 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)

    df = fabbri1905_fabax6pdb.measure_pd_joined_dataframe
    #df.head()
    return df




#%%
def main():
    # Import data from influxdb - calling the private_data_manager
    #df is the joint dataframes of all the measurements (or units) - Event not handled
    df = import_data()

    #list_events = import_data_events()
    #df_events_microfeatures = get_event_microfeatures(list_events,True)


if __name__ == '__main__':
    main()


#%%
df = import_data()
#%%
df.columns
#%%
decomp = decompose(df['Processor_Percent_Privileged_Time__Total'], period=2592000)

#%%
df['Processor_Percent_Privileged_Time__Total']

#%%
decomp.plot()

#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Stationary data
df_temp = df.copy()
df_temp['Processor_Percent_Privileged_Time__Total'] = df_temp['Processor_Percent_Privileged_Time__Total'] - df_temp['Processor_Percent_Privileged_Time__Total'].rolling(12).mean()

fig, ax = plt.subplots(3, figsize=(12,6))
x = (df_temp['Processor_Percent_Privileged_Time__Total'].dropna() - df_temp['Processor_Percent_Privileged_Time__Total'].dropna().shift(12)).dropna()
ax[0] = plot_acf(x, ax=ax[0], lags=25)
ax[1] = plot_pacf(x, ax=ax[1], lags=25)
ax[2].plot(x)

#%%
