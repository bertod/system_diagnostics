#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.exceptions import NotFittedError
from itertools import chain

#%%
class FeatureSelector():

    def __init__(self):
            self.record_collinear = None
           
            # Dictionary to hold removal operations
            self.removal_ops = {}
    

    def print_corr(self,data):
        pass


    def identify_collinear(self, data, correlation_threshold):
       
        self.correlation_threshold = correlation_threshold

        # Calculate the correlations between every column
        corr_matrix = data.corr()
        
        self.corr_matrix = corr_matrix
    
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        
        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop
        for column in to_drop:

            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]    

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                             'corr_feature': corr_features,
                                             'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index = True)

            
        self.record_collinear = record_collinear
        self.removal_ops['collinear'] = to_drop
        
        print('%d features with a correlation greater than %0.2f.\n' % (len(self.removal_ops['collinear']), self.correlation_threshold))
            
            
                    

#%%
fs = FeatureSelector()

#%%
fabbri1905_fabax6pdb = data_manager.CustomerHostDiagnostics(
            'fabbri', 'alyvix', 'influx', 'fabbri1905', 'FABAX6PDB',
            '2019-02-04 00:00:01', '2019-02-10 00:59:59', 'Europe/Rome',
            'diagnostics_map.json', local_data=True,
            database_queries=True, preprocess_data=True)

df_event = fabbri1905_fabax6pdb.measure_pd_dataevent_samples[0]
#%%
fs.identify_collinear(df_event,0.90)

#%%
fs.removal_ops

#%%
