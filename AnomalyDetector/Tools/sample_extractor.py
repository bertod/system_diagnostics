import pandas as pd
from sklearn.decomposition import PCA


class SampleExtractor:
    """
    This module is responsible for sample generation.
    It allows you to apply feature extraction and dimensionality reduction.
    The technique used for the first task consists into extracting statistics from data
    like mean,std and and trend. you can add more modifying the method generate_samples.
    Regarding the dim. reduction, it is implemented via PCA
    """
    def __init__(self, list_events=[]):
        self.list_events = list_events
        self.df_samples = None
        self.df_samples_reduce = None

    def linreg(self, X, Y):
        """
        return a,b in solution to y = ax + b such that root mean square distance
        between trend line and original points is minimized
        """
        N = len(X)
        Sx = Sy = Sxx = Syy = Sxy = 0.0
        for x, y in zip(X, Y):
            Sx = Sx + x
            Sy = Sy + y
            Sxx = Sxx + x*x
            Syy = Syy + y*y
            Sxy = Sxy + x*y
        det = Sxx * N - Sx * Sx
        # extrapolatedtrendline=[a*index + b for index in range(len(x))]
        return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

    def generate_samples(self, feature_engineering=True):
        """
            combine all the events and generate a dataframe with microfeatures
            it can also apply feature engineering (mean and std method)   
        """ 
        if feature_engineering == True:
            # work on all events - shape: n.events x 36features (mean + variance)
            print('\n---Building the Event Dataframe with microfeatures (mean, std method, trend): STARTED')
            events_dict = {}
            n_events = len(self.list_events)
            progress_list = [round(n_events*20/100), round(n_events*40/100), round(n_events*50/100),
                             round(n_events*70/100), round(n_events*90/100), n_events-1]
            for j, event in enumerate(self.list_events):
                if j in progress_list:
                    print('Event modelling: event dataframe n', j, 'out of ', n_events)
                microfeatures_list = []
                event_timespan = []
                count = 0
                # for i in event.iloc[:,0:12].columns:
                for i in event.columns:
                    microfeatures_list.append(event.loc[:, i].mean())
                    microfeatures_list.append(event.loc[:, i].std())
                    # microfeatures_list.append(event.loc[:,i].max()-event.loc[:,i].min())  # dinamica del segnale
                    x = event.loc[:, i]
                    a, b = self.linreg(range(len(x)), x)
                    microfeatures_list.append(a)
                   
                event_timespan.append(str(event.index[0]))
                event_timespan.append(str(event.index[len(event)-1]))
                events_dict[pd.to_datetime(event_timespan[0])] = microfeatures_list
                # events_dict[' - '.join(event_timespan)] = microfeatures_list
            df_samples = pd.DataFrame(events_dict)
            self.df_samples = df_samples.T
            print('---Building the Event Dataframe with microfeatures: DONE')
        else:
            # work on all events - shape: n.events x 16200 (micro)features
            print('\n---Building the Event Dataframe with microfeatures (without feature engineering): STARTED')
            events_dict = {}
            n_events = len(self.list_events)
            progress_list = [round(n_events*20/100), round(n_events*40/100), round(n_events*50/100),
                             round(n_events*70/100), round(n_events*90/100), n_events-1]
            for j, event in enumerate(self.list_events):
                if j in progress_list:
                    print('Event modelling: event dataframe n ', j, 'out of ', n_events)
                microfeatures_list = []
                event_timespan = []
                count = 0
                for i in event.index:
                    # microfeatures_list = microfeatures_list + event.loc[i,:].values.tolist()
                    if count == 0 or count == len(event)-1:
                        event_timespan.append(str(i))
                    count += 1
                events_dict[event_timespan[0]] = microfeatures_list
                # events_dict[' - '.join(event_timespan)] = microfeatures_list
            df_samples = pd.DataFrame(events_dict)
            self.df_samples = df_samples.T
            print('\n---Building the Event Dataframe with microfeatures (without feature engineering): DONE')
        
    def apply_pca(self, n_components=2):
        print('Dimensionality Reduction via PCA: START')
        pca = PCA(n_components=n_components)
        X_reduce = pca.fit_transform(self.df_samples) 
        self.df_samples_reduce = pd.DataFrame(X_reduce, index=self.df_samples.index)
        print('Dimensionality Reduction via PCA: DONE')
