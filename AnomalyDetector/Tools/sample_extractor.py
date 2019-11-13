import pandas as pd

class SampleExtractor():

    def __init__(self, list_events = []):
        self.list_events = list_events
        self.df_samples = None

    def generate_samples(self,feature_engineering = True):
        """
            combine all the events and generate a dataframe with microfeatures
            it can also apply feature engineering (mean and std method)   
        """ 
        if feature_engineering == True:
            ## work on all events - shape: n.events x 36features (mean + variance)
            print('\n---Bulding the Event Dataframe with microfeatures (mean and std method): STARTED')
            events_dict = {}
            n_events = len(self.list_events)
            progress_list = [round(n_events*20/100),round(n_events*40/100),round(n_events*50/100),round(n_events*70/100),round(n_events*90/100),n_events-1]
            for j, event in enumerate(self.list_events):
                if j in progress_list:
                    print('Event modelling: event dataframe n',j, 'out of ',n_events)
                microfeatures_list = []
                event_timespan = []
                count = 0
                for i in event.columns:
                    microfeatures_list.append(event.loc[:,i].mean())
                    microfeatures_list.append(event.loc[:,i].std())
                event_timespan.append(str(event.index[0]))
                event_timespan.append(str(event.index[len(event)-1]))
                events_dict[' - '.join(event_timespan)] = microfeatures_list
            df_samples = pd.DataFrame(events_dict)
            self.df_samples = df_samples.T
            print('---Bulding the Event Dataframe with microfeatures: DONE')
            #return df_samples.T
        else:
            ## work on all events - shape: n.events x 16200 (micro)features
            print('\n---Bulding the Event Dataframe with microfeatures (without feature engineering): STARTED')
            events_dict = {}
            n_events = len(self.list_events)
            progress_list = [round(n_events*20/100),round(n_events*40/100),round(n_events*50/100),round(n_events*70/100),round(n_events*90/100),n_events-1]
            for j, event in enumerate(self.list_events):
                if j in progress_list:
                    print('Event modelling: event dataframe n ',j, 'out of ',n_events)
                microfeatures_list = []
                event_timespan = []
                count = 0
                for i in event.index:
                    #microfeatures_list = microfeatures_list + event.loc[i,:].values.tolist()
                    if count == 0 or count == len(event)-1:
                        event_timespan.append(str(i))
                    count += 1
                events_dict[' - '.join(event_timespan)] = microfeatures_list
            df_samples = pd.DataFrame(events_dict)
            self.df_samples = df_samples.T
            print('\n---Bulding the Event Dataframe with microfeatures (without feature engineering): DONE') 
        
