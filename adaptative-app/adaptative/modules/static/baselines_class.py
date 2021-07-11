import numpy as np


class Mean_baseline:
    """
    This is a basic baseline where the model is trained taking into account
    the mean of the data and predicts the afluence for some hour based on that.
    Arguments:
        - data
        - metadata: dictionary which saves:
                - what granularity I want to use to train the model, it can only
                    contain hour, day_of_week, month. 
    """
    def __init__(self, metadata):
        # self.data = data
        self.metadata_granularity = metadata['granularity']

    def fit(self, data):
        if self.metadata_granularity:
            # hour of the day
            if 'hour' in self.metadata_granularity:
                # Mean number of people who go to the ED per day each hour
                data['hour'] = data['Date'].dt.hour
                days = (data.iloc[len(data)-1] - data.iloc[0])[0].days
                hour = data['hour'].value_counts() / days
                hour = [hour[i] for i in range(24)]
            else:
                hour = np.ones(24)
            
            # day of the week
            if 'day_of_week' in self.metadata_granularity:
                # Mean number of people who go to the ED per week day
                data['day_of_week'] = data['Date'].dt.dayofweek
                week = data['day_of_week'].value_counts()
                week = [week[i] for i in range(7)]
            else:
                week = np.ones(7)
            
            # month
            if 'month' in self.metadata_granularity:
                # Mean number of people who go to the ED per day each month
                data['month'] = data['Date'].dt.month
                month = data['month'].value_counts()
                # month is codified from 1 to 12, that's why we add 1
                month = [month[i+1] for i in range(12)]
            else:
                month = np.ones(12)

        # We want to scale the hour per day and month so we scale week and month 
        # by dividing them by their mean 
        week = week / np.mean(week)
        month = month / np.mean(month)
        
        self.hour = hour
        self.dof = week
        self.month = month

        return [self.hour, self.dof, self.month]

    def predict(self, unlabeled):
        '''
        Returns the mean number of people who assist
        to the ED department in a specific hour sclaed by
        day and month
        '''
        res = []
    
        for i in range(len(unlabeled)):
            x = unlabeled.iloc[i][0]
            h = x.hour
            d = x.dayofweek
            # substract 1 month from 1 to 12
            m = x.month - 1
            res.append(self.hour[h] * self.dof[d] * self.month[m])

        return res
