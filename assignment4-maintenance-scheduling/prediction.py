import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

"""
SCORING FUNCTION
"""

def mean_penalty_scoring(y_true, y_pred):
    # scoring scheme where we penalize more 
    # when the predicted RUL is later than the actual RUL
    deltas = y_pred - y_true
    result = []
    for delta in deltas:
        result.append((math.exp(-delta/13)-1) if delta < 0 else (math.exp(delta/10)-1))
    return np.mean(result)

def rmse(y_true, y_pred):
    # root mean square error
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

"""
RUL Prediction Problem Wrapper
"""

class RULPrediction(object):
    def __init__(self, n_backtrack=3, rul_ceil=None):
        # only take attributes that have variability
        self.sensors = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
        self.settings = ["setting1", "setting2"]
        self.n_backtrack = n_backtrack
        self.rul_ceil = rul_ceil
        self.train_engines_list = None

        self.scaler = None
        self.mdl = None
        self.mdl_class = None
        self.mdl_params = None
        
    def load(self, file_name):
        # load and format data types of the input file
        settings = ['setting1', 'setting2', 'setting3']
        sensors = ['s' + str(s+1) for s in range(21)]

        col_names = ['engine_id','cycle'] + settings + sensors

        data_dtypes = {
            "engine_id" : np.int64,
            "cycle"     : np.int64,
            "s1"        : np.float64,
            "s2"        : np.float64,
            "s3"        : np.float64,
            "s4"        : np.float64,
            "s5"        : np.float64,
            "s6"        : np.float64,
            "s7"        : np.float64,
            "s8"        : np.float64,
            "s9"        : np.float64,
            "s10"       : np.float64,
            "s11"       : np.float64,
            "s12"       : np.float64,
            "s13"       : np.float64,
            "s14"       : np.float64,
            "s15"       : np.float64,
            "s16"       : np.float64,
            "s17"       : np.float64,
            "s18"       : np.float64,
            "s19"       : np.float64,
            "s20"       : np.float64,
            "s21"       : np.float64,
            "setting1"  : np.float64,
            "setting2"  : np.float64
        }
        df = pd.read_csv(file_name, delimiter=' ', index_col=None, header=None, names=col_names, dtype=data_dtypes)
        df = df[["engine_id", "cycle"] + self.sensors + self.settings]
        return df      

    def add_rul(self, df):
        # add RUL column as target variable based on the cycle for training set
        max_cycle_per_engine = df.groupby(['engine_id']).agg({'cycle':'max'}).rename(columns={'cycle' : 'max_cycle'})
        df = df.merge(max_cycle_per_engine, how='left', left_on='engine_id', right_index=True)
        ruls = df['max_cycle'] - df['cycle'] + 1
        ruls = ruls.apply(lambda rul: rul if self.rul_ceil is None or rul < self.rul_ceil else self.rul_ceil)
        df['RUL'] = ruls
        return df
    
    def train_test_split(self, df, n_test=20, random_seed=1337):
        # split the training dataset into train and test set
        # pick n_test random engines for test set and the rest belong to training set
        np.random.seed(random_seed)
        self.train_engines_list = list(df['engine_id'].unique())
        test_engines = []
        for i in range(n_test):
            random_engine = np.random.randint(len(self.train_engines_list))
            test_engines.append(self.train_engines_list.pop(random_engine))
            
        train_df = df[df['engine_id'].isin(self.train_engines_list)].reset_index(drop=True)
        test_df = df[df['engine_id'].isin(test_engines)].reset_index(drop=True)
        
        return train_df, test_df
    
    def prepare(self, train_file):
        # produce train-val-test sets for the modeling

        # load train file
        self.data_train = self.load(train_file)

        # add RUL to train file as target variable
        self.data_train = self.add_rul(self.data_train)

        # split to train-val and test
        self.train_val_df, self.test_df = self.train_test_split(self.data_train)
        self.train_df, self.val_df = self.train_test_split(self.train_val_df)
        
        # normalization and time-window features
        # use train set scaler for validation and test
        self.train_df, scaler = self.preprocess(self.train_df)
        self.val_df, _ = self.preprocess(self.val_df, scaler)
        self.test_df, _ = self.preprocess(self.test_df, scaler)
        return True

    def get_all_data_train(self):
        return self.preprocess(self.data_train)

    def prepare_file(self, schedule_file, scaler):
        # load the file
        schedule_df = self.load(schedule_file)

        # normalization and time-window features
        schedule_prep_df, _ = self.preprocess(schedule_df, scaler)

        return schedule_prep_df

    
    def normalize(self, df, scaler=None):
        # min-max normalization algorithm for each sensor and setting columns
        # only train scaler if the dataframe is a training set

        col_names_norm = [s + '_norm' for s in self.settings+self.sensors]
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(-1,1))
            scaler.fit(df[self.settings + self.sensors])
        norm_df = pd.DataFrame(scaler.transform(df[self.settings + self.sensors]), columns=col_names_norm, index=df.index)
        info_attr = ['engine_id', 'cycle'] 
        info_attr += (['RUL'] if 'RUL' in df.columns.values else [])
        result_df = df[info_attr].join(norm_df)
        return result_df, scaler
    
    def add_lag_data(self, df, n_backtrack):
        # add time-window features to the dataframe
        # look back n_backtrack cycle and remove cycles less than n_backtrack

        col_names_norm = [s + '_norm' for s in self.settings+self.sensors]
        col_names_backtrack = self.col_names[:]
        result_df = df.copy()
        temp_df = df.copy()
        for i in range(n_backtrack):
            temp_df['cycle_shifted_backward'] = temp_df['cycle'] + i + 1
            result_df = result_df.merge(
                                    temp_df[['engine_id', 'cycle_shifted_backward'] + col_names_norm],
                                    how = 'left',
                                    left_on = ['engine_id', 'cycle'], 
                                    right_on = ['engine_id', 'cycle_shifted_backward'], 
                                    suffixes=['', '_t-' + str(i + 1)]
                                )
            col_names_backtrack += [s + '_t-' + str(i+1) for s in col_names_norm]
        self.col_names = col_names_backtrack
        return result_df.dropna()
    
    def preprocess(self, df, scaler=None):
        # preprocessing steps of a dataframe: scaling and add time-window features

        result_df, scaler = self.normalize(df, scaler=scaler)
        self.col_names = [s + '_norm' for s in self.settings+self.sensors]
        if self.n_backtrack is not None and self.n_backtrack > 0:
            result_df = self.add_lag_data(result_df, self.n_backtrack)
        return result_df, scaler