import os
import pandas as pd
import numpy as np
from numpy import array
import datetime
from random import shuffle
from sklearn.model_selection import train_test_split


COLUMNS = ['adset_id', 'label', 'time_index',
           'day_adset_impressions', 'day_adset_click', 'day_nominal_cpi',
           'life_convert_trident', 'life_cost_rmb',
           'day_ecpm', 'day_impressions_trident', 'day_clicks_trident']
Data_Header = ['time_index', 'day_adset_impressions', 'day_adset_click', 'day_nominal_cpi',
               'life_convert_trident', 'life_cost_rmb',
               'day_ecpm', 'day_impressions_trident', 'day_clicks_trident',
               'label']


def read_raw(path='data/formmated_daily_data.csv', stop=False, stop_i=500):
    raw_data = pd.read_csv(path)
    cleaned_data = clean_columns(raw_data)

    x_list = list()
    y_list = list()
    adset_count = 0
    for adset_id in cleaned_data.adset_id.unique():
        adset_data = cleaned_data[cleaned_data['adset_id'] == adset_id]
        time_indexes = adset_data['time_index'].tolist()
        adset_data = adset_data.drop(columns=['adset_id'])
        for i in range(max(time_indexes) - min(time_indexes)):
            if int(i + 1) not in time_indexes:
                adset_data = adset_data.append(pd.DataFrame([[int(i + 1), 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=Data_Header))
        adset_data = adset_data.sort_values(by=['time_index'])
        adset_np = adset_data[Data_Header[1:]].to_numpy()
        cur_x, cur_y = split_sequences(adset_np, 5)
        x_list.extend(cur_x)
        y_list.extend(cur_y)

        adset_count += 1
        # print(str(adset_count) + '  ' + str(adset_id))

        if stop and adset_count >= stop_i:
            break

    return np.asarray(x_list), y_list


# aplit adset data into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[i, -1]
        X.append(seq_x)
        y.append(seq_y)
    return X, y


def clean_columns(pd, cols=COLUMNS, start_date='2021-03-04'):
    d0 = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    pd['time_index'] = pd.event_date.apply(lambda dt:
                                           (datetime.datetime.strptime(dt, "%Y-%m-%d").date() - d0).days)
    return pd[cols]


def shuffle_list(*ls):
    l =list(zip(*ls))
    shuffle(l)
    return zip(*l)


def split_one_zero(x_list, y_list):
    x_zeros = []
    x_ones = []
    for x, y in zip(x_list, y_list):
        if y == 0:
            x_zeros.append(x)
        else:
            x_ones.append(x)
    return x_ones, x_zeros


def return_train_data():
    x_list, y_list = shuffle_list(read_raw())

    x_ones, x_zeros = split_one_zero(x_list, y_list)
    half_len = len(x_ones)
    x_zeros = x_zeros[0:half_len]

    y_ones = [1] * half_len
    y_zeros = [0] * half_len

    x_ones.extend(x_zeros)
    y_ones.extend(y_zeros)

    X_train, X_test, y_train, y_test = train_test_split(np.asarray(x_ones), y_ones, test_size=0.33, random_state=42)
    print('x train shape: ' + str(X_train.shape))
    print('y train length: ' + str(len(y_train)))
    print('x test shape:  ' + str(X_test.shape))
    print('y test length: ' + str(len(y_test)))

    return X_train, y_train, X_test, y_test


def return_eval_data():
    # stop=True, stop_i=100
    # path='data/formmated_daily_data_0420.csv'
    x, y = read_raw(path='data/formmated_daily_data_0420.csv')
    print('x shape: ' + str(x.shape))
    print('y length: ' + str(len(y)))
    print('count 1: ' + str(y.count(1)))
    print(('count 0: ' + str(y.count(0))))
    return x, y


if __name__ == '__main__':
    return_train_data()

