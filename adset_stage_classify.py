import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
import numpy as np
import os

from datetime import datetime, timedelta

HOUR_HEADER = ['adset_id', 'event_date',
       'event_hour', 'hour_adset_click', 'hour_adset_convert',
       'hour_adset_impressions', 'hour_avg_click', 'hour_avg_impression',
       'hour_clicks_trident', 'hour_convert_trident', 'hour_cost_rmb',
       'hour_ctr', 'hour_cvr', 'hour_earnings_rmb', 'hour_ecpm',
       'hour_impressions_trident', 'hour_nominal_cpi', 'hour_roas', 'hour_rr',
       'hour_trident_cpi', 'learning_phase', 'life_adset_click',
       'life_adset_convert', 'life_adset_impressions', 'life_clicks_trident',
       'life_convert_trident', 'life_cost_rmb', 'life_ctr', 'life_cvr',
       'life_earnings_rmb', 'life_ecpm', 'life_impressions_trident',
       'life_nominal_cpi', 'life_rr', 'life_trident_cpi', 'media_source']

DATE_HEADER = ['adset_id', 'event_date',
       'event_hour','day_adset_click','learning_phase', 'life_adset_click',
       'life_adset_convert', 'life_adset_impressions', 'life_clicks_trident',
       'life_convert_trident', 'life_cost_rmb', 'life_ctr', 'life_cvr',
       'life_earnings_rmb', 'life_ecpm', 'life_impressions_trident',
       'life_nominal_cpi', 'life_rr', 'life_trident_cpi',
       'day_adset_convert', 'day_adset_impressions', 'day_clicks_trident',
       'day_convert_trident', 'day_cost_rmb', 'day_ctr', 'day_cvr',
       'day_earnings_rmb', 'day_ecpm', 'day_impressions_trident',
       'day_nominal_cpi', 'day_rr', 'day_trident_cpi']


def adset_hour_history(df, adset_id, date, hour):
    return df[(df['adset_id'] == adset_id) & ((df['event_date'] < date) | ((df['event_date'] == date) & (df['event_hour'] <= hour)))]


def adset_date_history(df, adset_id, date=None, date_range=None):
    if date_range:
        df = df[(df['adset_id'] == adset_id) &
                (df['event_date'] >= (date - timedelta(days=date_range))) &
                (df['event_date'] < date)]
    elif date:
        df = df[(df['adset_id'] == adset_id) & (df['event_date'] < date)]
    else:
        df = df[df['adset_id'] == adset_id]

    ll = []
    for d in df.event_date.unique():
        cur_date_df = df[df['event_date'] == d]
        end_hour = max(df.event_hour)
        cur_date_df = cur_date_df[cur_date_df['event_hour'] == end_hour]
        ll.append(cur_date_df)

    df = pd.concat(ll, ignore_index=True)
    return df[DATE_HEADER]


def draw_two_axes_fig(title, x, y1, y2,
                      basepath='/Users/tracy/Desktop/0301_adset_figs/',
                      y1_label='hour_nominal_cpi',
                      y2_label='hour_adset_impressions'):
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(211)
    ax1.plot(x, y1, color='tab:blue', marker='o')
    ax1.set_ylabel(y1_label)

    ax2 = fig.add_subplot(212)
    ax2.plot(x, y2, color='tab:orange', marker='o')
    ax2.set_ylabel(y2_label)

    fig.suptitle(str(title))

    if not os.path.exists(basepath):
        os.makedirs(basepath)
    plt.savefig(os.path.join(basepath, str(title) + '.png'))
    plt.close()


def read_data(basepath='data', target_date=None, target_hour=None):
    files = [os.path.join(basepath, f) for f in os.listdir(basepath) if f.endswith('.csv')]
    data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    data['event_date'] = pd.to_datetime(data['event_date'])
    if target_date:
        return data[data['event_date'] <= target_date]
    return data


def classify_stage(test_size=50):
    # 计算日期，小时
    target_date = datetime.strptime('2021-03-14', '%Y-%m-%d')
    hour = 23

    data = read_data(target_date=target_date)

    adset_id_list = data.adset_id.unique()
    print('adset count: ' + str(len(adset_id_list)))

    count = 0
    for adset_id in adset_id_list:
        # df = adset_hour_history(data, adset_id, date, hour)
        # df['datetimeHour'] = pd.to_timedelta(df.Hour, unit='hour')
        # df['datetimeTime'] = df['Date'] + df['datetimeHour']
        # df = df.sort_values(by=['datetimeTime'])

        df = adset_date_history(data, adset_id, target_date, date_range=8)
        df = df.sort_values(by=['event_date'])

        if 'LEARN_FAILED' in df.learning_phase:
            class_type = 'learning_failed'
        elif sum(df.day_adset_impressions) <= 10:
            class_type = 'small_convert'
        else:
            class_type = 'running'

        draw_two_axes_fig(adset_id, df.Date, df.day_nominal_cpi, df.day_adset_impressions,
                          os.path.join('/Users/tracy/Desktop/0301_adset_figs/', class_type),
                          y1_label='cpi', y2_label='impression')

        if count >= test_size:
            break

        count += 1


def datapoints_view(data, x_label, y_label, save_data=True):
    x = []
    y = []

    for adset in data.adset_id.unique():
        date_df = data[data['adset_id'] == adset]
        for d in date_df.event_date.unique():

            df = date_df[date_df['event_date'] == d]
            end_hour = max(df['event_hour'].values)
            df = df[df['event_hour'] == end_hour]
            y.append(df[y_label].values[0])
            x.append(df[x_label].values[0])
    if save_data:
        datapoint_dict = {x_label: x, y_label: y}
        df = pd.DataFrame(datapoint_dict)
        df.to_csv(os.path.join('results', x_label+'_' + y_label + '.csv'))
    return x, y


def rate_date_points_view(save_data=True):
    y_labels = ['day_nominal_cpi', 'day_adset_impressions', 'day_rr'
                'life_nominal_cpi', 'life_adset_impressions', 'life_rr']

    cols = ['adset', 'event_date']
    for y in y_labels:
        for i in (5, 3, 2):
            range_label = y + str(i)
            header = [range_label, range_label + 'min', range_label + 'max', range_label + 'slope', range_label + 'count']
            cols.extend(header)


def get_slope_indicators(data, label):
    return None


def get_range(data, adset_id, end_date, day_range):
    return data[(data['adset_id'] == adset_id) &
                (data['event_date'] >= (end_date - timedelta(days=(day_range-1))))]


def test(target_date):
    data = pd.read_csv('results/daily_data2021-03-10.csv')
    data['event_date'] = pd.to_datetime(data['event_date'])
    data = data.sort_values(by=['event_date'])
    header = ['adset_id', 'event_date', 'day_range', 'y_label', 'mean', 'max', 'min', 'slope', 'r2_score']
    result = [header]
    for adset_id in data.adset_id.unique():
        day_range = 8
        range_data = get_range(data, adset_id, target_date, day_range)
        if len(range_data.index) == day_range:
            print(range_data[['adset_id', 'event_date', 'day_adset_impressions']])
            x = [[i] for i in range(day_range)]
            y_list = range_data['day_adset_impressions']
            y_min = min(y_list)
            y_max = max(y_list)
            y_mean = np.mean(y_list)

            lm = LinearRegression()
            lm.fit(x, y_list)
            y_predict = lm.predict(x)
            r2 = r2_score(y_list, y_predict)

            slope = lm.coef_[0]
            result.append([adset_id, target_date, day_range, 'day_adset_impressions',
                           y_mean, y_max, y_min, slope, r2])

    result_df = pd.DataFrame(result[1:], columns=result[0])
    read_append_test('results/day_adset_impressions_rate_0310.csv', result_df)
    # result_df.to_csv('results/day_adset_impressions_rate_0310.csv', index=False)


def read_append_test(path, result2):
    result1 = pd.read_csv(path)
    df_new = pd.concat([result1, result2], ignore_index=True)
    df_new.to_csv(path, index=False)


if __name__ == '__main__':
    arr = np.zeros((8, 8), dtype=float)
    arr[:, :4] = 4.0
    arr[:, 4:] = 1.0
    print(arr)

    # 计算日期，小时
    # target_date = datetime.strptime('2021-03-11', '%Y-%m-%d')
    # hour = 23
    # test(target_date)

    # data = read_data(target_date=target_date)

    # to_save_data = []
    # for ad in data.adset_id.unique():
    #     to_save_data.append(adset_date_history(data, ad))
    #
    # df = pd.concat(to_save_data, ignore_index=True)
    # df.to_csv('results/daily_data2021-03-10.csv')

    # x_label = 'day_nominal_cpi'
    # y_label = 'day_adset_impressions'
    #
    # x, y = datapoints_view(data, x_label, y_label, save_data=True)
    #
    # plt.hist(y, color='blue', edgecolor='black')
    # plt.title(y_label)
    # plt.show()






    # 风控：爆量 & CPI跑飘
    # 阶段判断：冷启动，观察，放量，衰退
