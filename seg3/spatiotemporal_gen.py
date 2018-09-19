#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 19/09/2018
from data.data_reader import DataReader
import pandas as pd
from pyecharts import Map
import numpy as np

df = pd.read_csv('./../data/kpro final/kpro-adv_ver-submission02.csv', index_col=0)

x = df.groupby(by=['iyear', 'CLUSTER_SCORE', 'region'])

m = x.count()['eventid']

df_plain = DataReader(name='./../data/gtd').df

df_plain['CLUSTER_SCORE'] = df['CLUSTER_SCORE']
df_tar = df_plain[['eventid', 'iyear', 'imonth', 'iday', 'region', 'country_txt', 'provstate', 'latitude', 'longitude',
                   'CLUSTER_SCORE']]
df_tar = df_tar[(df_tar['iyear'] == 2015) | (df_tar['iyear'] == 2016) | (df_tar['iyear'] == 2017)]
df_tar = df_tar[~(df_tar['longitude'].isna() | df_tar['latitude'].isna())]

m = df_tar.groupby(by=['iyear', 'imonth', 'country_txt']).count()['eventid']
n = m.reset_index()


def plot_my_map(year: int, month: int):
    df_tmp = n[(n['iyear'] == year) & (n['imonth'] == month)]
    value = df_tmp['eventid'].values
    attr = df_tmp['country_txt'].values

    map = Map("%d年%2d月恐怖袭击频次分布图" % (year, month), width=1200, height=600)
    map.add("", attr, value, maptype="world", is_visualmap=True, visual_text_color='#000')

    map.render(path='./../data/seg3/%d_%2d_terr_freq_overall.html' % (year, month))

# _geo = df_tar[['eventid', 'latitude', 'longitude']].values
# _data = df_tar[['eventid', 'CLUSTER_SCORE']].values
#
# y_geo = {}
# y_data = []
#
# for t_geo, t_data in zip(_geo, _data):
#     y_data.append({'name': str(int(t_data[0])), 'value': t_data[1]})
#     y_geo[str(int(t_geo[0]))] = [t_geo[1], t_geo[2]]


# print(y_data)
# print(y_geo)
#
# import json
# with open('./../data/seg3/data.json', 'w+') as fw:
#     fw.write(json.dumps(y_data))
#
# with open('./../data/seg3/geo.json', 'w+') as fw:
#     fw.write(json.dumps(y_geo))
