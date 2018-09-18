#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 17/09/2018

from data.data_reader import DataReader

dr = DataReader(name='./../data/gtd')

df = dr.df
all_cols = ['eventid', 'iyear', 'imonth', 'iday', 'approxdate', 'extended', 'resolution', 'country', 'country_txt',
            'region', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'location',
            'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'alternative_txt', 'multiple', 'success',
            'suicide', 'attacktype1', 'attacktype1_txt', 'attacktype2', 'attacktype2_txt', 'attacktype3',
            'attacktype3_txt', 'targtype1', 'targtype1_txt', 'targsubtype1', 'targsubtype1_txt', 'corp1', 'target1',
            'natlty1', 'natlty1_txt', 'targtype2', 'targtype2_txt', 'targsubtype2', 'targsubtype2_txt', 'corp2',
            'target2', 'natlty2', 'natlty2_txt', 'targtype3', 'targtype3_txt', 'targsubtype3', 'targsubtype3_txt',
            'corp3', 'target3', 'natlty3', 'natlty3_txt', 'gname', 'gsubname', 'gname2', 'gsubname2', 'gname3',
            'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'individual', 'nperps', 'nperpcap',
            'claimed', 'claimmode', 'claimmode_txt', 'claim2', 'claimmode2', 'claimmode2_txt', 'claim3', 'claimmode3',
            'claimmode3_txt', 'compclaim', 'weaptype1', 'weaptype1_txt', 'weapsubtype1', 'weapsubtype1_txt',
            'weaptype2', 'weaptype2_txt', 'weapsubtype2', 'weapsubtype2_txt', 'weaptype3', 'weaptype3_txt',
            'weapsubtype3', 'weapsubtype3_txt', 'weaptype4', 'weaptype4_txt', 'weapsubtype4', 'weapsubtype4_txt',
            'weapdetail', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 'property', 'propextent',
            'propextent_txt', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays',
            'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'ransomnote',
            'hostkidoutcome', 'hostkidoutcome_txt', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource',
            'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related']

# get all data here.

target_col_name = [
    'eventid', 'iyear', 'imonth', 'iday', 'extended',
    'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple',
    'country', 'region', 'provstate', 'city', 'specificity', 'vicinity',
    'attacktype1', 'attacktype2', 'attacktype3', 'success', 'suicide',
    'weaptype1', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4',
    'targtype1', 'targsubtype1', 'targtype2', 'targsubtype2', 'targtype3', 'targsubtype3',
    'gname', 'nperps', 'nperpcap', 'claimed', 'claimmode', 'compclaim',
    'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 'property', 'propextent', 'propvalue', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'ransom', 'hostkidoutcome', 'nreleased',
    'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related'
]
df_target = df[target_col_name]

# fill nas
df_target.fillna({
    'eventid': 0, 'iyear': 0, 'imonth': 0, 'iday': 0, 'extended': 0,
    'crit1': 0, 'crit2': 0, 'crit3': 0, 'doubtterr': 0., 'alternative': 0., 'multiple': 0.,
    'country': 0, 'region': 0, 'provstate': '', 'city': '', 'specificity': 0., 'vicinity': 0,
    'attacktype1': 9, 'attacktype2': 9, 'attacktype3': 9, 'success': 0, 'suicide': 0,
    'weaptype1': 13, 'weapsubtype1': 27, 'weaptype2': 13, 'weapsubtype2': 27, 'weaptype3': 13, 'weapsubtype3': 27, 'weaptype4': 13, 'weapsubtype4': 27,
    'targtype1': 20, 'targsubtype1': 0, 'targtype2': 20, 'targsubtype2': 0, 'targtype3': 20, 'targsubtype3': 0,
    'gname': 'Unknown', 'nperps': 0, 'nperpcap': 0, 'claimed': 0, 'claimmode': 10, 'compclaim': 0,
    'nkill': 0, 'nkillus': 0, 'nkillter': 0, 'nwound': 0, 'nwoundus': 0, 'nwoundte': 0, 'property': -9, 'propextent': 4, 'propvalue': -99, 'ishostkid': 0, 'nhostkid': 0, 'nhostkidus': 0, 'nhours': 0, 'ndays': -99, 'ransom': 0, 'hostkidoutcome': 0, 'nreleased': 0,
    'INT_LOG': 0, 'INT_IDEO': 0, 'INT_MISC': 0, 'INT_ANY': 0, 'related': ''

}, inplace=True)


# do pre-process
df_needed_ori = df_target.drop(['eventid', 'iyear', 'imonth', 'iday', 'related'], axis=1)
df_1516_ori = df_target.loc[(df_target['iyear'] == 2015) | (df_target['iyear'] == 2016), :].drop(['eventid', 'iyear', 'imonth', 'iday', 'related'], axis=1)
df_17_ori = df_target.loc[df_target['iyear'] == 2017, :].drop(['eventid', 'iyear', 'imonth', 'iday', 'related'], axis=1)
df_needed = df_needed_ori[df_needed_ori['gname'] != 'Unknown']
df_needed_unk = df_needed_ori[df_needed_ori['gname'] == 'Unknown']
df_dbscan_train_unk = df_1516_ori[(df_1516_ori['gname'] == 'Unknown') & (df_1516_ori['claimed'] == 0)]
df_dbscan_train_real = df_dbscan_train_unk.drop(['gname'], axis=1)

# do labelencoding
from sklearn import preprocessing

# encode gname
le_gname = preprocessing.LabelEncoder()
en_gname = le_gname.fit_transform(df_needed['gname'])
df_needed['gname'] = en_gname.reshape((-1, 1))

# encode provstate
le_provstate = preprocessing.LabelEncoder()
en_provstate = le_provstate.fit_transform(df_needed['provstate'])
df_needed['provstate'] = en_provstate.reshape((-1, 1))
le_ds_provstate = preprocessing.LabelEncoder()
en_ds_provstate = le_ds_provstate.fit_transform(df_dbscan_train_real['provstate'])
df_dbscan_train_real['provstate'] = en_ds_provstate.reshape((-1, 1))

# encode city
le_city = preprocessing.LabelEncoder()
en_city = le_city.fit_transform(df_needed['city'])
df_needed['city'] = en_city.reshape((-1, 1))
le_ds_city = preprocessing.LabelEncoder()
en_ds_city = le_ds_city.fit_transform(df_dbscan_train_real['city'])
df_dbscan_train_real['city'] = en_ds_city.reshape((-1, 1))


# Filter all unknown gnames...
import pandas as pd
import numpy as np


allY = df_needed['gname']
allX = df_needed.drop(['gname'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("====> Begin split...")
SEED = 4
x_train, x_test, y_train, y_test = train_test_split(allX.values, allY.values, test_size=0.25, random_state=SEED)
print("====> Split finished...")

feat_labels = allX.columns[0:]
forest = RandomForestClassifier(n_estimators=10, random_state=SEED, n_jobs=-1, verbose=2)
print("====> Training begin...")
forest.fit(x_train, y_train)
print("====> Training Finished...")



importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


threshold = 0.003193
x_selected = x_train[:, importances > threshold]
print(x_selected.shape)

all_tar_idxs = []
for f in range(x_train.shape[1]):
    if importances[indices[f]] > threshold:
        all_tar_idxs.append(feat_labels[indices[f]])

print(all_tar_idxs)

x_selected_fea = df_dbscan_train_real[all_tar_idxs]
print(x_selected_fea.shape)


# NOW Doing DBSCAN
