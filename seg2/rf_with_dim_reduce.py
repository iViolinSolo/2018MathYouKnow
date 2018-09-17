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
df_needed = df[target_col_name]

# 'location',
#    'summary',   'alternative_txt',
#
#      'gsubname', 'gname2', 'gsubname2', 'gname3',
#    'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'individual',
#     'claimmode_txt', 'claim2', 'claimmode2', 'claimmode2_txt', 'claim3', 'claimmode3',
#    'claimmode3_txt',
#    'weapdetail',
#    'propextent_txt',  'propcomment',
#    'divert', 'kidhijcountry',  'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'ransomnote',
#     'hostkidoutcome_txt',  'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource',


# Filter all unknown gnames...
import pandas as pd
import numpy as np


allY = df_needed['gname']
allX = df_needed.drop(['gname'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 4
x_train, x_test, y_train, y_test = train_test_split(allX.values, allY.values, test_size=0.25, random_state=SEED)

feat_labels = df_needed.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=SEED, n_jobs=-1)
forest.fit(x_train, y_train)


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


threshold = 0.15
x_selected = x_train[:, importances > threshold]
x_selected.shape