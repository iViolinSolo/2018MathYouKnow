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


threshold = 0.005
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
import hdbscan
from sklearn.cluster import DBSCAN
import time


#DBSCAN
t0 = time.time()
# dbscan = DBSCAN(eps=100, min_samples=3)
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
cluster_labels = clusterer.fit_predict(x_selected_fea)
# hierarchy = clusterer.cluster_hierarchy_
# alt_labels = hierarchy.get_clusters(0.100, 5)
# hierarchy.plot()

cunique = np.unique(cluster_labels)
l = {i: cluster_labels[cluster_labels==i].size for i in cunique}
l2 = [cluster_labels[cluster_labels==i].size for i in cunique]

# ordered dict for unique labels and it's count
dict = sorted(l.items(), key=lambda d:d[1], reverse = True)


# to store final cluster result..
df_hdbscan_final_sub = pd.DataFrame(x_selected_fea)
df_hdbscan_final_sub['CLUSTER'] = cluster_labels.reshape((-1, 1))

df_q1_res = pd.read_csv('./../data/kpro final/kpro-alldata-submission02.csv')
df_q1_res = df_q1_res.drop('Unnamed: 0', 1)
# add additional infos
df_q1_res.loc[df_q1_res['CLUSTER'] == 4, 'CLUSTER_CLASSIFIED'] = 'Level One'
df_q1_res.loc[df_q1_res['CLUSTER'] == 4, 'CLUSTER_SCORE'] = 5
df_q1_res.loc[df_q1_res['CLUSTER'] == 0, 'CLUSTER_CLASSIFIED'] = 'Level Two'
df_q1_res.loc[df_q1_res['CLUSTER'] == 0, 'CLUSTER_SCORE'] = 4
df_q1_res.loc[df_q1_res['CLUSTER'] == 2, 'CLUSTER_CLASSIFIED'] = 'Level Three'
df_q1_res.loc[df_q1_res['CLUSTER'] == 2, 'CLUSTER_SCORE'] = 3
df_q1_res.loc[df_q1_res['CLUSTER'] == 1, 'CLUSTER_CLASSIFIED'] = 'Level Four'
df_q1_res.loc[df_q1_res['CLUSTER'] == 1, 'CLUSTER_SCORE'] = 2
df_q1_res.loc[df_q1_res['CLUSTER'] == 3, 'CLUSTER_CLASSIFIED'] = 'Level Five'
df_q1_res.loc[df_q1_res['CLUSTER'] == 3, 'CLUSTER_SCORE'] = 1

df_q1_res_tar_slice = df_q1_res.loc[df_hdbscan_final_sub.index, :]

# Agg all data, combine them
df_q1_res_tar_slice['CODE_NAME'] = cluster_labels.reshape((-1, 1))
df_q1_res_tar_slice['CODE_NAME_POSSIBILITY'] = clusterer.probabilities_.reshape((-1, 1))
df_q1_res_tar_slice.to_csv('./../data/dbscan final/dbscan-slice-15_16-codename-submission01.csv')


df_hdbscan_final_sub['POSSIBILITY'] = clusterer.probabilities_.reshape((-1, 1))
df_hdbscan_final_sub['eventid'] = df_q1_res_tar_slice['eventid']
df_hdbscan_final_sub.to_csv('./../data/dbscan final/dbscan-ori-trainwithres-submission00.csv')


# calculate danger_confidence
x = pd.Series(df_q1_res_tar_slice.groupby(by=['CODE_NAME']).sum()['CLUSTER_SCORE'])
x = x.sort_values(ascending=False)
x.to_csv('./../data/dbscan final/dbscan-dangerlevel-sum-codename-submission02.csv')

# DOING distance ....
all_five_codename = x.index[1:6].tolist()
exemplars_ = clusterer.exemplars_
best_top_5_in_rank5 = np.asarray([exemplars_[i][0] for i in all_five_codename])

# 2017 needed to predicts
df_predict_index_target = [
    201701090031,
    201702210037,
    201703120023,
    201705050009,
    201705050010,
    201707010028,
    201707020006,
    201708110018,
    201711010006,
    201712010003
]

df_predict_from_source = df_target[df_target['eventid'].isin(df_predict_index_target)]


df_predict_selected_fea = df_predict_from_source[all_tar_idxs]
# x_selected_fea = df_dbscan_train_real[all_tar_idxs]

# encode provstate
le_pr_provstate = preprocessing.LabelEncoder()
en_pr_provstate = le_pr_provstate.fit_transform(df_predict_selected_fea['provstate'])
df_predict_selected_fea['provstate'] = en_pr_provstate.reshape((-1, 1))

# encode city
le_pr_city = preprocessing.LabelEncoder()
en_pr_city = le_pr_city.fit_transform(df_predict_selected_fea['city'])
df_predict_selected_fea['city'] = en_pr_city.reshape((-1, 1))


def calEuclideanDistance(vec1, vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist


# calculate distance
all_dists = np.zeros((len(df_predict_index_target), len(all_five_codename)))
for i in range(len(df_predict_index_target)):
    for j in range(len(all_five_codename)):
        dist = calEuclideanDistance(df_predict_selected_fea.values[i], best_top_5_in_rank5[j])
        all_dists[i][j] = dist

print(all_dists)