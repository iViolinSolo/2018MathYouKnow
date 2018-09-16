#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 15/09/2018

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

tar_cols = ['eventid', 'iyear', 'imonth', 'iday',
            'attacktype1', 'success', 'suicide',
            'weaptype1',
            'extended', 'crit1', 'crit2', 'crit3', 'doubtterr', 'multiple', 'related', 'region', 'vicinity',
            'nperps', 'nperpcap', 'claimed', 'compclaim',
            'targtype1', 'nkill', 'nkillter', 'nwound',
            'property', 'propextent', 'propvalue',
            'ishostkid', 'nhostkid', 'nhours', 'ndays', 'ransom', 'ransomamt', 'hostkidoutcome',
            'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']
dftarget = df[tar_cols]

dftarget.fillna({'eventid':0, 'iyear':0, 'imonth':0, 'iday':0,
            'attacktype1':0, 'success':0, 'suicide':0,
            'weaptype1':0,
            'extended':0, 'crit1':0, 'crit2':0, 'crit3':0, 'doubtterr':0., 'multiple':0., 'related':'', 'region':0, 'vicinity':0,
            'nperps':0, 'nperpcap':0, 'claimed':0, 'compclaim':0,
            'targtype1':0, 'nkill':0, 'nkillter':0, 'nwound':0,
            'property':0, 'propextent':0, 'propvalue':0,
            'ishostkid':0, 'nhostkid':0, 'nhours':0, 'ndays':0, 'ransom':0, 'ransomamt':0, 'hostkidoutcome':0,
            'INT_LOG':0, 'INT_IDEO':0, 'INT_MISC':0, 'INT_ANY':0}, inplace=True)



import numpy as np
from kmodes.kmodes import KModes

km = KModes(n_clusters=5, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(dftarget.drop(['eventid', 'iyear', 'imonth', 'iday',
                                         'related',
                                         'nperps', 'nperpcap', 'nkill', 'nkillter', 'propvalue',
                                         'nwound', 'nhostkid', 'nhours', 'ndays', 'ransomamt'], axis=1))

# Print the cluster centroids
print(km.cluster_centroids_)



import pandas as pd
df_result = pd.DataFrame(dftarget)
df_result['CLUSTER'] = clusters.reshape((-1, 1))