#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 15/09/2018

from data.data_reader import DataReader

dr = DataReader(name='./../data/gtd')

dr.df.info()
