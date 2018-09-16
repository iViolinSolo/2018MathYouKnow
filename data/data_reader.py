#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 15/09/2018

import os
import pandas as pd


class DataReader:
    def __init__(self, name=None):
        self.df = self.read_data('gtd' if not name else name)

    @staticmethod
    def read_data(name) -> pd.DataFrame:
        if os.path.exists('%s.pkl' % name):
            return pd.read_pickle('%s.pkl' % name)
        elif os.path.exists('%s.xlsx' % name):
            _df = pd.read_excel('%s.xlsx' % name)
            pd.to_pickle(_df, '%s.pkl' % name)
            return _df
        else:
            return None

