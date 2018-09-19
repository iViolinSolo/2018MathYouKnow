#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 19/09/2018
from pyecharts import Map

value = [95.1, 23.2, 43.3, 66.4, 88.5]
attr= ["China", "Canada", "Brazil", "Russia", "United States"]
map = Map("世界地图示例", width=1200, height=600)
map.add("", attr, value, maptype="world", is_visualmap=True, visual_text_color='#000')

map.render()