# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # slide
# - Name: Yixuan Feng
# - Email: fengyx@umich.edu

# # Question 0 - Topics in Pandas
# #### For this question, please pick a topic - such as a function, class, method, recipe or idiom related to the pandas python library and create a short tutorial or overview of that topic. The only rules are below.
# 1. Pick a topic not covered in the class slides.
# 2. Do not knowingly pick the same topic as someone else.
# 3. Use bullet points and titles (level 2 headers) to create the equivalent of 3-5 “slides” of key points. They shouldn’t actually be slides, but please structure your key points in a manner similar to the class slides (viewed as a notebook).
# 4. Include executable example code in code cells to illustrate your topic.

import numpy as np 
import pandas as pd 

# ## pandas.DataFrame.cumsum
# - Cumsum is the cumulative function of pandas, used to return the cumulative values of columns or rows.

# ## Example 1 - Without Setting Parameters
# - This function will automatically return the cumulative value of all columns.

values_1 = np.random.randint(10, size=10) 
values_2 = np.random.randint(10, size=10) 
group = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'] 
df = pd.DataFrame({'group':group, 'value_1':values_1, 'value_2':values_2}) 
df

df.cumsum()

# ## Example 2 - Setting Parameters
# - By setting the axis to 1, this function will return the cumulative value of all rows.
# - By combining with groupby() function, other columns (or rows) can be used as references for cumulative addition.

df['cumsum_2'] = df[['group', 'value_2']].groupby('group').cumsum() 
df 
