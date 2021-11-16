# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Topic Title](#Topic-Title) 
# + [Topic 2 Title](#Topic-2-Title)

# ## Pivot tables
# Zeyuan Li
# zeyuanli@umich.edu
# 10/19/2021
# 
# 

# ## Pivot tables in pandas
# 
# The pivot tables in Excel is very powerful and convienent in handling with numeric data. Pandas also provides ```pivot_table()``` for pivoting with aggregation of numeric data. There are 5 main arguments of ```pivot_table()```:
# * ***data***: a DataFrame object
# * ***values***: a column or a list of columns to aggregate.
# * ***index***: Keys to group by on the pivot table index. 
# * ***columns***:  Keys to group by on the pivot table column. 
# * ***aggfunc***: function to use for aggregation, defaulting to ```numpy.mean```.

# ### Example

# In[3]:


df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": np.random.randn(24),
        "E": np.random.randn(24),
        "F": [datetime.datetime(2013, i, 1) for i in range(1, 13)]
        + [datetime.datetime(2013, i, 15) for i in range(1, 13)],
    }
)
df


# ### Do aggregation
# 
# * Get the pivot table easily. 
# * Produce the table as the same result of doing ```groupby(['A','B','C'])``` and compute the ```mean``` of D, with different values of D shown in seperate columns.
# * Change to another ***aggfunc*** to finish the aggregation as you want.

# In[4]:


pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])


# In[9]:


pd.pivot_table(df, values="D", index=["B"], columns=["A", "C"], aggfunc=np.sum)


# ### Display all aggregation values
# 
# * If the ***values*** column name is not given, the pivot table will include all of the data that can be aggregated in an additional level of hierarchy in the columns:

# In[6]:


pd.pivot_table(df, index=["A", "B"], columns=["C"])


# ### Output
# 
# * You can render a nice output of the table omitting the missing values by calling ```to_string```

# In[10]:


table = pd.pivot_table(df, index=["A", "B"], columns=["C"])
print(table.to_string(na_rep=""))

# *Kunheng Li(kunhengl@umich.edu)*
# *Oct 20, 2021*

import pandas as pd

# ## Topics in Pandas

# The reason I choose this function is because last homework. Before the hint from teachers, I found some ways to transfrom one row to many rows. Therefore, I will introduce a function to deal with this type of data.

# First, let's see an example.

data = {
    "first name":["kevin","betty","tony"],
    "last name":["li","jin","zhang"],
    "courses":["EECS484, STATS507","STATS507, STATS500","EECS402,EECS482,EECS491"]   
}
df = pd.DataFrame(data)
df = df.set_index(["first name", "last name"])["courses"].str.split(",", expand=True)\
    .stack().reset_index(drop=True, level=-1).reset_index().rename(columns={0: "courses"})
print(df)

# This is the first method I want to introduce, stack() or unstack(), both are similar. 
# Unstack() and stack() in DataFrame are to make itself to a Series which has secondary index.
# Unstack() is to transform its index to secondary index and its column to primary index, however, 
# stack() is to transform its index to primary index and its column to secondary index.

# However, in Pandas 0.25 version, there is a new method in DataFrame called explode(). They have the result, let's see the example.

df["courses"] = df["courses"].str.split(",")
df = df.explode("courses")
print(df)

# We can see the result is the same.

# # Title: pandas.DataFrame.cumsum
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

# [link](https://github.com/fyx1009/Stats507/blob/main/pandas_notes/pd_topic_fengyx.py)
