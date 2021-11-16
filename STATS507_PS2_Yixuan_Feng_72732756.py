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

# # Question 0 - Code review warmup

# ### Code Snippet

# +
#sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
#op = []
#for m in range(len(sample_list)):
#    li = [sample_list[m]]
#        for n in range(len(sample_list)):
#            if (sample_list[m][0] == sample_list[n][0] and
#                    sample_list[m][3] != sample_list[n][3]):
#                li.append(sample_list[n])
#        op.append(sorted(li, key=lambda dd: dd[3], reverse=True)[0])
#res = list(set(op))
# -

# #### a. Concisely describe what task the code above accomplishes. Say what it does (in total) and not how it accomplishes it. You may wish to understand the snippet step-by-step, but your description should not state each step individually.

# This code snippet serves to find a specific sublist for a given sample list. This code snippet finds the tuples that have the largest tail/third term among the ones with the same first term, collects them, and eventually eliminates the duplicates.

# #### b. Write a short code review that offers 3-5 (no more) concrete suggestions to make the snippent more efficient, literate (easier to read), or “pythonic”. Focus your suggestions on concepts or principles that would help the author of this code snippet write better code in the future.

# This code snippet didn't work initially because of its indentation and index out-of-range issues. The overall style of this code snippet is satisfactory (e.g., it follows the requirements of no spaces around “=" for parameter assignment and each line of code is within 75 characters) and is easy to understand. This code snippet is relatively inefficient as it is not necessary to sort the entire sublist with the same first term.
#
# Suggestions：
# 1. When the code structure is complex, attention should be paid to indentation issues.
# 2. If the length of tuples in the given list is uncertain, their length should be verified in advance.
# 3. This code snippet is relatively inefficient and should be made more efficient by, for example, taking the tuple with maximum value directly instead of sorting the whole sublist with the same first term.

# # Question 1 - List of Tuples

# #### Write a function that uses NumPy and a list comprehension to generate a random list of n k-tuples containing integers ranging from low to high. Choose an appropriate name for your function, and reasonable default values for k, low, and high.
#
# #### Use assert to test that your function returns a list of tuples.

import numpy as np
import time
import pandas as pd


def random_list(n, k=6, low=0, high=10):
    '''
    

    Parameters
    ----------
    n : int
        Number of tuples in the list.
    k : int
        Number of terms in each tuple. The default is 6.
    low : TYPE, optional
        The lower limit of the random number. The default is 2.
    high : TYPE, optional
        The upper limit of the random number. The default is 10.

    Returns
    -------
    None.

    '''
    res = []
    for i in range(n):
        res.append(tuple(sorted(tuple(np.random.choice(range(low, high),
                                                       k, replace = True)))))
    return(res) 
assert type(random_list(3)) == list, "Not a list."
for i in range(len(random_list(3))):
    assert type(random_list(3)[i]) == tuple, "Not a list of tuples."


# # Question 2 - Refactor the Snippet

# #### a. Encapsulate the code snippet from the warmup into a function that parameterizes the role of 0 and 3 and is otherwise unchanged. Choose appropriate names for these paramters.

def find_big_sublist(sample, refer=0, pos=2):
    '''
    This function serves to find a specific sublist for a given sample list. 
    It finds the tuples that have the largest term that used for comparison 
    among the ones with the same reference term, collects them, and eventually 
    eliminates the duplicates.

    Parameters
    ----------
    sample : list
        The given sample list.
    refer : int, optional
        The reference position. Used when testing whether the corresponding 
        values of the reference positions of the two tuples are the same. The 
        default is 0.
    pos : int, optional
        The comparison position. Used when testing which of the two tuples has 
        the greater corresponding value for the compared position. The default 
        is 2.

    Returns
    -------
    res : list
        A list of tuples that have the largest term that used for comparison 
        among the ones with the same reference term.

    '''
    op = []
    for m in range(len(sample)):
        li = [sample[m]]
        for n in range(len(sample)):
            if (sample[m][refer] == sample[n][refer] and
                    sample[m][pos] != sample[n][pos]):
                li.append(sample[n])
        op.append(sorted(li, key=lambda dd: dd[pos], reverse=True)[0])
    res = list(set(op))
    return res


# #### b. Write an improved version of the function form part a that implements the suggestions from the code review you wrote in part b of the warmup.

def find_big_sublist1(sample, refer=0, pos=2):
    '''
    This function serves to find a specific sublist for a given sample list. 
    It finds the tuples that have the largest term that used for comparison 
    among the ones with the same reference term, collects them, and eventually 
    eliminates the duplicates.

    Parameters
    ----------
    sample : list
        The given sample list.
    refer : int, optional
        The reference position. Used when testing whether the corresponding 
        values of the reference positions of the two tuples are the same. The 
        default is 0.
    pos : int, optional
        The comparison position. Used when testing which of the two tuples has 
        the greater corresponding value for the compared position. The default 
        is 3.

    Returns
    -------
    res : list
        A list of tuples that have the largest term that used for comparison 
        among the ones with the same reference term.

    '''
    op = []
    for m in range(len(sample)):
        li = [sample[m]]
        for n in range(len(sample)):
            if (sample[m][refer] == sample[n][refer] and
                        sample[m][pos] != sample[n][pos]):
                li.append(sample[n])
        max_tuple = max(li, key=lambda x: x[pos]) 
        op.append(max_tuple)
    res = list(set(op))
    return res


# #### c. Write a function from scratch to accomplish the same task as the previous two parts. Your solution should traverse the input list of tuples no more than twice. Hint: consider using a dictionary or a default dictionary in your solution.

def find_big_sublist2(sample, refer=0, pos=2):
    '''
    This function serves to find a specific sublist for a given sample list. 
    It finds the tuples that have the largest term that used for comparison 
    among the ones with the same reference term, collects them, and eventually 
    eliminates the duplicates.

    Parameters
    ----------
    sample : list
        The given sample list.
    refer : int, optional
        The reference position. Used when testing whether the corresponding 
        values of the reference positions of the two tuples are the same. The 
        default is 0.
    pos : int, optional
        The comparison position. Used when testing which of the two tuples has 
        the greater corresponding value for the compared position. The default 
        is 3.

    Returns
    -------
    res : list
        A list of tuples that have the largest term that used for comparison 
        among the ones with the same reference term.

    '''    
    op = []
    a = list(zip(*sample))
    for m in a[refer]:
        li = []
        for n in sample:
            if n[refer] == m:
                li.append(n)
        max_tuple = max(li, key=lambda x: x[pos])
        for i in li:
            if i[pos] == max_tuple[pos]:
                op.append(i)
    res = list(set(op))
    return res


# #### d. Use the function you wrote in question 1 to generate a list of tuples as input(s), run and summarize a small Monte Carlo study comparing the execution times of the three functions above (a-c).

# +
i = 0
a1, b1, c1 = [], [], []
while i < 10000:
    sample_list = random_list(np.random.choice(range(3, 10)), 
                              np.random.choice(range(3, 10)))
    for f in (find_big_sublist, find_big_sublist1, find_big_sublist2):
        start = time.time()
        f(sample_list)
        end = time.time()
        optime = end-start
        if f == find_big_sublist:
            a1.append(optime)
        if f == find_big_sublist1:
            b1.append(optime)
        if f == find_big_sublist2:
            c1.append(optime)
    i += 1

print(np.mean(a1))
print(np.mean(b1))
print(np.mean(c1))
# -

# # Question 3

# #### a. Use Python and Pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Add an additional column identifying to which cohort each case belongs. Rename the columns with literate variable names using all lower case and convert each column to an appropriate type. Finally, save the resulting data frame to a serialized “round-trip” format of your choosing (e.g. pickle, feather, or parquet).

df1 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT')
df2 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT')
df3 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT')
df4 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT')

df1 = df1.loc[:, ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 
                  'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df2 = df1.loc[:, ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 
                  'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df3 = df1.loc[:, ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 
                  'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df4 = df1.loc[:, ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 
                  'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df1.columns = ['unique_ids', 'age', 'race&ethnicity', 'education', 
               'marital_status', 'interview_status', 
               'masked_variance_unit_pseudo_PSU_variable', 
               'masked_variance_unit_pseudo_stratum_variable', 
               'interviewed&mec_examined_participants', 
               'interviewed_participants']
df2.columns = ['unique_ids', 'age', 'race&ethnicity', 'education', 
               'marital_status', 'interview_status', 
               'masked_variance_unit_pseudo_PSU_variable', 
               'masked_variance_unit_pseudo_stratum_variable', 
               'interviewed&mec_examined_participants', 
               'interviewed_participants']
df3.columns = ['unique_ids', 'age', 'race&ethnicity', 'education', 
               'marital_status', 'interview_status', 
               'masked_variance_unit_pseudo_PSU_variable', 
               'masked_variance_unit_pseudo_stratum_variable', 
               'interviewed&mec_examined_participants', 
               'interviewed_participants']
df4.columns = ['unique_ids', 'age', 'race&ethnicity', 'education', 
               'marital_status', 'interview_status', 
               'masked_variance_unit_pseudo_PSU_variable', 
               'masked_variance_unit_pseudo_stratum_variable', 
               'interviewed&mec_examined_participants', 
               'interviewed_participants']
df1['cohort'] = 2011
df2['cohort'] = 2013
df3['cohort'] = 2015
df4['cohort'] = 2017
df1 = df1.fillna(-1).astype(int)
df2 = df2.fillna(-1).astype(int)
df3 = df3.fillna(-1).astype(int)
df4 = df4.fillna(-1).astype(int)

df = df1.append(df2)
df = df.append(df3)
df = df.append(df4)
df.to_pickle("./demographic.pkl")
df

# #### b. Repeat part a for the oral health and dentition data (OHXDEN_*.XPT) retaining the following variables: SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).

df5 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OHXDEN_G.XPT')
df6 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/OHXDEN_H.XPT')
df7 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/OHXDEN_I.XPT')
df8 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHXDEN_J.XPT')

pd.set_option('display.max_columns', None)
df5

l = [0, 2]
l.extend([i for i in range(4, 64)])
df5 = df5.iloc[:, l]
df6 = df6.iloc[:, l]
df7 = df7.iloc[:, l]
df8 = df8.iloc[:, l]

li = ['respondent_sequence_number', 'dentition_status_code']
for i in range(32):
    li.append("tooth_counts" + str(i+1))
for i in range(2,16):
    li.append("coronal_cavities" + str(i))
for i in range(18,32):
    li.append("coronal_cavities" + str(i))
df5.columns = li
df6.columns = li
df7.columns = li
df8.columns = li

df5 = pd.concat([df5.iloc[:, :34].fillna(-1).astype(int), 
                 df5.iloc[:, 34:-1].astype(str)], axis = 1)
df6 = pd.concat([df6.iloc[:, :34].fillna(-1).astype(int), 
                 df6.iloc[:, 34:-1].astype(str)], axis = 1)
df7 = pd.concat([df7.iloc[:, :34].fillna(-1).astype(int), 
                 df7.iloc[:, 34:-1].astype(str)], axis = 1)
df8 = pd.concat([df8.iloc[:, :34].fillna(-1).astype(int), 
                 df8.iloc[:, 34:-1].astype(str)], axis = 1)

df5['cohort'] = 2011
df6['cohort'] = 2013
df7['cohort'] = 2015
df8['cohort'] = 2017

df_new = df5.append(df6)
df_new = df_new.append(df7)
df_new = df_new.append(df8)
df_new

df_new.to_pickle('./ohxden.pkl')

# #### c. In your notebook, report the number of cases there are in the two datasets above.

df.shape

df_new.shape
