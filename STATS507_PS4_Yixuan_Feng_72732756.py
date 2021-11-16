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

# # Question 0 - Topics in Pandas
# #### For this question, please pick a topic - such as a function, class, method, recipe or idiom related to the pandas python library and create a short tutorial or overview of that topic. The only rules are below.
# 1. Pick a topic not covered in the class slides.
# 2. Do not knowingly pick the same topic as someone else.
# 3. Use bullet points and titles (level 2 headers) to create the equivalent of 3-5 “slides” of key points. They shouldn’t actually be slides, but please structure your key points in a manner similar to the class slides (viewed as a notebook).
# 4. Include executable example code in code cells to illustrate your topic.

import numpy as np 
import pandas as pd 
import scipy.stats as stats
from scipy.stats import norm, binom, beta
import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from warnings import warn
import statsmodels.stats.proportion

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

# # Question 1 - NHANES Table 1
# ## part a)
# Revise your solution to PS2 Question 3 to also include gender (RIAGENDR) in the demographic data.
#
# Update (October 14): Include your data files in your submission and with extension .pickle, .feather or .parquet and include a code cell here that imports those files from the local directory (the same folder as your .ipynb or .py source files).

df1 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT')
df2 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT')
df3 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT')
df4 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT')

df1 = df1.loc[:, ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 
        'DMDMARTL', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df2 = df2.loc[:, ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 
        'DMDMARTL', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df3 = df3.loc[:, ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 
        'DMDMARTL', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
df4 = df4.loc[:, ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 
        'DMDMARTL', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]

for i in (df1, df2, df3, df4):
    i.columns = ['unique_ids', 'gender', 'age', 'race&ethnicity', 'education', 
                   'marital_status', 'exam_status', 
                   'masked_variance_unit_pseudo_PSU_variable', 
                   'masked_variance_unit_pseudo_stratum_variable', 
                   'interviewed&mec_examined_participants', 
                   'interviewed_participants']

df1['cohort'] = 2011
df2['cohort'] = 2013
df3['cohort'] = 2015
df4['cohort'] = 2017

df_a = df1.append(df2)
df_a = df_a.append(df3)
df_a = df_a.append(df4)
df_a.loc[df_a['gender'] == 1, 'gender'] = "Male"
df_a.loc[df_a['gender'] == 2, 'gender'] = "Female"
df_a.loc[df_a['race&ethnicity'] == 1, 'race&ethnicity'] = 'Mexican American'
df_a.loc[df_a['race&ethnicity'] == 2, 'race&ethnicity'] = 'Other Hispanic'
df_a.loc[df_a['race&ethnicity'] == 3, 'race&ethnicity'] = 'Non-Hispanic White'
df_a.loc[df_a['race&ethnicity'] == 4, 'race&ethnicity'] = 'Non-Hispanic Black'
df_a.loc[df_a['race&ethnicity'] == 5, 'race&ethnicity'] = 'Non-Hispanic Asian'
df_a.loc[df_a['race&ethnicity'] == 6,
         'race&ethnicity'] = 'Other Race - Including Multi-Racial'
df_a.loc[df_a['education'] == 1, 'education'] = 'Less Than 9th Grade'
df_a.loc[df_a['education'] == 2,
         'education'] = '9-11th Grade (Includes 12th grade with no diploma)'
df_a.loc[df_a['education'] == 3,
         'education'] = 'High School Grad/GED or Equivalent'
df_a.loc[df_a['education'] == 4, 'education'] = 'Some College or AA degree'
df_a.loc[df_a['education'] == 5, 'education'] = 'College Graduate or above'
df_a.loc[df_a['education'] == 7, 'education'] = 'Refused'
df_a.loc[df_a['education'] == 9, 'education'] = "Don't Know"
df_a.loc[df_a['marital_status'] == 1, 'marital_status'] = "Married"
df_a.loc[df_a['marital_status'] == 2, 'marital_status'] = "Widowed"
df_a.loc[df_a['marital_status'] == 3, 'marital_status'] = "Divorced"
df_a.loc[df_a['marital_status'] == 4, 'marital_status'] = "Separated"
df_a.loc[df_a['marital_status'] == 5, 'marital_status'] = "Never married"
df_a.loc[df_a['marital_status'] == 6, 'marital_status'] = "Living with partner"
df_a.loc[df_a['marital_status'] == 77, 'marital_status'] = "Refused"
df_a.loc[df_a['marital_status'] == 99, 'marital_status'] = "Don't Know"
df_a.loc[df_a['exam_status'] == 1, 'exam_status'] = "Interviewed only"
df_a.loc[df_a['exam_status'] == 2,
         'exam_status'] = "Both interviewed and MEC examined"
df_a['unique_ids'] = df_a['unique_ids'].astype(int)
df_a['age'] = df_a['age'].astype(int)
df_a.to_pickle("./demographic.pkl")

# ## part b)
# The variable OHDDESTS contains the status of the oral health exam. Merge this variable into the demographics data.
#
# Use the revised demographic data from part a and the oral health data from PS2 to create a clean dataset with the following variables:
#
# - id (from SEQN)
#
# - gender
#
# - age
#
# - under_20 if age < 20
#
# - college - with two levels:
#
#     - ‘some college/college graduate’ or
#
#     - ‘No college/<20’ where the latter category includes everyone under 20 years of age.
#
# - exam_status (RIDSTATR)
#
# - ohx_status - (OHDDESTS)
#
# Create a categorical variable in the data frame above named ohx with two levels “complete” for those with exam_status == 2 and ohx_status == 1 or “missing” when ohx_status is missing or corresponds to “partial/incomplete.”

df5 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OHXDEN_G.XPT')
df6 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/OHXDEN_H.XPT')
df7 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/OHXDEN_I.XPT')
df8 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHXDEN_J.XPT')

l = [0, 2]
df_b = df5.iloc[:, l]
df_b = df_b.append(df6.iloc[:, l])
df_b = df_b.append(df7.iloc[:, l])
df_b = df_b.append(df8.iloc[:, l])
df_b.columns = ['unique_ids', 'ohx_status']
df_b.loc[df_b['ohx_status'] == 1,
         'ohx_status'] = "Complete"
df_b.loc[df_b['ohx_status'] == 2,
         'ohx_status'] = "Partial"
df_b.loc[df_b['ohx_status'] == 3,
         'ohx_status'] = "Not Done"
df_b['unique_ids'] = df_b['unique_ids'].astype(int)

df_a.drop(columns=['masked_variance_unit_pseudo_PSU_variable',
                   'masked_variance_unit_pseudo_stratum_variable',
                   'interviewed&mec_examined_participants',
                   'interviewed_participants'],  inplace=True)

df_a = df_a.merge(df_b, how='left', on='unique_ids')
df_a.reset_index(inplace=True)

df_a['under_20'] = df_a.apply(lambda x: True if x.age < 20 else False, axis=1)
df_a['college'] = df_a.apply(lambda x: 'some college/college graduate' if 
                             x.education == ('Some College or AA degree' or 
                            'College Graduate or above') else 'No college/<20',
                            axis=1)
a = df_a.query("under_20")==True
for i in a['college']:
    assert i == False

df_a.drop(columns=['index', 'race&ethnicity', 'marital_status', 'cohort'],
          inplace=True)

df_a = df_a[['unique_ids', 'gender', 'age', 'under_20', 'college', 
             'exam_status', 'ohx_status']]

df_a['ohx'] = df_a.apply(lambda x: 'complete' if 
                         (x.exam_status == 'Both interviewed and MEC examined' 
                          and x.ohx_status == 'Complete') else 'missing',
                         axis=1)
b = df_a[df_a['ohx']=='missing']
for i in b['ohx_status']:
    assert i != "Complete"

df_a

# ## part c)
# Remove rows from individuals with exam_status != 2 as this form of missingness is already accounted for in the survey weights. Report the number of subjects removed and the number remaining.

df_a=df_a[-df_a.exam_status.isin(['Interviewed only'])]
df_a

# Before Removing: 39156 rows
#
# After Removing: 37399 rows
#
# Number of removed: 1757
#
# Number of remaining: 37399

# ## part d)
# Construct a table with ohx (complete / missing) in columns and each of the following variables summarized in rows:
#
# age
#
# under_20
#
# gender
#
# college
#
# For the rows corresponding to categorical variable in your table, each cell should provide a count (n) and a percent (of the row) as a nicely formatted string. For the continous variable age, report the mean and standard deviation [Mean (SD)] for each cell.
#
# Include a column ‘p-value’ giving a p-value testing for a mean difference in age or an association beween each categorical varaible and missingness. Use a chi-squared test comparing the 2 x 2 tables for each categorical characteristic and OHX exam status and a t-test for the difference in age.
#
# **Hint*: Use scipy.stats for the tests.

df_new = pd.melt(df_a, id_vars=['ohx'],
                 value_vars=['age','under_20','gender','college'])
df_comp = df_new[df_new['ohx'] == 'complete']
df_miss = df_new[df_new['ohx'] == 'missing']

age1 = pd.DataFrame(df_comp.groupby
                    (by='variable'))[1][0]['value'].value_counts()
under_201 = pd.DataFrame(df_comp.groupby
                         (by='variable'))[1][1]['value'].value_counts()
gender1 = pd.DataFrame(df_comp.groupby
                       (by='variable'))[1][2]['value'].value_counts()
college1 = pd.DataFrame(df_comp.groupby
                        (by='variable'))[1][3]['value'].value_counts()
age2 = pd.DataFrame(df_miss.groupby
                    (by='variable'))[1][0]['value'].value_counts()
under_202 = pd.DataFrame(df_miss.groupby
                         (by='variable'))[1][1]['value'].value_counts()
gender2 = pd.DataFrame(df_miss.groupby
                       (by='variable'))[1][2]['value'].value_counts()
college2 = pd.DataFrame(df_miss.groupby
                        (by='variable'))[1][3]['value'].value_counts()

df_age1 = pd.DataFrame(age1)
dic1 = dict(zip(df_age1.index.values.tolist(), 
                df_age1['value'].tolist()))
df_under_201 = pd.DataFrame(under_201)
dic2 = dict(zip(df_under_201.index.values.tolist(),
                df_under_201['value'].tolist()))
df_gender1 = pd.DataFrame(gender1)
dic3 = dict(zip(df_gender1.index.values.tolist(),
                df_gender1['value'].tolist()))
df_college1 = pd.DataFrame(college1)
dic4 = dict(zip(df_college1.index.values.tolist(),
                df_college1['value'].tolist()))
df_age2 = pd.DataFrame(age2)
dic5 = dict(zip(df_age2.index.values.tolist(),
                df_age2['value'].tolist()))
df_under_202 = pd.DataFrame(under_202)
dic6 = dict(zip(df_under_202.index.values.tolist(),
                df_under_202['value'].tolist()))
df_gender2 = pd.DataFrame(gender2)
dic7 = dict(zip(df_gender2.index.values.tolist(),
                df_gender2['value'].tolist()))
df_college2 = pd.DataFrame(college2)
dic8 = dict(zip(df_college2.index.values.tolist(),
                df_college2['value'].tolist()))

df_c = pd.DataFrame([dic2.keys(), dic2.values()]).T
df_c = df_c.append(pd.DataFrame([dic3.keys(), dic3.values()]).T)
df_c = df_c.append(pd.DataFrame([dic4.keys(), dic4.values()]).T)
df_m = pd.DataFrame([dic6.keys(), dic6.values()]).T
df_m = df_m.append(pd.DataFrame([dic7.keys(), dic7.values()]).T)
df_m = df_m.append(pd.DataFrame([dic8.keys(), dic8.values()]).T)
df_fin = df_c.merge(df_m, how='left', on=0)
df_fin['class'] = ['under_20', 'under_20', 'gender',
                   'gender', 'college', 'college']
df_fin.rename(columns={0:'ohx'}, inplace=True)
df_fin.set_index(['class', 'ohx'], inplace=True)
df_fin.columns = ['complete', 'missing']
df_fin1 = df_fin.copy()

c = list(df_fin['complete'])
m = list(df_fin['missing'])
c1, m1 = [], []
for i in range(len(c)):
    c1.append(str(c[i])+' (' +str(round(c[i]/(c[i]+m[i])*100, 1))+'%)')
    m1.append(str(m[i])+' (' +str(round(m[i]/(c[i]+m[i])*100, 1))+'%)')

df_fin['complete'] = c1
df_fin['missing'] = m1

ac, am = [], []
for i in dic1:
    j = 0
    while j < dic1[i]:
        ac.append(i)
        j+=1
for i in dic5:
    j = 0
    while j < dic5[i]:
        am.append(i)
        j+=1
acav = np.mean(ac)
amav = np.mean(am)
acsd = np.std(ac, ddof=1)
amsd = np.std(am, ddof=1)
acstr = str(round(acav, 2))+ ' (SD='+str(round(acsd, 2))+')'
amstr = str(round(amav, 2))+ ' (SD='+str(round(amsd, 2))+')'
df_fin.loc[('age', 'age'), :] = acstr, amstr
df_fin = df_fin.reindex([('age', 'age'), ('under_20', 'No college/<20'),
                ('under_20', 'some college/college graduate'),
                ('gender', 'Female'), ('gender', 'Male'), 
                ('college', False), ('college', True)])
df_fin

pvalue=[]
pvalue.append(stats.ttest_ind(ac, am, equal_var=False)[1])
t1=pd.DataFrame([[df_fin1.iat[0,0], df_fin1.iat[0,1]],
                 [df_fin1.iat[1,0], df_fin1.iat[1,1]]])
t2=pd.DataFrame([[df_fin1.iat[2,0], df_fin1.iat[2,1]],
                 [df_fin1.iat[3,0], df_fin1.iat[3,1]]])
t3=pd.DataFrame([[df_fin1.iat[4,0], df_fin1.iat[4,1]],
                 [df_fin1.iat[5,0], df_fin1.iat[5,1]]])
pvalue.append(stats.fisher_exact(t1)[1])
pvalue.append('-')
pvalue.append(stats.fisher_exact(t2)[1])
pvalue.append('-')
pvalue.append(stats.fisher_exact(t3)[1])
pvalue.append('-')
for i in range(len(pvalue)):
    pvalue[i]=str(pvalue[i])
df_fin['p-value']=pvalue

df_fin


# # Question 2 - Monte Carlo Comparison

# In this question you will use your functions from problem set 1, question 3 for construcing binomial confidence intervals for a population proprotion in a Monte Carlo study comparing the performance of the programmed methods.
#
# In the instructions that follow, let n refer to sample size and p to the population proportion to be estimated.
#
# Choose a nominal confidence level of 80, 90 or 95% to use for all parts below.
#
# You may wish to collect your confidence interval functions in a separate file and import them for this assignment. See here for helpful discussion.
#
# Update, October 14 - Make sure to correct any mistakes in your functions from PS1, Q3. It is also acceptable to revise your functions to use vectorized operations to make the Monte Carlo study more efficient.

# ## part a) Level Calibration
# In this part, you will examine whether the nominal confidence level is achieved by each method over a grid of values for n and p. Recall that the confidence level is the proportion of intervals that (nominally should) contain the true population proportion.
#
# Pick a sequence of values to examine for p∈(0,0.5] or p∈[0.5,1) and a sequence of values for n>0. For each combination of n and p use Monte Carlo simulation to estimate the actual confidence level each method for generating intervals achieves. Choose the number of Monte Carlo replications so that, if the nominal level were achieved, the margin of error around your Monte Carlo estimate of the confidence level would be no larger than 0.005.
#
# For each confidence interval method, construct a contour plot (with axes n and p) showing the estimated confidence level. Use subplots to collect these into a single figure.
#
#

def ci_mean(
    x,
    level=0.95,
    str_fmt="{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]"
):
    """
    Construct an estimate and confidence interval for the mean of `x`.

    Parameters
    ----------
    x : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector from which to form the estimates.
    level : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    str_fmt: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]".

    Returns
    -------
    By default, the function returns a string with a 95% confidence interval
    in the form "mean [95% CI: (lwr, upr)]". A dictionary containing the mean,
    confidence level, lower, bound, and upper bound can also be returned.

    """
    # check input
    try:
        x = np.asarray(x)  # or np.array() as instructed.
    except TypeError:
        print("Could not convert x to type ndarray.")
    # construct estimates
    xbar = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(x.size)
    z = norm.ppf(1 - (1 - level) / 2)
    lwr, upr = xbar - z * se, xbar + z * se
    out = {"mean": xbar, "level": 100 * level, "lwr": lwr, "upr": upr}
    # format output
    if str_fmt is None:
        return(out)
    else:
        return(str_fmt.format_map(out))    


def ci_prop(
    x,
    level=0.95,
    str_fmt="{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]",
    method="Normal"
):
    """
    Construct point and interval estimates for a population proportion.

    The "method" argument controls the estimates returned. Available methods
    are "Normal", to use the normal approximation to the Binomial, "CP" to
    use the Clopper-Pearson method, "Jeffrey" to use Jeffery's method, and
    "AC" for the Agresti-Coull method.

    By default, the function returns a string with a 95% confidence interval
    in the form "mean [level% CI: (lwr, upr)]". Set `str_fmt=None` to return
    a dictionary containing the mean, confidence level (%-scale, level),
    lower bound (lwr), and upper bound (upr) can also be returned.

    Parameters
    ----------
    x : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    level : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    str_fmt: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".
    method: str, optional
        The type of confidence interval and point estimate desired.  Allowed
        values are "Normal" for the normal approximation to the Binomial,
        "CP" for a Clopper-Pearson interval, "Jeffrey" for Jeffrey's method,
        or "AC" for the Agresti-Coull estimates.

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.
    """
    # check input type
    try:
        x = np.asarray(x)  # or np.array() as instructed.
    except TypeError:
        print("Could not convert x to type ndarray.")

    # check that x is bool or 0/1
    if x.dtype is np.dtype('bool'):
        pass
    elif not np.logical_or(x == 0, x == 1).all():
        raise TypeError("x should be dtype('bool') or all 0's and 1's.")

    # check method
    assert method in ["Normal", "CP", "Jeffrey", "AC"]

    # determine the length
    n = x.size

    # compute estimate
    if method == 'AC':
        z = norm.ppf(1 - (1 - level) / 2)
        n = (n + z ** 2)
        est = (np.sum(x) + z ** 2 / 2) / n
    else:
        est = np.mean(x)

    # warn for small sample size with "Normal" method
    if method == 'Normal' and (n * min(est, 1 - est)) < 12:
        warn(Warning(
            "Normal approximation may be incorrect for n * min(p, 1-p) < 12."
        ))

    # compute bounds for Normal and AC methods
    if method in ['Normal', 'AC']:
        se = np.sqrt(est * (1 - est) / n)
        z = norm.ppf(1 - (1 - level) / 2)
        lwr, upr = est - z * se, est + z * se

    # compute bounds for CP method
    if method == 'CP':
        alpha = 1 - level
        s = np.sum(x)
        lwr = beta.ppf(alpha / 2, s, n - s + 1)
        upr = beta.ppf(1 - alpha / 2, s + 1, n - s)

    # compute bounds for Jeffrey method
    if method == 'Jeffrey':
        alpha = 1 - level
        s = np.sum(x)
        lwr = beta.ppf(alpha / 2, s + 0.5, n - s + 0.5)
        upr = beta.ppf(1 - alpha / 2, s + 0.5, n - s + 0.5)

    # prepare return values
    out = {"mean": est, "level": 100 * level, "lwr": lwr, "upr": upr}
    if str_fmt is None:
        return(out)
    else:
        return(str_fmt.format_map(out))


z = stats.norm.ppf(.975)
n_min =  (1 / 0.005 * z * 1 * np.sqrt(0.95 / 1 * (1 - 0.95 / 1))) ** 2
n_total = 7300

N = 1000
n = [100, 300, 500, 700, 900]
p = [0.5, 0.6, 0.7, 0.8, 0.9]
n_p = []
for i in n:
    for j in p:
        n_p.append((i,j))

li = []
for i in range(len(n_p)):
    count = 0
    while count < n_total:
        n_ones = np.ones(N)
        n_zeros = N*(1-n_p[i][1])
        n_ones[:int(n_zeros)]=0
        random.shuffle(n_ones)
        n_list = n_ones[0:n_p[i][0]]
        li.append((n_p[i][0], n_p[i][1], n_list))
        count+=1

check_nor, check_i, check_ii, check_iii, check_iv = [], [], [], [], []
for i in li:
    dic1 = ci_mean(i[2], str_fmt = None)
    if (i[1] < dic1['upr'] and i[1] > dic1['lwr']):
        check_nor.append(1)
    else:
        check_nor.append(0)
    dic2 = ci_prop(i[2], method="Normal", str_fmt = None)
    if (i[1] < dic2['upr'] and i[1] > dic2['lwr']):
        check_i.append(1)
    else:
        check_i.append(0)
    dic3 = ci_prop(i[2], method='AC', str_fmt = None)
    if (i[1] < dic3['upr'] and i[1] > dic3['lwr']):
        check_ii.append(1)
    else:
        check_ii.append(0)
    dic4 = ci_prop(i[2], method='CP', str_fmt = None)
    if (i[1] < dic4['upr'] and i[1] > dic4['lwr']):
        check_iii.append(1)
    else:
        check_iii.append(0)
    dic5 = ci_prop(i[2], method='Jeffrey', str_fmt = None)
    if (i[1] < dic5['upr'] and i[1] > dic5['lwr']):
        check_iv.append(1)
    else:
        check_iv.append(0)

c_nor, c_i, c_ii, c_iii, c_iv = [], [], [], [], []
for i in range(25):
    c_nor.append(check_nor[i*n_total:(i+1)*n_total].count(1)/n_total)
    c_i.append(check_i[i*n_total:(i+1)*n_total].count(1)/n_total)
    c_ii.append(check_ii[i*n_total:(i+1)*n_total].count(1)/n_total)
    c_iii.append(check_iii[i*n_total:(i+1)*n_total].count(1)/n_total)
    c_iv.append(check_iv[i*n_total:(i+1)*n_total].count(1)/n_total)

n1, p1 = [], []
for i in range(25):
    n1.append(n_p[i][0])
    p1.append(n_p[i][1])
txt=['p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9']*5

_ = plt.figure(figsize=(10, 20))
_ = plt.subplots_adjust(hspace=0.5)
ax1 = plt.subplot(511)
_ = plt.scatter(
    x=n1,
    y=c_nor,
)
_ = ax1.set_ylabel('Mean')
ax2 = plt.subplot(512)
_ = plt.scatter(
    x=n1,
    y=c_i,
)
_ = ax2.set_ylabel('Normal')
ax3 = plt.subplot(513)
_ = plt.scatter(
    x=n1,
    y=c_ii,
)
_ = ax3.set_ylabel('AC')
ax4 = plt.subplot(514)
_ = plt.scatter(
    x=n1,
    y=c_iii,
)
_ = ax4.set_ylabel('CP')
ax5 = plt.subplot(515)
_ = plt.scatter(
    x=n1,
    y=c_iv,
)
_ = ax5.set_ylabel('Jeffrey')
fig.tight_layout()

# ## part b) Relative Efficiency
# As part of your simulation for part a, record the widths of the associated confidence intervals. Estimate the average width of intervals produced by each method at each level of n and p and use a collection of contour plots to visualize the results. Finally, using the Clopper-Pearson method as a reference, estimate the average relative width (at each value of n and p) and display these results using one more countour plots.

CI_nor, CI_i, CI_ii, CI_iii, CI_iv = [], [], [], [], []
for i in li:
    dic1 = ci_mean(i[2], str_fmt = None)
    CI_nor.append(dic1['upr']-dic1['lwr'])
    dic2 = ci_prop(i[2], method="Normal", str_fmt = None)
    CI_i.append(dic2['upr']-dic2['lwr'])
    dic3 = ci_prop(i[2], method='AC', str_fmt = None)
    CI_ii.append(dic3['upr']-dic3['lwr'])
    dic4 = ci_prop(i[2], method='CP', str_fmt = None)
    CI_iii.append(dic4['upr']-dic4['lwr'])
    dic5 = ci_prop(i[2], method='Jeffrey', str_fmt = None)
    CI_iv.append(dic5['upr']-dic5['lwr'])

CI_norag, CI_iag, CI_iiag, CI_iiiag, CI_ivag = [], [], [], [], []
for i in range(25):
    CI_norag.append(np.mean(CI_nor[i*n_total:(i+1)*n_total]))
    CI_iag.append(np.mean(CI_i[i*n_total:(i+1)*n_total]))
    CI_iiag.append(np.mean(CI_ii[i*n_total:(i+1)*n_total]))
    CI_iiiag.append(np.mean(CI_iii[i*n_total:(i+1)*n_total]))
    CI_ivag.append(np.mean(CI_iv[i*n_total:(i+1)*n_total]))

_ = plt.figure(figsize=(10, 20))
_ = plt.subplots_adjust(hspace=0.5)
ax1 = plt.subplot(511)
_ = plt.scatter(
    x=n1,
    y=CI_norag,
)
_ = ax1.set_ylabel('Mean')
ax2 = plt.subplot(512)
_ = plt.scatter(
    x=n1,
    y=CI_iag,
)
_ = ax2.set_ylabel('Normal')
ax3 = plt.subplot(513)
_ = plt.scatter(
    x=n1,
    y=CI_iiag,
)
_ = ax3.set_ylabel('AC')
ax4 = plt.subplot(514)
_ = plt.scatter(
    x=n1,
    y=CI_iiiag,
)
_ = ax4.set_ylabel('CP')
ax5 = plt.subplot(515)
_ = plt.scatter(
    x=n1,
    y=CI_ivag,
)
_ = ax5.set_ylabel('Jeffrey')
fig.tight_layout()

X, Y = np.meshgrid([np.log(i) for i in n1], CI_norag)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig1, ax1 = plt.subplots()
CS1 = ax1.contour(X, Y, Z)
_ = ax1.set_xlim(4, 6)

X, Y = np.meshgrid([np.log(i) for i in n1], CI_iag)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig2, ax2 = plt.subplots()
CS2 = ax2.contour(X, Y, Z)
_ = ax2.set_xlim(4, 6)

X, Y = np.meshgrid([np.log(i) for i in n1], CI_iiag)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig3, ax3 = plt.subplots()
CS3 = ax3.contour(X, Y, Z)
_ = ax3.set_xlim(4, 6)

X, Y = np.meshgrid([np.log(i) for i in n1], CI_iiiag)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig4, ax4 = plt.subplots()
CS4 = ax4.contour(X, Y, Z)
_ = ax4.set_xlim(4, 6)

X, Y = np.meshgrid([np.log(i) for i in n1], CI_ivag)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig5, ax5 = plt.subplots()
CS5 = ax5.contour(X, Y, Z)
_ = ax5.set_xlim(4, 6)

cp = []
for i in li:
    a, b = statsmodels.stats.proportion.proportion_confint(
        list(i[2]).count(1), i[0], alpha=0.05, method='beta')
    cp.append(b-a)

CI_cpag = []
for i in range(25):
    CI_cpag.append(np.mean(cp[i*n_total:(i+1)*n_total]))

X, Y = np.meshgrid([np.log(i) for i in n1], CI_cpag)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig6, ax6 = plt.subplots()
CS6 = ax6.contour(X, Y, Z)
_ = ax6.set_xlim(4, 6)
