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
