#!/usr/bin/env python
# coding: utf-8

# ## TRM_genes_arrays 
# 
# - Importing files and produced a join file with ['GeneSymbol', 'mRNA_Accession', 'adj.P.Val', 'logFC'] information per file

# ### Notes
# 
# - Per file: We are going to have one unique row per GeneSymbol, mean for numeric and NM_ for mRNA_Accession. We do not want EMSM (Ensamble).
# 
# - We are going to do an inner/outer join 

# In[1]:


#============================================================
# Read me
#============================================================
# TRM_genes_arrays.py
# Author: Yesika Contreras
#  
# This code generated a dataframe from a list of datasets to be used in the modeling part
#
# 
# python scripts generated 06-29-2019
# Modifications on 09-26-2019

import datetime
d = datetime. datetime. today()
print(d.strftime('%m-%d-%Y'))


# ## Importing packages

# In[2]:


#============================================================
# Packages
#============================================================

import pandas as pd
import numpy as np
import os # Accesing to directory
import re # Regular Expressions
from six.moves import reduce # Merge dataframes


## Setting the seed value for reproducibility

seed_value= 123# Set a seed value

# Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

seed =  np.random.RandomState(123)
# do not call numpy.random.rand() but seed.rand()


# 3. Set environment
os.urandom(seed_value)


# ## Defining Functions
# 
# - Defining function to be used in this script

# ### Importing datasets

# In[3]:


#============================================================
# Importing datasets
#============================================================


def set_directory(path_1):
    ''' 
    set working directory: 
    ''' 
    os.chdir(path_1) 
    path_1 = os.getcwd()
    return path_1

def list_of_datasets(path, columns_to_keep):
    '''
    Getting a list of filenames 
    (We dont need the full path as we set the directory). 
    input Files are saved as '.txt'
    Converting the files into dataframes snd saving them as 
    a list of dataframes.
    and adding the file name at the end of every selected column name
    '''
    # Getting a list of filenames
    all_files = [x for x in os.listdir() if x.endswith(".txt")]
    print('List of files:')
    print(all_files)
    print('------- \n ')

    # Converting the files into dataframes snd saving them as 
    # a list of dataframes.
    list_of_datasets = [pd.read_csv(filename, delimiter="\t", 
                                    usecols= columns_to_keep).add_suffix('_' + filename) 
                   for filename in all_files]
    
    #Verifying data type
    print('data type per file:')
    for file in list_of_datasets:
        print( file.dtypes)
        print('-------')
    # print(list_of_dfs[0])
    
    return list_of_datasets


def dataset_size(list_of_datasets):
    for file in list_of_datasets:
        print('File name: ' + str(file.columns[0][11:]) + 
              ', ', 'Total Observations: ' +  str(file.size/4))
    print('-------')
    ## Initial Total size of observations in the list of datasets:
    print('Total size of observations in the list of datasets: given as '
          +' (observations, columns)')
    print(pd.concat(list_of_datasets, sort=False).shape )
    print('-------  \n ')


# ### Cleaning files

# In[4]:


#============================================================
# Cleaning files
#============================================================

def GeneSymbol_remove_multiple_values(column):
    ''' Selecting one GeneSymbol when more than one is provided by 
    record and the separator is: /// 
        Input example: Srp54c///Srp54b///Srp54a
        Rule: select a GeneSymbol that do not contain LOC, GM, #RIK
    '''
#     if re.findall("[///]", column):
    if '///' in column: 
        column = str(column)
        GeneSymbol_list = re.split(r'///', column)  
        patterns = ['LOC', 'GM', '#RIK']
        omit =[]
        result = []
        for i in GeneSymbol_list:
            if re.search(r"LOC", i ): omit.append(i)
            elif re.search(r"GM", i ): omit.append(i)
        result = sorted(list(set(GeneSymbol_list) - set(omit)))
        if result == []: record = GeneSymbol_list[0]
        else: record = result[0] 
    else:
        record = column
    return record

# str1 = 'LOC///LOC'
# print(GeneSymbol_remove_multiple_values3(str1))


def cleaning_dataframe(df):
    '''
    input/output file is a dataframe'''
    for column in df.columns:
#         if column in ['GeneSymbol', 'mRNA_Accession']:
        if re.search('^GeneSymbol|^mRNA_Accession', column):
            #print (column)
            
            # Remove dupplicate string in observation per column. 
            # Example: Srp54c///Srp54b///Srp54a///
            df[column] = df[column].astype(str).apply(GeneSymbol_remove_multiple_values)
            
            # Remove dupplicate string in observation per column. Example: Emp1 // Emp1
            df[column] = df[column].str.split(" //", expand=True)[0]
            
            # remove white space
            df[column] = df[column].str.strip()
            
            # Replace '---' & '0' with NaN
            df[column].replace('---',np.nan, inplace=True)
            df[column].replace('0',np.nan, inplace=True)
            df[column].replace('',np.nan, inplace=True)
            df[column].replace('nan',np.nan, inplace=True)
    
    return df


# ### Identifying Missing values

# In[5]:


#============================================================
# Identifying Missing values
#============================================================

def percentage_missing_values(list_of_datasets):
    '''
    Identifying percentage of missing values given a list of dataframes
    '''
    for file in list_of_datasets:
        print ( round(file.isna().sum() *100 / (file.size/4) ,2) ) 
        print('---------')
        
        
#Removing missing values for the GeneSymbol columns.
def drop_missing_values(list_of_datasets):
    '''
    If there are missing values on the GeneSymbol, then we drop the row. 
    Otherwise we keep the row.
    '''
    for file in list_of_datasets:
        file.dropna(subset = [file.filter(regex='^GeneSymbol',
                                          axis=1).columns[0]], inplace=True)
#     file.dropna(inplace=True)
    return list_of_datasets


# Imputing missing values
def impute_missing_values(list_of_datasets):
    '''
    If there are missing values on the numeric fields, then impute with the mean.
    If there are missing values on the categorical fields, then impute with the mode.
    '''
    for file in list_of_datasets:

        GeneSymbol_col = file.filter(regex='^GeneSymbol',axis=1).columns[0]
        mRNA_col = file.filter(regex='^mRNA_Accession',axis=1).columns[0]
        adj_P_Val_col = file.filter(regex='^adj.P.Val',axis=1).columns[0]
        logFC_col = file.filter(regex='^logFC',axis=1).columns[0]
        
        file[mRNA_col] = file.groupby(GeneSymbol_col)[mRNA_col].transform(
            lambda x: x.fillna(x.mode().get(0,'NaN/#N/A')))
        file[adj_P_Val_col] = file.groupby(GeneSymbol_col)[adj_P_Val_col].transform(
            lambda x: x.fillna(x.mean()))
        file[logFC_col] = file.groupby(GeneSymbol_col)[logFC_col].transform(
            lambda x: x.fillna(x.mean()))

    return list_of_datasets


# ### Removing Duplicate Records

# In[6]:


#============================================================
# Removing duplicate records (Duplicate rows having the same GeneSymbol)
#============================================================

## Removing duplicate records (rows)
def duplicate_rows_one_column (data):
    '''
    Remove duplicate rows having the same GeneSymbol
    Calculate mean for numeric columns
    Maintain mRNA_Accession NM
    input/output file is a dataframe
    '''
    df_duplicate_rows = data.groupby([data.filter(regex='^GeneSymbol',
                                                  axis=1).columns[0]], 
                                     as_index=False).aggregate({data.filter(regex='^mRNA_Accession',axis=1).columns[0]: 'max', 
                                                                data.filter(regex='^adj.P.Val',axis=1).columns[0]: 'mean',
                                                                data.filter(regex='^logFC',axis=1).columns[0]: 'mean'})
    return df_duplicate_rows
# data.filter(regex='^mRNA_Accession',axis=1).columns[0]: pd.Series.mode,


# ### Merging Multiple dataframes

# In[7]:


#============================================================
# Joinining Dataframes
#============================================================

## Merging the files using merge and reduce function and after compiling the list of dataframes to merge.
# To keep the values that belong to the same gene symbol we need to merge it on the GeneSymbol. 
# We are doing an outer join (NAs will be added)


def merging_list_of_datasets(list_of_datasets, join_type = 'outer'):

    df_merged = reduce(lambda  left, right: pd.merge(left, right, 
                                                     left_on = left.filter(regex='^GeneSymbol',axis=1).columns[0] , 
                                                     right_on= right.filter(regex='^GeneSymbol',axis=1).columns[0],
                                                     how = join_type), 
                       list_of_datasets)

    return df_merged


#If column multi indexes 
#(it was injecting the 'on' as a column which worked for the first merge, but subsequent merges failed), 
#instead I got it to work with: 
#df = reduce(lambda left, right: left.join(right, how='outer', on='Date'), dfs) 


# ### Retriving Files

# In[8]:


#============================================================
# Retriving Files
#============================================================

def output_file(output_file, output_path, output_file_name):
    '''
    Retrive a single file, 
    inputs output_file: dataframe, 
    output_path:folder location, 
    output_file_name: file name with .txt extension
    '''
    output_file.to_csv(os.path.join(output_path, output_file_name), sep='\t')
    
def output_list_of_datasets(output_list_of_datasets, output_path, output_file_name):
    '''
    Retrive a list_of_datasets, 
    inputs output_file: list_of_datasets, 
    output_path:folder location, 
    output_file_name: file name with .txt extension
    '''
    pd.concat(output_list_of_datasets, sort=True).to_csv(os.path.join(output_path, output_file_name), sep='\t')
    


# In[ ]:





# ## Runing Main Functions
# 
# - User input information manually
# - Computation and outputs generated

# In[9]:


#============================================================
# User input:
#============================================================

#path = input("Enter the folder location of the dataset files: \n: ")
path = '../TRM_GeneArrays/input_files'

#columns_to_keep = input("Enter the name of the columns to keep for the GPL file: \n: ")
columns_keep = ['GeneSymbol', 'mRNA_Accession', 'adj.P.Val', 'logFC']

#output_path = input('\nEnter the location where you want to store the output file:\n ')
output_path = '../TRM_GeneArrays/output_files'


# In[10]:


#============================================================
# Computation Importing datasets
#============================================================

# setting working directory:        
path = set_directory(path)

# Converting files into dataframes 
list_of_dfs = list_of_datasets(path, columns_keep)

# Original datasets sizes
print('\nOriginal dataset size: ')
dataset_size(list_of_dfs)


# In[11]:


#============================================================
# Computation Cleaning datasets
#============================================================
print('Checking values before cleaning: ')
print(list_of_dfs[1]['GeneSymbol_GSE47045_GEO2R_TRM_v_TN_Carbone.txt'][34732])

for file in list_of_dfs:
    cleaning_dataframe(file)

print('\nChecking values after cleaning: ')   
print(list_of_dfs[1]['GeneSymbol_GSE47045_GEO2R_TRM_v_TN_Carbone.txt'][34732])
print('---------')


# In[12]:


#============================================================
# Computation Identifying Missing values
#============================================================

## Percentage of missing values
print('\nPercentage of missing values: ')
percentage_missing_values(list_of_dfs)

#  Let's figure out which values are missing!
print('\nExample of missing values for GeneSymbol_GSM2386506_Kupper: ')
print( list_of_dfs[0][ list_of_dfs[0]['GeneSymbol_GSM2386506_Kupper.txt'].isnull() ].head(3) )
print('---------')

# Removing missing values for the GeneSymbol columns.
drop_missing_values(list_of_dfs)
list_of_dfs = impute_missing_values(list_of_dfs)

## Checking the new Percentage of missing values
print('\nNew percentage of missing values: ')
percentage_missing_values(list_of_dfs)


# Datasets size after removing missing values:
print('\nDatasets size after removing missing values: ')
dataset_size(list_of_dfs)

# Saving the list of datasets  after the cleaning step
file_name_cleaned = 'cleaned_files.txt'
output_list_of_datasets(list_of_dfs, output_path, file_name_cleaned)


# In[13]:


for file in list_of_dfs:
#     print (file[file.filter(regex='^GeneSymbol',axis=1).columns[0]].astype(object).nunique() )
    print( file[file.filter(regex='^GeneSymbol',axis=1).columns[0]].astype(object).describe() )


# In[14]:


#============================================================
# Computation Removing duplicate records
#============================================================

## creating a new list of dataframes after removing duplicate records
new_list_of_dfs = []
for file in list_of_dfs:
    new_list_of_dfs.append(duplicate_rows_one_column(file))
    
    
# Datasets size after removing duplicates in GeneSymbol
print('\nDatasets size after removing duplicates in GeneSymbol: ')
dataset_size(new_list_of_dfs)

# Saving the list of datasets after removing duplicates
file_name_duplicates= 'removed_duplicates_files.txt'
output_list_of_datasets(new_list_of_dfs, output_path, file_name_duplicates)


# In[15]:


#============================================================
# Computation Merging multiple dataframes
#============================================================

df_merged_outer = merging_list_of_datasets(new_list_of_dfs, join_type = 'outer')
print('\nMerged dataframe with outer join')
print('Total size of observations in the outer dataframe given as (observations, columns): ')
print(df_merged_outer.shape)
display(df_merged_outer.head(3))
print('---------')

df_merged_inner = merging_list_of_datasets(new_list_of_dfs, join_type = 'inner')
print('\nMerged dataframe with inner join')
print('Total size of observations in the inner dataframe given as (observations, columns): ')
print(df_merged_inner.shape)
display(df_merged_inner.head(3))


# In[16]:


# Saving the merged dataframes
outer_file_name = 'df_merged_outer.txt'
inner_file_name = 'df_merged_inner.txt'
output_file(df_merged_outer, output_path, outer_file_name)
output_file(df_merged_inner, output_path, inner_file_name)


# In[ ]:




