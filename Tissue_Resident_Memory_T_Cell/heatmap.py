# model.py

#============================================================
# Read me
#============================================================
# Heatmap.py
# Author: Yesika Contreras
#  
# This code cluster the genesymbols for n files based on the Log Fold Change

# Code adapted from:
# https://cnls.lanl.gov/external/qbio2018/Slides/Cluster_Lab_June18/clustering_lab.pdf
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
# https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
# 
# python scripts generated 10-14-2019


#============================================================
# Packages
#============================================================

import pandas as pd
import numpy as np #operations
import os # Accesing to directory
import re # Regular Expressions
from six.moves import reduce # Merge dataframes
from scipy.cluster.hierarchy import dendrogram, linkage #for dendogram
from scipy.cluster.hierarchy import fcluster #cluster labels
from matplotlib import pyplot as plt # for visualizations
import seaborn as sns # For heatmap

import pickle
import requests
import json

#python3 -m pip install scipy

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

# In[3]:


#============================================================
# Reading Files
#============================================================

def read_files(path_file, skip_rows =0):
    '''
    Function that read the file as dataframe
    Extract the geneSymbol column and the Log Fold Change
    Input: User need to provide the path for file
    Output =  dataframes
    '''
    df_file = pd.read_csv(path_file, delimiter="\t", skiprows= skip_rows)
    
    
    # Removing columns that are not GeneSymbol or logFC 
    list_dropping =[]
    for column in df_file.columns:
        if not re.search('^GeneSymbol|^logFC', column):
            #print (column)
            list_dropping.append(column)
            
    df_file.drop(columns =  list_dropping,  inplace=True)
    
    
    # Maintain only one GeneSymbol column
    list_dropping =[]
    for column in df_file.columns:
        if re.search('^GeneSymbol', column):
            #print (column)
            list_dropping.append(column)  
            
    df_file.drop(columns =  list_dropping[1:],  inplace=True)
    
    old_name = df_file.filter(regex='^GeneSymbol',axis=1).columns[0]
    df_file.rename(columns = { old_name: 'GeneSymbol'},
                   inplace =True)

    return df_file 


def cleaning_dataframe(df_file):
    '''
    input/output file is a dataframe'''
    for column in df_file.columns:
            
            # Replace '---' & '0' with NaN
            df_file[column].replace('---',np.nan, inplace=True)
            df_file[column].replace('0',np.nan, inplace=True)
            df_file[column].replace('',np.nan, inplace=True)
            df_file[column].replace('nan',np.nan, inplace=True)
    
    return df_file


# ### Hierarchical clustering 
# 
# - Agglomerative or bottom-up approach.
# - Data points are clustered using a bottom-up approach starting with individual data points as its own cluster 
# - and then combine clusters based on some similarity measure. 

# ### Clustermap

# cluster heatmaps visualize a hierarchically clustered data matrix using a reordered heatmap with dendrograms in the margin. 
# In addition to coloring cells, cluster heatmaps reorder the rows and/or columns of the matrix based on the results of hierarchical clustering. The hierarchical structure used to reorder the matrix is often displayed as dendrograms in the margins
# 
# columns represent different sample
# rows represent measurement from different genes
# red signified high expression of a gene and blue means lower expression for a gene
# Hierarchical clustering orders the rows and/or the columns based on similarity.
# This meakes it easy to see correlation in the data.
# REading the heatmap:
# vertical columns: these samples express the same genes
# rows: these genes behave the same
# Hierarchical clustering is usually accompanied by a dendrogram. 
# It indicates both the similarity and the order that the clusters were formed.
# 

# In[4]:


def create_clustermap(dataframe, title, output_file = 'static/images', method= 'ward', metric='euclidean'):
    '''
    Create a 
    '''
    import seaborn as sns
    import matplotlib.pyplot as pyplot

    # dataframe = read_files(dataframe)
    # dataframe = cleaning_dataframe(dataframe)
    dataframe = dataframe.set_index('GeneSymbol')
    # mask = dataframe.isnull()

    sns_plot = sns.clustermap(data =dataframe,
                              method='ward', 
                              metric='euclidean', 
                              cmap="RdBu",
                              # mask=mask
                              #linewidth=.05,
                              #row_cluster=True,
                              #col_cluster=False,
                              #linecolor="grey",
                              #standard_scale=0,
                              #square=True,
#                               yticklabels=True, # label all rows in the heatmap
                              #figsize=(12,(.05*(len(dataframe.index))))
                             )
    
#     # Customizing the dimensions:
#     hm_plot = sns_plot.ax_heatmap.get_position()
#     plt.setp(sns_plot.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
#     sns_plot.ax_heatmap.set_position([hm_plot.x0, hm_plot.y0, hm_plot.width*0.5, hm_plot.height])
#     col = sns_plot.ax_col_dendrogram.get_position()
#     sns_plot.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.5, col.height*0.25])

#     sns_plot.savefig("{}.png".format(title))
    sns_plot.fig.suptitle(title, fontsize=16)
    sns_plot.savefig(os.path.join(output_file, title))
    return sns_plot


# ### Returning output dataset

# In[5]:


def output_file(output_file, output_path, output_file_name):
    '''
    Retrive a single file, 
    inputs output_file: dataframe, 
    output_path:folder location, 
    output_file_name: file name with .txt extension
    '''
    output_file.to_csv(os.path.join(output_path, output_file_name), sep='\t')


if __name__ == '__main__':
    # file_path = '../output_files/5_datasets_input/df_merged_inner.txt'
    file_path = '../TopGenes/Top1200.txt'
    output_path = '../output_files/5_datasets_input'
    heatmap_title = 'Heatmap_1200'
    heatmap_method = 'ward'
    heatmap_metric='euclidean'


    print (create_clustermap(dataframe= file_path, 
      title= heatmap_title, 
      output_file = output_path, 
      method= heatmap_method, 
      metric= heatmap_metric))


