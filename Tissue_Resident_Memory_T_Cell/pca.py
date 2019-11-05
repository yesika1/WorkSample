
# coding: utf-8

# In[1]:


#============================================================
# Read me
#============================================================
# pca.py
# Author: Yesika Contreras
#  
# This code calculates the PCA from a collections of dataframes.
# for now we have data from 5 datasets.
#
# 
# python scripts generated 11-05-2019
# Modifications on 09-26-2019

import datetime
d = datetime. datetime. today()
print(d.strftime('%m-%d-%Y'))


# ## Principal Component Analysis (PCA) 

# In[2]:


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
from sklearn.decomposition import PCA # PCA
import numpy # adding arrays 
import matplotlib
import seaborn as sns
import matplotlib.style as style


# In[3]:


#============================================================
# Reading Files
#============================================================

def read_files(df_file, skip_rows =0):
    '''
    Function that read the file as dataframe
    Extract the geneSymbol column and the Log Fold Change
    Input: User need to provide the path for file
    Output =  dataframes
    '''
    df_file = pd.read_csv(df_file, delimiter="\t", skiprows= skip_rows,)
    
    
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


def set_directory(path_1):
    ''' 
    set working directory: 
    ''' 
    os.chdir(path_1) 
    path_1 = os.getcwd()
    return path_1


# In[7]:


def pca_preprocessing(df_file):
    '''
    Operations before running the PCA algorithm
    includes assigning the attribute GeneSymbol as index
    Dropping all the missing values
    transposing the data, we are going to analyze by dataset
    '''
    # Setting index 
    df_file= df_file.set_index('GeneSymbol')
    # dropping missing values if exist
    df_file.dropna(how="all", inplace=True)

    #  Transpose data and split data table into data X and class la = bels y
    df_file_transposed = df_file.transpose() 
    df_file_transposed.reset_index(inplace =True)
    
    X = df_file_transposed.iloc[:,1:].values
    y = df_file_transposed.iloc[:,0].values
    
    # mapping the possible values, as all names are unique using the factorize function
    df_file_transposed['index_map'] = pd.factorize(df_file_transposed['index'])[0] 
    
    ## Standardizing
    # from sklearn.preprocessing import StandardScaler
    # X_std = StandardScaler().fit_transform(X)

    return (df_file_transposed,X)


def scatter_plot_PCA(x,y,c,label, title):
    '''
    x= axis x values, series object
    y= axis y values, series object
    c = # c=setting the color based on the index, series object
    label= label for the data points, series object
    '''
    
    plt.style.use('ggplot') 
    palette = plt.get_cmap('viridis_r')
    sns.set_context('talk')
    
    fig, ax = plt.subplots(figsize=(12,10))
    img = ax.scatter(x =x, y =y, c= c, alpha=0.8, s=100) # s= size of points
#     plt.colorbar(img, ax=ax)
    plt.title(title)

    # adding point labels
    for i, txt in enumerate(label):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()
    return img
    
sns.reset_defaults()
sns.reset_orig()


# In[5]:


#============================================================
# User input:
#============================================================

file_path = '../output_files/5_datasets_input/df_merged_inner.txt'
output_path = '../output_files/5_datasets_input'


# In[6]:


#============================================================
# Computation Importing datasets
#============================================================

df_pca= read_files(file_path)


# ### PCA
# 
# Principal Component Analysis (PCA) is a linear transformation technique used to identify patterns in data and it is useful when the variables within a dataset are highly correlated. 
# 
# PCA allows us to summarize and visualize observations from a multi-dimensional hyperspace by reducing the dimensionality of the data to a lower-dimensional subspace with minimal loss of information. For instance, every attribute in a dataset could be considered as a different dimension, and PCA identify the multiple inter-correlated quantitative variables to transform them and express their content as a set of few new attributes called principal components. This components explaining most of the variance in the original variables, where the information in a given dataset corresponds to the total variation it contains.
# 
# Eigenvectors are the principal components and determine the directions of the new feature space, and each of those eigenvectors is associated with an eigenvalue which can be interpreted as the "length" or "magnitude" of the corresponding eigenvector, where the higher the value the more information captured about the distribution of the data( measure the amount of variation retained along the new feature axes).
# 
# #### A Summary of the PCA Approach
# 
# PCA aims to reduce the dimensions of a 𝑑-dimensional dataset by projecting it onto a (𝑘)-dimensional subspace (where 𝑘<𝑑) in order to increase the computational efficiency while retaining most of the information. 
# 
# 1. Standardize the data. (mean=0 and variance=1). 
# 2. Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition. PCA yields a feature subspace that maximizes the variance along the axes.
# While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, most PCA implementations perform a Singular Vector Decomposition (SVD) to improve the computational efficiency. 
# 3. Sort eigenvalues in descending order and choose the 𝑘 eigenvectors that correspond to the 𝑘 largest eigenvalues where 𝑘 is the number of dimensions of the new feature subspace (𝑘≤𝑑)/.
# An eigenvalue > 1 indicates that PCs account for more variance than accounted by one of the original variables in standardized data. This is commonly used as a cutoff point for which PCs are retained or another % decided.
# 4. Construct the projection matrix 𝐖 from the selected 𝑘 eigenvectors.
# 5. Transform the original dataset 𝐗 via 𝐖 to obtain a 𝑘-dimensional feature subspace 𝐘.
# 
# ***References:*** [ploty](https://plot.ly/python/v3/ipython-notebooks/principal-component-analysis/)
# [kassambara](http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/112-pca-principal-component-analysis-essentials/)

# In[8]:


df_pca, X = pca_preprocessing(df_pca)
df_pca


# ### Identifying the number of Principal Components

# In[9]:


my_model = PCA(n_components=5)
X_pca = my_model.fit_transform(X)


# In[10]:


my_model = PCA().fit(X)

# single value decomposition (SVD)  to compute eigenvectors
print (my_model.explained_variance_) #returns a vector of the variance explained by each dimension.
print (my_model.explained_variance_ratio_)  # returns variance explained solely by the i+1st dimension.
print (my_model.explained_variance_ratio_.cumsum()) #returns the cumulative variance explained by the first i+1 dimensions. (thats how we pick the n to retain 93.3% of the variance.)

plt.style.use('ggplot') #seaborn-whitegrid
palette = plt.get_cmap('viridis_r')
sns.set_context('talk')

plt.figure(figsize=(10,7))
plt.xlim(0, 29)
plt.yticks(np.arange(0, 1.1, 0.1))
var= numpy.append([0],my_model.explained_variance_ratio_.cumsum()) # start from 0
plt.plot(var, color='k', lw=2)
plt.xlabel('Number of Components')
plt.ylabel('% Explained variance')
plt.title('PCA Analysis \n' +'Variance Ratio Cutoff')
plt.axvline(3, c='b')
plt.axhline(0.89, c='r')

plt.show();


# In[11]:


# my_model_full = sklearnPCA(n_components='mle', svd_solver='full')
# # Minka's maximum-likelihood estimation-will guess the number of 
# principal components.
# n_components='mle' is only supported if n_samples >= n_features
# Y_my_model_full = my_model_full.fit_transform(X)


# ### PCA with 2 Components

# In[12]:


np.random.seed(0)
my_model = PCA(n_components=2,random_state = 17)
X_pca = my_model.fit_transform(X)

## plot
scatter_plot_PCA(x=X_pca[:, 0], 
                 y= X_pca[:, 1], 
                 c= df_pca['index_map'],
                 label = df_pca['index'],
                 title= 'PCA projection, k=2')


# ### t-SNE method

# t-Distributed Stochastic Neighbor Embedding (t-SNE) is a is a non-linear dimensionality reduction technique used to represent high-dimensional dataset in a low-dimensional space of two or three dimensions. t-SNE creates a reduced feature space where similar samples are modeled by nearby points and dissimilar samples are modeled by distant points with high probability.
# 
# In the distribution, the points with the smallest distance with respect to the current point have a high likelihood, whereas the points far away from the current point have very low likelihoods.
# 
# #### A Summary of the t-SNE pproach
# 
# 1. Calculate the probability of similarity of points in high-dimensional space and calculating the probability of similarity of points in the corresponding low-dimensional space. 
# The similarity of points is calculated as the conditional probability that a point A would choose point B as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian (normal distribution) centered at A.
# 2. Minimize the difference between these conditional probabilities (or similarities) in higher-dimensional and lower-dimensional space for a perfect representation of data points in lower-dimensional space.
# To measure the minimization of the sum of difference of conditional probability t-SNE minimizes the sum of Kullback-Leibler divergence of overall data points using a gradient descent method.
# 
# ***References:***
# [Maklin](https://towardsdatascience.com/t-sne-python-example-1ded9953f26)
# [Violante](https://www.kdnuggets.com/2018/08/introduction-t-sne-python.html)
# [datacamp](https://www.datacamp.com/community/tutorials/introduction-t-sne)

# In[14]:


# Invoke the TSNE method
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000,
                  random_state = 17)
X_tsne = tsne_model.fit_transform(X)

scatter_plot_PCA(x=X_tsne[:, 0], 
             y= X_tsne[:, 1], 
             c= df_pca['index_map'],
             label = df_pca['index'],
             title= 'TSNE projection')


# In[ ]:


print(tsne_model)


# In[15]:


tsne_model = TSNE(n_components=2, init='pca',random_state = 17)
X_tsne = tsne_model.fit_transform(X)

scatter_plot_PCA(x=X_tsne[:, 0], 
             y= X_tsne[:, 1], 
             c= df_pca['index_map'],
             label = df_pca['index'],
             title= 'TSNE projection')


# In[16]:


print(tsne_model)


# In[20]:


## Alternative plot

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 5)

sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=df_pca['index'], legend='full', palette=palette)

