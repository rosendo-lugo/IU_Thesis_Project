'''
*------------------*
|                  |
|     EXPLORE      |
|                  |
*------------------*
'''

 # ----------------------------------------------------------------------------------  

# standard imports
import pandas as pd
import numpy as np

# visualized your data
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind, levene
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

print('imports loaded successfully, awaiting commands...')

'''
*------------------*
|                  |
|     SUMMARY      |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# a function that show a summary of the dataset
def data_summary(df):
    # Print the shape of the DataFrame
    print(f'data shape: {df.shape}')
    # set all the columns names to a lowercase
    df.columns = df.columns.str.lower()
    # Separate numeric and non-numeric describes, merging them later:
    desc_numeric = df.select_dtypes(include='number').describe()
    desc_object  = df.select_dtypes(exclude='number').describe(datetime_is_numeric=False)
    # Create a summary DataFrame
    summary = pd.DataFrame(df.dtypes, columns=['data type'])
    # Calculate the number of missing values
    summary['#missing'] = df.isnull().sum().values 
    # Calculate the percentage of missing values
    summary['%missing'] = df.isnull().sum().values / len(df)* 100
    # Calculate the number of unique values
    summary['#unique'] = df.nunique().values
    # Create a descriptive DataFrame
    desc = pd.DataFrame(df.describe(include='all', datetime_is_numeric=True).transpose())
    # Add the minimum, maximum, and first three values to the summary DataFrame
    summary['count'] = desc['count'].values
    summary['mean'] = desc['mean'].values
    summary['std'] = desc['std'].values
    summary['min'] = desc['min'].values
    summary['25%'] = desc['25%'].values
    summary['50%'] = desc['50%'].values
    summary['75%'] = desc['75%'].values
    summary['max'] = desc['max'].values
    # summary['head(1)'] = df.loc[0].values
    # summary['head(2)'] = df.loc[1].values
    # summary['head(3)'] = df.loc[2].values
    
    # Return the summary DataFrame
    return summary