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
from statsmodels.tsa.seasonal import seasonal_decompose

# Stats
from scipy.stats.mstats import winsorize
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

# ----------------------------------------------------------------------------------
def bivariate_boxplot(df, numerical_cols, categorical_cols):
    """
    Creates boxplots to explore bivariate relationships between 'median_sale_price'
    and other columns. The function does two things:

    1) For numeric columns in numerical_cols:
       - Segments 'median_sale_price' into four quartile bins ('Low', 'Medium', 'High', 'Very High')
         using pd.qcut and stores this in a new column, 'price_bin'.
       - Plots a boxplot with 'price_bin' on the x-axis and each numeric column on the y-axis.
         This lets you see how those numeric features vary across different price tiers.

    2) For categorical columns in categorical_cols:
       - Plots a boxplot with the categorical column on the x-axis and 'median_sale_price' on the y-axis,
         but only if that categorical column has fewer than 10 unique values (to avoid clutter).
       - This helps you see how the median sale price distribution differs by category.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing your housing data, including 'median_sale_price'.
    numerical_cols : list of str
        List of numeric feature columns to explore relative to 'price_bin'.
    categorical_cols : list of str
        List of categorical feature columns. For each column with fewer than 10 unique categories,
        a boxplot is created comparing 'median_sale_price' across the categories.

    Returns:
    --------
    None
        Displays the generated boxplots.
    """

    # Create quartile-based 'price_bin' for the target column
    df['price_bin'] = pd.qcut(df['median_sale_price'], q=4, 
                              labels=['Low','Medium','High','Very High'])
    
    # 1) Numeric columns: Boxplot with price_bin on x-axis, numeric col on y-axis
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='price_bin', y=col, data=df)
        plt.title(f'{col} by Price Bin')
        plt.xlabel('Price Bin')
        plt.ylabel(col)
        plt.show()

    # 2) Categorical columns: Boxplot with categorical col on x-axis, median_sale_price on y-axis
    for col in categorical_cols:
        if df[col].nunique() < 10:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=col, y='median_sale_price', data=df)
            plt.title(f'Median Sale Price by {col}')
            plt.xticks(rotation=45)
            plt.xlabel(col)
            plt.ylabel('Median Sale Price')
            plt.show()
# ----------------------------------------------------------------------------------
def bivariate_scatterplot(df, numerical_cols, categorical_cols):
    """
    Creates scatterplots for each numeric feature (except 'median_sale_price') 
    against the target 'median_sale_price', colored by 'region'.
    
    Additionally, assigns 'price_bin' to df (via pd.qcut) to create quartiles 
    of 'median_sale_price' (Low, Medium, High, Very High). This is useful 
    if you later want to compare other variables by price tier or do a 
    categorical analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing your housing data.
    numerical_cols : list of str
        List of numeric columns to use for the scatterplots. Each column 
        in this list will be plotted against 'median_sale_price'.
    categorical_cols : list of str
        (Currently unused in this function, but can be integrated later 
        if you want to do bivariate analysis of categorical variables.)

    Returns:
    --------
    None
        Displays one or more scatterplots showing each numeric feature 
        vs. 'median_sale_price', color-coded by 'region'.
    """
    
    # Create a quartile-based categorical variable for 'median_sale_price'
    df['price_bin'] = pd.qcut(df['median_sale_price'], q=4, 
                              labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Generate scatterplots for each numeric column (except 'median_sale_price')
    for col in numerical_cols:
        if col != 'median_sale_price':
            plt.figure(figsize=(12, 10))
            sns.scatterplot(x=col, y='median_sale_price', hue='region', data=df)
            plt.title(f'{col} vs. Median Sale Price')
            plt.xlabel(col)
            plt.ylabel('Median Sale Price')
            plt.show()
# ----------------------------------------------------------------------------------
def plot_average_price_over_time(df, date_col='month_of_period_end', price_col='median_sale_price'):
    """
    Drops any time zone info from the specified date column,
    then groups by month_of_period_end to plot the average median_sale_price over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your DataFrame containing the housing data.
    date_col : str, optional
        The name of the date column (default is 'month_of_period_end').
    price_col : str, optional
        The name of the price column (default is 'median_sale_price').

    Returns:
    --------
    None
        Displays a matplotlib plot of the average price over time.
    """
    
    # Ensure the date column is tz-naive (removes any leftover timezone info)
    df[date_col] = df[date_col].dt.tz_localize(None)
    
    # Group by the (now tz-naive) date column and compute mean price
    price_over_time = df.groupby(date_col)[price_col].mean()
    
    # Plot
    plt.figure(figsize=(10, 5))
    price_over_time.plot()
    
    plt.title('Average Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Median Sale Price')
    plt.show()
    
# ----------------------------------------------------------------------------------
def plot_seasonal_decomposition(df, date_col='month_of_period_end', target_col='median_sale_price',
                                model='multiplicative', period=12):
    """
    Performs and plots a seasonal decomposition of the specified time-series target column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing your time-series data.
    date_col : str, optional
        The name of the datetime column to set as index (default 'month_of_period_end').
    target_col : str, optional
        The name of the column to decompose (default 'median_sale_price').
    model : {'additive', 'multiplicative'}, optional
        The type of seasonal component (default 'multiplicative').
    period : int, optional
        The seasonal period. For monthly data, 12 is common. (default 12)

    Returns:
    --------
    None
        Displays a decomposition plot with trend, seasonal, and residual components.
    """
    
    # Ensure the date_col is set as the DataFrame index
    df_ts = df.set_index(date_col).sort_index()
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df_ts[target_col], model=model, period=period)
    
    # Plot the results
    decomposition.plot()
    plt.show()
# ----------------------------------------------------------------------------------
def plot_correlation_heatmap(df, columns=None, figsize=(16, 12), cmap='coolwarm'):
    """
    Plots a correlation matrix heatmap for the specified columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing your data.
    columns : list of str, optional
        A list of column names to include in the correlation matrix. 
        If None, uses all columns. (default None)
    figsize : tuple, optional
        The size of the figure, e.g. (width, height). (default (16, 12))
    cmap : str, optional
        The color map to use for the heatmap. (default 'coolwarm')

    Returns
    -------
    None
        Displays a heatmap of the correlation matrix.
    """
    
    # If specific columns are provided, use them; otherwise, use all columns
    if columns is not None:
        data = df[columns]
    else:
        data = df
    
    # Compute correlation matrix
    corr = data.corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap=cmap)
    plt.title('Correlation Matrix')
    plt.show()
    
# ----------------------------------------------------------------------------------
    
'''
*------------------*
|                  |
|     OUTLIERS     |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
def detect_outliers_iqr(df):
    """
    Identifies outliers using the IQR method.

    Parameters:
        df (pd.DataFrame): The dataset to analyze.

    Returns:
        pd.Series: A count of outliers for each numerical column.
    """
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Compute IQR for each numerical column
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers using the 1.5*IQR rule
    outliers = ((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).sum()

    return outliers

# ----------------------------------------------------------------------------------

def visualize_outliers(df):
    """
    Generates boxplots for visualizing outliers in selected numerical columns.

    Parameters:
        df (pd.DataFrame): The dataset to analyze.
        outlier_cols (list): List of column names to visualize.

    Returns:
        None (Displays the boxplot)
    """
    plt.figure(figsize=(12, 8))
    df[outlier_cols].boxplot(rot=45)
    plt.title("Boxplot of Columns with Significant Outliers")
    plt.show()
    
# ----------------------------------------------------------------------------------
def apply_winsorization(df, outlier_cols, limits=(0.05, 0.05)):
    """
    Applies Winsorization (capping extreme values) to specified columns and visualizes the results.

    Parameters:
        df (pd.DataFrame): The dataset to modify.
        outlier_cols (list): List of columns to apply Winsorization.
        limits (tuple): Percentage of values to cap on both ends (default: 5% on both sides).

    Returns:
        pd.DataFrame: A new DataFrame with Winsorized values.
    """
    df_winsorized = df.copy()
    
    for col in outlier_cols:
        df_winsorized[col] = winsorize(df[col], limits=limits)  # Apply Winsorization
    
    # Generate boxplot to confirm changes
    plt.figure(figsize=(12, 8))
    df_winsorized[outlier_cols].boxplot(rot=45)
    plt.title("Boxplot After Winsorization")
    plt.show()
    
    return df_winsorized

# ----------------------------------------------------------------------------------
def plot_correlation_heatmap(df, figsize=(12, 8), cmap="coolwarm"):
    """
    Generates a heatmap to visualize the correlation matrix of a dataset.

    Parameters:
        df (pd.DataFrame): The dataset to analyze.
        figsize (tuple): Figure size (default: (12, 8)).
        cmap (str): Colormap for visualization (default: "coolwarm").

    Returns:
        None (Displays the heatmap)
    """
    correlation_matrix = df.corr()  # Compute correlation matrix
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()

# ----------------------------------------------------------------------------------
'''
*------------------*
|                  |
|   FEATURE ENG    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
def create_features(df):
    """
    Performs feature engineering by creating a market demand ratio 
    and categorizing days on market.

    Parameters:
        df (pd.DataFrame): The dataset to modify.

    Returns:
        pd.DataFrame: A new DataFrame with additional engineered features.
    """
    df = df.copy()
    
    # Create market demand ratio (homes sold per new listing)
    df["market_demand_ratio"] = df["homes_sold"] / df["new_listings"]
    
    # Categorize days on market into bins
    df["days_on_market_category"] = pd.cut(
        df["days_on_market"],
        bins=[0, 15, 30, 100],
        labels=["Fast", "Moderate", "Slow"]
    )
    
    return df

# ----------------------------------------------------------------------------------
def drop_redundant_features(df, columns_to_drop):
    """
    Drops specified redundant features from the dataset.

    Parameters:
        df (pd.DataFrame): The dataset to modify.
        columns_to_drop (list): List of column names to remove.

    Returns:
        pd.DataFrame: A new DataFrame with selected columns removed.
    """
    df_reduced = df.drop(columns=columns_to_drop, errors="ignore")  # Avoids errors if column not found
    return df_reduced