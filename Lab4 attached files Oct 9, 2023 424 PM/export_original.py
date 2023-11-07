# %% [markdown]
# ### Lab 3_2: Decision Tree

# %%
import pandas as pd
import scipy as sp
import numpy as np
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier

# %% [markdown]
# ## 0. Data Loading

# %%
#load data and extract data
names = ['age', 'workclass', 'fnlwgt', 'edu', 'edu-num', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pd.read_csv('adult.data', names=names)
print( "Total columns: ", len(df.columns))
df.head()                 # display all columns
#df[df.columns[:10]].head() # display the first 10 columns

# %%
df.info()

# %%
from sklearn import preprocessing

# it is required that all feature/target values be numerical
# Systematically convert all string (labeled as object) type into labels(1,2,3,...)
label_encoding = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = label_encoding.fit_transform(df[column_name])
    else:
        pass

# %%
# extract X, y
y = df['income']      # two labels: <=50K, >50K
X = df.drop('income', axis=1)

# %% [markdown]
# ## 1. Decision Tree With Gini Index

# %%
# your code below

# create a decision tree classifier (gini, max_depth=2)


# %%
# generate the image


# %%
# display the image here

# %% [markdown]
# ## 2. Decision Tree With Entropy

# %%
# your code below

# create a decision tree classifier (entropy, max_depth=2)


# %%
# generate the image


# %%
# display the image here

# %%



