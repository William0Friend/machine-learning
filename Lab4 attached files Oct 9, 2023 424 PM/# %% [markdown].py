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
# Create a decision tree classifier with gini criterion and max_depth=2
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=2)
clf_gini.fit(X, y)


# %%
# generate the image
# For visualization purposes, we'll use libraries like graphviz and functions from sklearn.tree
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

dot_data_gini = export_graphviz(clf_gini, out_file=None, filled=True, rounded=True, 
                                feature_names=X.columns, 
                                class_names=["<=50K", ">50K"])
graph_gini = pydotplus.graph_from_dot_data(dot_data_gini)
Image(graph_gini.create_png())



# %%
# display the image here


# %% [markdown]
# ## 2. Decision Tree With Entropy

# %%
# your code below

# create a decision tree classifier (entropy, max_depth=2)
# Create a decision tree classifier with entropy criterion and max_depth=2
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf_entropy.fit(X, y)


# %%
# generate the image
dot_data_entropy = export_graphviz(clf_entropy, out_file=None, filled=True, rounded=True, 
                                   feature_names=X.columns, 
                                   class_names=["<=50K", ">50K"])
graph_entropy = pydotplus.graph_from_dot_data(dot_data_entropy)
Image(graph_entropy.create_png())


# %%
# display the image here

# %%



