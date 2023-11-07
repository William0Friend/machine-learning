# %% [markdown]
# # Lab 6: Ensemble Learning

# %%
import pandas as pd
import scipy as sp
import numpy as np
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print('Train/Test sizes:', y_train.shape[0], y_test.shape[0])

# %% [markdown]
# ## Decision Tree

# %%
dt_clf=DecisionTreeClassifier(random_state=1)
dt_clf.fit(X_train, y_train)
accuracy_score(dt_clf.predict(X_test), y_test)

# %% [markdown]
# #### Important:  You should NOT modify code above this line

# %% [markdown]
# ## 1. Majority Voting

# %%
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# your code below



# %% [markdown]
# ## 2. Bagging

# %%
from sklearn.ensemble import BaggingClassifier

# your code below



# %% [markdown]
# ## 3. Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# your code below




# %% [markdown]
# ## 4. AdaBoost

# %%
from sklearn.ensemble import AdaBoostClassifier

# your code below




