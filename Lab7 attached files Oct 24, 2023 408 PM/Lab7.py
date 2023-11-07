# %% [markdown]
# # Lab 7: Model Evaluation

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
#df[df.columns[:5]].head() # display the first 5 columns

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

y.value_counts()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print('Train/Test sizes:', y_train.shape[0], y_test.shape[0])

# %% [markdown]
# ## 1. Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

random_forest_clf = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=-1)

# %%


# %% [markdown]
# ## 2 Confusion matrix, precision, recall, f1-score, accuracy

# %%
# use 3-fold cross validation to predict y labels on the training dataset
# the predicted labels should be used 
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(random_forest_clf, X_train, y_train, cv=3)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# your code below for confusion matrix, precision, recall, f1-score, accuracy
# start of your code (don't modify code outside your code block)





# end of your code (don't modify code outside your code block)

# %% [markdown]
# ## 3 ROC curve and AUC score

# %%
import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.show()

# %%
# use 3-fold cross validation to predict y probabilities (for each class) on the training dataset
y_train_prob = cross_val_predict(random_forest_clf, X_train, y_train, cv=3, method="predict_proba")

#Important: 
# y_train_prob is numpy 2-d array, where each row is the predicted probablities for each class
# in this dataset we have two classes (<=50K, and >50K), so we have two probability scores
# e.g., the first row (0.76, 0.24) means the probability predicting <=50K is .76, 
#       and the probability predicting >50K is .24, 
#       for ROC curve and score calculation, we only need the second column as our predicted scores
y_train_prob[:3]

# %%
from sklearn.metrics import roc_curve

# your code below for roc_curve (i.e., display roc plot)

# start of your code (don't modify code outside your code block)





# end of your code (don't modify code outside your code block)

# %%
from sklearn.metrics import roc_auc_score

# your code below for auc score
# start of your code (don't modify code outside your code block)





# end of your code (don't modify code outside your code block)

# %% [markdown]
# ## 4 Grid Search

# %%
from sklearn.model_selection import GridSearchCV


# %%
# your code below for setting up grid search score



# %%
# your code below for print out the best parameters and best score




