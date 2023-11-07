    # %% [markdown]
    # # Lab 5: Ordinal and OneHot Encoding
    # 
    # The goal of this lab is to evaluate the impact of using ordinal encoding and onehot encoding for categorical variables along with a Logistic Regression and Decision Tree.

    # %% [markdown]
    # ## 0. Data Loading

    # %%
    import pandas as pd

    adult_census = pd.read_csv("adult-census.csv")

    # %%
    target_name = "class"
    target = adult_census[target_name]
    data = adult_census.drop(columns=[target_name, "education-num"])

    # %% [markdown]
    # **Note**: we could use `sklearn.compose.make_column_selector` to automatically select columns with `object` dtype that correspond to categorical features in our dataset.

    # %%
    from sklearn.compose import make_column_selector as selector

    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(data)
    print(categorical_columns)
    data_categorical = data[categorical_columns]

    # %%
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data_categorical, target, test_size=0.3, random_state=42)

    # %%


    # %% [markdown]
    # ## 1. DummyClassifier
    # 
    # **DummyClassifier** makes predictions that ignore the input features. This classifier serves as a *simple baseline to compare against other more complex classifiers*. The specific behavior of the baseline is selected with the strategy parameter.

    # %%
    from sklearn.pipeline import make_pipeline
    from sklearn.dummy import DummyClassifier

    dummy_model = make_pipeline(DummyClassifier())

    # %%
    from sklearn.metrics import accuracy_score

    dummy_model.fit(X_train, y_train)
    accuracy_score(y_test, dummy_model.predict(X_test))

    # %%


    # %% [markdown]
    # ## 2. OrdinalEncoder and LogisticRegression
    # 
    # **Task#1**: Define a scikit-learn pipeline composed of an `OrdinalEncoder` and a
    # `LogisticRegression` classifier. Fit your pipeline on training dataset and evaluate your prediction accuracy on your test dataset.
    # - `OrdinalEncoder` can raise errors if it sees an unknown category at
    # prediction time, you can set the `handle_unknown="use_encoded_value"` and
    # `unknown_value=-1` parameters. You can refer to the
    # [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
    # for more details regarding these parameters.
    # - Use hyperparameter of `max_iter=500` in your LogisticRegression

    # %%
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.linear_model import LogisticRegression

    # Write your code here (model)


    # %%
    # Write your code here (accuracy)


    # %% [markdown]
    # ## 3. OneHotEncoder and LogisticRegression
    # 
    # **Task#2**: Define a scikit-learn pipeline composed of an `OneHotEncoder` and a
    # `LogisticRegression` classifier. Fit your pipeline on training dataset and evaluate your prediction accuracy on your test dataset.
    # 
    # - `OneHotEncoder` can raise errors if it sees an unknown category at
    # prediction time, you can set the `handle_unknown="ignore"` parameter. 
    # - Use hyperparameter of `max_iter=500` in your LogisticRegression

    # %%
    from sklearn.preprocessing import OneHotEncoder

    # Write your code here (model).


    # %%
    # Write your code here (accuracy)



    # %%


    # %% [markdown]
    # ## 4. DecisionTree on Categorical Data

    # %% [markdown]
    # **Important**: tree in sklkearn only accecpt numerical values, categorical values not accepted.

    # %%
    from sklearn.tree import DecisionTreeClassifier
    tree_model = make_pipeline(DecisionTreeClassifier())

    #tree_model.fit(X_train, y_train)

    # %%


    # %% [markdown]
    # ## 5. OrdinalEncoder and DecisionTree
    # 
    # **Task#3**: Define a scikit-learn pipeline composed of an `OrdinalEncoder` and a
    # `DecisionTree` classifier. Fit your pipeline on training dataset and evaluate your prediction accuracy on your test dataset.
    # 
    # - `OrdinalEncoder` can raise errors if it sees an unknown category at
    # prediction time, you can set the `handle_unknown="use_encoded_value"` and
    # `unknown_value=-1` parameters.

    # %%
    # Write your code here (model)



    # %%
    # Write your code here (accuracy)



    # %% [markdown]
    # ## 6. OneHotEncoder and DecisionTree
    # 
    # **Task#4**: Define a scikit-learn pipeline composed of an `OneHotEncoder` and a
    # `DecisionTreeClassifier'. Fit your pipeline on training dataset and evaluate your prediction accuracy on your test dataset.
    # 
    # - `OneHotEncoder` can raise errors if it sees an unknown category at
    # prediction time, you can set the `handle_unknown="ignore"` parameter. 

    # %%
    # Write your code here (model)



    # %%
    # Write your code here (accuracy)



    # %%



