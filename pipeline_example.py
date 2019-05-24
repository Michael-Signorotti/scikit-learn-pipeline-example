"""Create scikit-learn pipeline with grid search example."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

'''
Load the standard iris multi-class classification dataset
'''
iris_data = load_iris()

'''
Transform the scikit-learn dataset into a pandas dataframe
'''
col_names = iris_data['feature_names'] + ['target']
df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                  columns=col_names)

'''
Utilize an 80/20 train/test split
'''
X = df[[col for col in df.columns if col != 'target']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Build the machine learning pipeline, define the grid of
hyperparameters to search, and create a scorer for
measuring model performance during the cross
validation process.
'''
pipeline = Pipeline(steps=[('standardize', StandardScaler()),
                           ('decision_tree', DecisionTreeClassifier(criterion='entropy'))])

params = [{'decision_tree__max_depth': range(2, 8)}]
scorer = make_scorer(f1_score, average='micro')
clf = GridSearchCV(estimator=pipeline, scoring=scorer,
                   param_grid=params, n_jobs=-1, cv=10, verbose=3,
                   return_train_score=True)

clf.fit(X_train, y_train)

print('Best CV F1 Score:  %s' % str(clf.best_score_))
print('Best Model Params:  %s' % str(clf.best_params_))
print('Test set F1 Score:  %s' % str(scorer(clf, X_test, y_test)))
