import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit

#Data load

path_df = "/Users/macbookpro/Desktop/INNObyte/modules/Pickles/df.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# features_train
path_features_train = "/Users/macbookpro/Desktop/INNObyte/modules/Pickles/features_train.pickle"
with open(path_features_train, 'rb') as data:
    features_train = pickle.load(data)

# labels_train
path_labels_train = "/Users/macbookpro/Desktop/INNObyte/modules/Pickles/labels_train.pickle"
with open(path_labels_train, 'rb') as data:
    labels_train = pickle.load(data)

# features_test
path_features_test = "/Users/macbookpro/Desktop/INNObyte/modules/Pickles/features_test.pickle"
with open(path_features_test, 'rb') as data:
    features_test = pickle.load(data)

# labels_test
path_labels_test = "/Users/macbookpro/Desktop/INNObyte/modules/Pickles/labels_test.pickle"
with open(path_labels_test, 'rb') as data:
    labels_test = pickle.load(data)

#Tuning the hyperparameter phase
rf1 = RandomForestClassifier(random_state = 8)
# view the current parameters
# pprint(rf1.get_params())

# some changes on the rf parameters
# n_estimators
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# grid creation
rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# pprint(rf_grid)

# model to tune
rf_c = RandomForestClassifier(random_state=8)

# random search define
random_search = RandomizedSearchCV(estimator=rf_c,
                                   param_distributions=rf_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3,
                                   verbose=1,
                                   random_state=8)
"""
# fit the model with training features and labels (this may take a while)
random_search.fit(features_train, labels_train)
# get hyperparameters
print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)
# accuracy test result was 0.4435628206657916 in this case which is low
# params : {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 4,
 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': False}
"""

# try Grid search cross validation

# Create the parameter grid based on the results of random search
bootstrap = [False]
max_depth = [10, 20, 30]
max_features = ['sqrt']
min_samples_leaf = [4, 6, 8]
min_samples_split = [10, 15, 20]
n_estimators = [600]

param_grid = {
    'bootstrap': bootstrap,
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators
}

# Create a base model
rfc = RandomForestClassifier(random_state=8)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# grid search model
grid_search = GridSearchCV(estimator=rfc,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)

print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)

"""
improved results of accuracy : 0.4494773519163762
params : {'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt',
    'min_samples_leaf': 4, 'min_samples_split': 20, 'n_estimators': 600}
"""


rfc_chosen = grid_search.best_estimator_
# fitting model for training
rfc_chosen.fit(features_train, labels_train)
rfc_pred = rfc_chosen.predict(features_test)

# comparing the training and test accuracies
print("The training accuracy is: ")
print(accuracy_score(labels_train, rfc_chosen.predict(features_train)))

print("The test accuracy is: ")
print(accuracy_score(labels_test, rfc_pred))

base_model = RandomForestClassifier(random_state = 8)
base_model.fit(features_train, labels_train)
print(accuracy_score(labels_test, base_model.predict(features_test)))

rfc_chosen.fit(features_train, labels_train)
print(accuracy_score(labels_test, rfc_chosen.predict(features_test)))

"""
Results for this prediction model: 
Training accuracy: 
0.673963133640553
Test accuracy: 
0.44155844155844154
"""