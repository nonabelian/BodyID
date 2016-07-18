# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 14:48:24 2016

@author: dylan

LICENSE:

    This is the HAR_analysis.py file from my data analysis project
    'di_har_proj'.

    Copyright (C) 2016  Dylan Albrecht
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description:

This script analyzes the "Human Activity Recognition Using Smartphones Data Set"
data set (see below for link to the data set).

One analysis is to predict the state of the person carrying the smartphone,
based on the sensor data.  This script, however, classifies based on user id
-- it learns to predict which user is carrying the phone -- using a
feature vector with 561 features, derived from smartphone measurements.

Note: You must copy the relevant data folders to the src/ dir.  Or you can
      change the script to load the appropriate data files.

Data:
        train/X_train.txt         -- 70% of data, 561 real numbered columns
        train/y_train.txt         -- 70% of data, one integer column, activity
        train/subject_train.txt   -- 70% of data, one integer column, user id
        test/X_test.txt           -- 30% of data, 561 real numbered columns
        test/y_test.txt           -- 30% of data, one integer column, activity
        tset/subject_test.txt     -- 30% of data, one integer column, user id

Run:
        python HAR_analysis.py

Output: 
        cfm.jpg         -- Normalized confusion matrix for the classification
        db.jpg          -- Decision boundary visualization from the best two
                           features.

data source: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics

def plot_cfm(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    label_list = list(range(31))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list)
    plt.yticks(tick_marks, label_list)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

columns = list(map(str, list(range(561))))
target_name = ['target class']
subject_name = ['subject name']

# Load data and select the columns we're interested in.
har_train_data = pd.read_fwf('train/X_train.txt', sep=' ', names=columns)
har_train_activity = pd.read_fwf('train/y_train.txt', sep=' ', names=target_name)
har_train_subject = pd.read_csv('train/subject_train.txt', names=subject_name)

har_test_data = pd.read_fwf('test/X_test.txt', sep=' ', names=columns)
har_test_activity = pd.read_fwf('test/y_test.txt', sep=' ', names=target_name)
har_test_subject = pd.read_csv('test/subject_test.txt', names=subject_name)

print('har_train_data.shape: ', har_train_data.shape)
print('har_train_activity.shape: ', har_train_activity.shape)
print('har_train_subject.shape: ', har_train_subject.shape)

# Combine data back together
har_data = har_train_data.append(har_test_data)
har_activity = har_train_activity.append(har_test_activity)
har_subject = har_train_subject.append(har_test_subject)

X = har_data.as_matrix()
Y = har_activity.as_matrix().ravel()
S = har_subject.as_matrix().ravel()

################################
# Training and model selection
################################

# Perform our own train/test split, at test size 20%
X_train, X_test, S_train, S_test = cross_validation.train_test_split(X, S, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=1, random_state=0)

clf = clf.fit(X_train, S_train)

# Reduce the features
model = SelectFromModel(clf, prefit=True)
X_reduced_train = model.transform(X_train)
X_reduced_test = model.transform(X_test)

print('X_train.shape: ', X_train.shape)
print('X.shape: ', X_reduced_train.shape)

# Train again based on the restricted set of features
clf = clf.fit(X_reduced_train, S_train)

# Let's see how well the new classifier does (should be avg: 92%)
print('score', clf.score(X_reduced_test, S_test))
predicted = clf.predict(X_reduced_test)
print(metrics.classification_report(S_test, predicted))

#############
# Plotting
#############

# Plot the confusion matrix
cfm = metrics.confusion_matrix(S_test, predicted)
cfm_normalized = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
cfm_fig = plt.figure(figsize=(10,10))
plot_cfm(cfm_normalized, title='Normalized Confusion Matrix')
plt.savefig('cfm.jpg')

# Now for the decision boundary visualization

# Grab the two best features for plotting purposes
ifeats = list(clf.feature_importances_)
i1f = ifeats.index(np.max(ifeats))
ifeats.remove(np.max(ifeats))
i2f = ifeats.index(np.max(ifeats))

print('feature_importances[%d,%d]: %s' % (i1f, i2f, clf.feature_importances_[[i1f, i2f]]))

X_2feat_train = X_reduced_train[:,[i1f, i2f]]
X_2feat_test = X_reduced_test[:, [i1f, i2f]]

# Refit, so we can predict!
clf.fit(X_2feat_train, S_train)

# Let's see how well our (too) much reduced set does (not very good, but
# reasonable for visualization purposes)
predicted = clf.predict(X_2feat_test)
print(metrics.classification_report(S_test, predicted))

# Color map for the class labels
cs = cm.Spectral(np.linspace(0, 1, len(np.unique(S_test))))

# Plot the decision boundaries based on the two feature TRAINING SET
x_min, x_max = X_2feat_train[:, 0].min() - 1, X_2feat_train[:, 0].max() + 1
y_min, y_max = X_2feat_train[:, 1].min() - 1, X_2feat_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min)/ 100), np.arange(y_min, y_max, (x_max - x_min)/ 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

decision_bdy_fig = plt.figure(figsize=(10,10))
plt.contourf(xx, yy, Z, cmap=cm.Spectral)

# plot the TEST SET data points
color_classes = []
for i, c in zip(range(0,len(np.unique(S_test))), list(map(colors.rgb2hex, cs))):
    idx = np.where(S_test == i)
    color_class = plt.scatter(X_2feat_test[idx, 0], X_2feat_test[idx, 1], c=c, label=i, cmap=cm.Spectral)
    color_classes.append(color_class)

plt.title('Decision Boundary')
plt.xlabel('Feature #1')
plt.ylabel('Feature #2')
plt.legend(handles=color_classes)
plt.savefig('db.jpg')

plt.show()

###############
# End of File
###############
