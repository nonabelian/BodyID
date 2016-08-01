# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 14:48:24 2016

@author: dylan

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
        Laying_db.jpg   -- Decision boundary of classification of individuals
                           'laying', based on t-SNE projection.
        Sitting_db.jpg  -- Decision boundary of classification of individuals
                           'sitting', based on t-SNE projection.
        Standing_db.jpg -- Decision boundary of classification of individuals
                           'standing', based on t-SNE projection.
        Walking_db.jpg  -- Decision boundary of classification of individuals
                           'walking', based on t-SNE projection.
        Walking Down Stairs_db.jpg  -- Decision boundary of classification of
                                       individuals 'walking down stairs', based
                                       on t-SNE projection.
        Walking Up Stairs_db.jpg    -- Decision boundary of classification of
                                       individuals 'walking down stairs', based
                                       on t-SNE projection.

data source: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Citation:

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge
L.Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using
Smartphones. 21th European Symposium on Artificial Neural Networks,
Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium
24-26 April 2013.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.markers as markers

from sklearn import cross_validation
from sklearn import manifold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics


#########################
# Some useful functions
#########################

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
    

def plot_decision_bdy(X_train_plot, X_test_plot, S_train, S_test, classifier, title='Decision Boundary',
                      xlabel='Feature 1', ylabel='Feature 2'):
    # Set up markers for plotting all 30 subjects -- removing 'nothing' markers
    #ms = list(markers.MarkerStyle.markers)
    #ms.remove('')
    #ms.remove(' ')
    #ms.remove('None')
    #ms.remove(None)

    plt.figure(figsize=(10,10))
    subject_cs = cm.Spectral(np.linspace(0, 1, len(np.unique(S_train))))

    # Plot the decision boundaries based on the two feature TRAINING SET
    x_min, x_max = X_train_plot[:, 0].min() - 1, X_train_plot[:, 0].max() + 1
    y_min, y_max = X_train_plot[:, 1].min() - 1, X_train_plot[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min)/ 100), np.arange(y_min, y_max, (x_max - x_min)/ 100))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=cm.Spectral)

    color_subjects = []
    for i, c in zip(range(0,len(np.unique(S_train))), list(map(colors.rgb2hex, subject_cs))):
        idx = np.where(S_test == i)
        # Scatter plot the TEST DATA
        #color_subject = plt.scatter(X_test_plot[idx, 0], X_test_plot[idx, 1], c=c, marker=ms[i], s=100, label=i, cmap=cm.Spectral)
        color_subject = plt.scatter(X_test_plot[idx, 0], X_test_plot[idx, 1], c=c, s=100, label=i)
        color_subjects.append(color_subject)

    plt.legend(handles=color_subjects)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def get_activity_dataset(data, d_columns, labels, l_columns, activity):
    X_activity = data[data.activity == activity][d_columns].as_matrix()
    S_activity = labels[labels.activity == activity][l_columns].as_matrix()
    
    return X_activity, S_activity

#######################
# Script -- Load Data
#######################

columns = list(map(str, list(range(561))))
target_name = ['target class']
subject_name = ['subject name']

activities = {1: 'Walking', 2: 'Walking Up Stairs', 3: 'Walking Down Stairs',
              4: 'Sitting', 5: 'Standing', 6: 'Laying'}

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

# Full dataset in matrix form
X = har_data.as_matrix()
# Full labels (activity label)
Y = har_activity.as_matrix().ravel()
# Full labels (user ID)
S = har_subject.as_matrix().ravel()

#############################
# For splitting by activity

# Add columns for the labels
har_data['activity'] = har_activity
har_data['subject']= har_subject['subject name']

# Add column for activity label
har_subject['activity'] = har_activity

################################
# Training and model selection
################################

# Perform our own train/test split, at test size 20%
X_train, X_test, S_train, S_test = cross_validation.train_test_split(X, S, test_size=0.2, random_state=42)
clf_feature_select = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=1, random_state=0)
clf_classify = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=1, random_state=0)

# Use a first pass classifier to pick features:
clf_feature_select = clf_feature_select.fit(X_train, S_train)

# Reduce the features
model = SelectFromModel(clf_feature_select, prefit=True)

###################################
# Classify regardless of activity

X_reduced_train = model.transform(X_train)
X_reduced_test = model.transform(X_test)

print('X_train.shape: ', X_train.shape)
print('X.shape: ', X_reduced_train.shape)

# Train again based on the restricted set of features
clf_classify = clf_classify.fit(X_reduced_train, S_train)
    
# Let's see how well the new classifier does (should be avg: 92%)
print('score', clf_classify.score(X_reduced_test, S_test))
predicted = clf_classify.predict(X_reduced_test)
print(metrics.classification_report(S_test, predicted))

##############################
# Classify for each activity

# T-SNE -- 2d reduction and clustering.
tsne = manifold.TSNE(n_components=2, random_state=0)
reduced_clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)

for a in list(activities):
    Xa, Sa = get_activity_dataset(har_data, columns, har_subject, 'subject name', a)
    
    # T-SNE and classification of clusters
    Xa_plot = tsne.fit_transform(Xa)
    Xa_train_plot, Xa_test_plot, Sa_train, Sa_test = cross_validation.train_test_split(Xa_plot, Sa, test_size=0.2, random_state=42)

    # Train a classifier on the projected data:
    reduced_clf.fit(Xa_train_plot, Sa_train)

    a_score = reduced_clf.score(Xa_test_plot, Sa_test)
    #a_predicted = clf_classify.predict(Xa_test_plot)
    #print(metrics.classification_report(S_test, predicted))
    print('len(Xa_train_plot): %d; len(Xa_test_plot): %d' % (len(Xa_train_plot), len(Xa_test_plot)))

    # Plot the test data in the training data contours
    a_title = 'Subject Decision Boundary; ' + activities[a] + '; Score: ' + '%.1f%%' % (a_score * 100)
    a_xlabel = 'Embedded dimension 1'
    a_ylabel = 'Embedded dimension 2'
    plot_decision_bdy(Xa_train_plot, Xa_test_plot, Sa_train, Sa_test,
                      reduced_clf, a_title, a_xlabel, a_ylabel)
                      
    a_fname = activities[a] + '_db.jpg'
    plt.savefig(a_fname)

    plt.show()

#############
# Plotting
#############

# Plot the confusion matrix
cfm = metrics.confusion_matrix(S_test, predicted)
cfm_normalized = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
cfm_fig = plt.figure(figsize=(10,10))
plot_cfm(cfm_normalized, title='Normalized Confusion Matrix')
plt.savefig('cfm.jpg')

# Grab the two best features for plotting purposes
ifeats = list(clf_classify.feature_importances_)
i1f = ifeats.index(np.max(ifeats))
ifeats.remove(np.max(ifeats))
i2f = ifeats.index(np.max(ifeats))

print('feature_importances[%d,%d]: %s' % (i1f, i2f, clf_classify.feature_importances_[[i1f, i2f]]))

X_2feat_train = X_reduced_train[:,[i1f, i2f]]
X_2feat_test = X_reduced_test[:, [i1f, i2f]]

# Refit, so we can predict!
clf_classify.fit(X_2feat_train, S_train)

# Let's see how well our (too) much reduced set does (not very good, but
# reasonable for visualization purposes)
predicted = clf_classify.predict(X_2feat_test)
print(metrics.classification_report(S_test, predicted))

plot_decision_bdy(X_2feat_train, X_2feat_test, S_train, S_test, clf_classify)
plt.savefig('db.jpg')

plt.show()

###############
# End of File
###############
