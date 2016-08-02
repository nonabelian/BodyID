**Description**:

This script analyzes the [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).
It uses Random Forest classifiers and t-SNE projections to cluster and
predict smartphone users, based on motion sensor data.

Data size: 10299 samples.

One analysis is to predict the state of the person carrying the smartphone,
based on the sensor data.  This script, however, classifies based on user id
-- it learns to predict which user is carrying the phone -- using a
feature vector with 561 features, derived from smartphone measurements.
This is regardless of the activity the individual is performing -- a physical
signature, if you will.  The prediction accuracy is on average 92%.

In addition, the script then computes a classification per activity.
First each activity's data undergoes t-SNE reduction to two dimensions,
to demonstrate clustering relations/closeness.  Then classification is
performed for each activity, with accuracy upwards of 95% for some activities.

**Note**:

		You must copy the relevant data folders to the src/ dir.  Or you can
    	change the script to load the appropriate data files.

**Data**:

		* train/X_train.txt         -- 70% of data, 561 real numbered columns
		* train/y_train.txt         -- 70% of data, one integer column, activity
		* train/subject_train.txt   -- 70% of data, one integer column, user id
		* test/X_test.txt           -- 30% of data, 561 real numbered columns
		* test/y_test.txt           -- 30% of data, one integer column, activity
		* tset/subject_test.txt     -- 30% of data, one integer column, user id

**Run**: `python HAR_analysis.py`

**Output**: 

* 'cfm.jpg':

   Normalized confusion matrix for the classification
* 'db.jpg':

   Decision boundary visualization from the best two features.
* 'Laying_db.jpg':

   Decision boundary of classification of individuals 'laying', based on t-SNE projection.
* 'Sitting_db.jpg':

   Decision boundary of classification of individuals 'sitting', based on t-SNE projection.
* 'Standing_db.jpg':

   Decision boundary of classification of individuals 'standing', based on t-SNE projection.
* 'Walking_db.jpg':

   Decision boundary of classification of individuals 'walking', based on t-SNE projection.
* 'Walking Down Stairs_db.jpg':

   Decision boundary of classification of individuals 'walking down stairs', based on t-SNE projection.
* 'Walking Up Stairs_db.jpg':

   Decision boundary of classification of individuals 'walking down stairs', based on t-SNE projection.

**Data source**: [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

**Citation**:

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge
L.Reyes-Ortiz. "*A Public Domain Dataset for Human Activity Recognition Using
Smartphones*." 21th European Symposium on Artificial Neural Networks,
Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium
24-26 April 2013.
