Description:

This script analyzes the "Human Activity Recognition Using Smartphones Data Set"
data set (see below for link to the data set).  Data size: 10299 samples.

One analysis is to predict the state of the person carrying the smartphone,
based on the sensor data.  This script, however, classifies based on user id
-- it learns to predict which user is carrying the phone -- using a
feature vector with 561 features, derived from smartphone measurements.
A physical signature, if you will.  The prediction accuracy is, on average,
92%.

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
