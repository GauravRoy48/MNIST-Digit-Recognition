##################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code contains the approach for Random Forest classification
#               technique on the MNIST data
##################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values

# Normalizing the Pixels
X = X/255

# Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

# Fitting Random Forest Classifier to Training Set
# Create Classifier Here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=16, min_samples_split=4)
classifier.fit(X_train, Y_train)

# Predicting the Test Set Results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying k-Fold Cross Validation where k=20
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=20)
avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()
print('\a')

# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[1000,1200], 'criterion':['entropy'],
               'max_depth':[14,15,16], 'min_samples_split':[4,5,6]}]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, Y_train)
# Gives the accuracy with the best parameters
best_accuracy = grid_search.best_score_

###########################################################
# Gives the values of the best hyperparameters
best_parameters = grid_search.best_params_
print('\a')
###########################################################

test = pd.read_csv('test.csv')
X_check = test.iloc[:,:].values
X_check = sc_X.transform(X_check)

Y_check = classifier.predict(X_check)

output=pd.DataFrame({'ImageId':np.arange(1,len(Y_check)+1),'Label':Y_check})
output.to_csv('output2.csv', index=False)