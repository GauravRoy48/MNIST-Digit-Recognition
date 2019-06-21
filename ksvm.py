#####################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code contains the Kernel Support Vector Machine model for the 
#               MNIST dataset.
#####################################################################################

# Importing Libraries
import numpy as np
import pandas as pd

# Import Dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values

# Normalizing the Pixels
X = X/255

## Encoding for the Label Column 
#Y = pd.Categorical(Y)
#Y = pd.get_dummies(Y)

# Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

####################################################################################################

# Fitting SVM Classifier to Training Set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0, gamma=0.1)
classifier.fit(X_train, Y_train)

#################################################################################################

# Applying k-Fold Cross Validation where k=10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()
print('\a')

# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[0.01,0.1,1,10], 'kernel':['linear']},
               {'C':[0.01,0.1,1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.01, 0.05, 0.1, 0.2, 0.3]}]

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
###########################################################