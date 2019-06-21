#####################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code compares the Kernel SVM and Random Forest Classification
#               model using the MNIST dataset.
#####################################################################################

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.model_selection import cross_val_score

avgs = []
stds = []

####################################################################################################
# SVM Classifier

# Fitting SVM Classifier to Training Set
from sklearn.svm import SVC
classifier1 = SVC(kernel='rbf')
classifier1.fit(X_train, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies1 = cross_val_score(estimator=classifier1, X=X_train, y=Y_train, cv=5)
avgs.append(accuracies1.mean()*100)
stds.append(accuracies1.std()*100)

####################################################################################################
# Random Forest Classifier

# Fitting Random Forest Classifier to Training Set
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=16, min_samples_split=4)
classifier2.fit(X_train, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies2 = cross_val_score(estimator=classifier2, X=X_train, y=Y_train, cv=10)
avgs.append(accuracies2.mean()*100)
stds.append(accuracies2.std()*100)

####################################################################################################

plt.figure(figsize=(5,6))
plt.bar(range(len(avgs)), avgs, color=(118/255,127/255,255/255), yerr=stds, capsize=10)
plt.ylim([92, 97])
plt.xticks(np.arange(2),['Kernel SVM','Random Forest'])
plt.grid(True)
plt.title('Comparison of Classification models')
plt.xlabel('Models')
plt.ylabel('Accuracy Percentage')
plt.show()

print('\a')