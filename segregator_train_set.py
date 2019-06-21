#####################################################################################
# Creator     : Gaurav Roy
# Date        : 12 June 2019
# Description : The code creates .png files from the pixel matrix of dataset 
#               and splits them into the corresponding folder. 
#####################################################################################

# Each image is 28x28 pixels
px_len = 28

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Importing dataset
train = pd.read_csv('train.csv')
X = train.iloc[:,1:].values
Y = train.iloc[:,0].values

plt.imshow(dataset.iloc[1,1:].values.reshape((28,28)), cmap='gray')

temp = X[0].reshape((px_len, px_len))
filname = 'temp.jpg'
plt.imsave(filname, arr=temp)
temp2 = plt.imread(filname)
temp2.shape


# Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)


# Creating a count matrix to help with naming
num_count = np.zeros(shape=(10,1))

###################################################################################################
# Training Set

# Iteration that puts the image together and stores in the correct folder
for i in range(0,len(X_train)):    
    img = X_train[i].reshape((px_len, px_len))
            
    filname = 'dataset/training_set/'+str(Y_train[i])+'/'+str(int(num_count[Y_train[i], 0]+1))+'.png'    
    plt.imsave(filname, arr=img)
    num_count[Y_train[i], 0] = num_count[Y_train[i], 0] +1;

###################################################################################################
# Test Set

# Iteration that puts the image together and stores in the correct folder
for i in range(0,len(X_test)):
    img = X_test[i].reshape((px_len, px_len))
            
    filname = 'dataset/test_set/'+str(Y_test[i])+'/'+str(int(num_count[Y_test[i], 0]+1))+'.png'    
    plt.imsave(filname, arr=img)
    num_count[Y_test[i], 0] = num_count[Y_test[i], 0] +1;

###################################################################################################
print('\a') # Alert to notify that job is finished


#test = plt.imread('dataset/training_set/0/1.png')