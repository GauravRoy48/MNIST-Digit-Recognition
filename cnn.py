#####################################################################################
# Creator     : Gaurav Roy
# Date        : 21 June 2019
# Description : The code contains the CNN model used for the creating the saved
#               model that is used in the website. 
#####################################################################################

# Each image is 28x28 pixels
px_len = 28

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

# Importing dataset
train = pd.read_csv('train.csv')
X = train.iloc[:,1:].values
Y = train.iloc[:,0].values

# Normalizing the Pixels
X = X/255

# Encoding for the Label Column 
Y = pd.Categorical(Y)
Y = pd.get_dummies(Y)

# Reshaping X to a 2D matrix (28x28pixel) with 1channel
X = X.reshape((len(X),px_len, px_len, 1))

# Building CNN

# Importing Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

# Initializing the CNN
classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate=0.15))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate=0.15))
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connections
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 
mem = classifier.fit(x=X,
               y=Y,
               batch_size=50,
               epochs=20,
               validation_split=0.2)


print("CNN: Max Train Acc: {0:.3f}%, Mean Train Acc: {1:.3f}%, Max Test Acc: {2:.3f}%, Mean Test Acc: {3:.3f}%".format(
      max(mem.history['acc'])*100,
      mean(mem.history['acc'])*100,
      max(mem.history['val_acc'])*100,
      mean(mem.history['val_acc'])*100))

# Save Model
classifier.save('model/cnn_model.h5')
## serialize model to JSON
#model_json = classifier.to_json()
#with open("cnn_model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize model to YAML
#model_yaml = classifier.to_yaml()
#with open("cnn_model.yaml", "w") as yaml_file:
#    yaml_file.write(model_yaml)
## serialize weights to HDF5
#classifier.save_weights("cnn_model_weights.h5")
#print("Saved model to disk")



print('\a') # Notification Alert




## Test with Kaggle test data and extract to csv
#test = pd.read_csv('test.csv')
#X_check = test.iloc[:,:].values
#X_check = X_check/255
#X_check = X_check.reshape((len(X_check), px_len, px_len, 1))
#
#Y_check = classifier.predict(X_check)
#Y_check_1 = [np.argmax(y) for y in Y_check]
#
#output=pd.DataFrame({'ImageId':np.arange(1,len(Y_check_1)+1),'Label':Y_check_1})
#output.to_csv('output5.csv', index=False)