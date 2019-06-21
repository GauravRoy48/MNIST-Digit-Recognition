#####################################################################################
# Creator     : Gaurav Roy
# Date        : 13 June 2019
# Description : The code compares the different CNN models which were used to test 
#               and improve the score of the CNN. 
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
from keras.layers import Flatten, Dropout, BatchNormalization

nets = 2
classifier  = [0]*nets
mem = [0]*nets
for i in range(nets):
    # Initializing the CNN
    classifier[i] = Sequential()
    
    if i>0:
        classifier[i].add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = 'relu'))
        classifier[i].add(BatchNormalization())
        classifier[i].add(Conv2D(32, (3,3), activation = 'relu'))
        classifier[i].add(BatchNormalization())
        classifier[i].add(MaxPooling2D(pool_size = (2, 2)))
        classifier[i].add(BatchNormalization())
        classifier[i].add(Dropout(rate=0.15))
        
        # Adding a second convolutional layer
        classifier[i].add(Conv2D(64, (3,3), activation = 'relu'))
        classifier[i].add(BatchNormalization())
        classifier[i].add(Conv2D(64, (3,3), activation = 'relu'))
        classifier[i].add(BatchNormalization())
        classifier[i].add(MaxPooling2D(pool_size = (2, 2)))
        classifier[i].add(BatchNormalization())
        classifier[i].add(Dropout(rate=0.15))
    
        # Step 3 - Flattening
        classifier[i].add(Flatten())
         
        # Step 4 - Full connections
        classifier[i].add(Dense(units = 128, activation = 'relu'))
        classifier[i].add(BatchNormalization())
    
        classifier[i].add(Dropout(rate=0.15))
        
    else:
        classifier[i].add(Conv2D(32, (5,5), input_shape = (28, 28, 1), activation = 'relu'))
        classifier[i].add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a second convolutional layer
        classifier[i].add(Conv2D(64, (5,5), activation = 'relu'))
        classifier[i].add(MaxPooling2D(pool_size = (2, 2)))
    
        # Step 3 - Flattening
        classifier[i].add(Flatten())
         
        # Step 4 - Full connections
        classifier[i].add(Dense(units = 256, activation = 'relu'))
        
    classifier[i].add(Dense(units = 10, activation = 'sigmoid'))

    # Compiling the CNN
    classifier[i].compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
   
    mem[i] = classifier[i].fit(x=X,
               y=Y,
               batch_size=50,
               epochs=30,
               validation_split=0.2)

names = ["Initial model","Final model"]

plt.figure(figsize=(17,5))
for j in range(nets):
    print("CNN: {0}, Max Train Acc: {1:.3f}%, Mean Train Acc: {2:.3f}%, Max Test Acc: {3:.3f}%, Mean Test Acc: {4:.3f}%".format(
          names[j],
          max(mem[j].history['acc'])*100,
          mean(mem[j].history['acc'])*100,
          max(mem[j].history['val_acc'])*100,
          mean(mem[j].history['val_acc'])*100))
    
    plt.plot(mem[j].history['val_acc'])
plt.title('Model Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(names)
plt.grid(True)
plt.show()
        
print('\a') # Notification Alert