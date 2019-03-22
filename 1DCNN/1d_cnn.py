from keras.models import Sequential, load_model
from keras.layers import Dense, Activation,Conv1D,MaxPooling1D,Flatten,Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.initializers import random_uniform
from keras.callbacks import ModelCheckpoint
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wandb
import time

wandb_use = True
start_time = time.time()
if wandb_use == True:
    wandb.init(project="1d_cnn", tensorboard=False)
    
num_input = 42
num_output =2
SEED = 42
total_epochs = 200
batch_size = 128


#hyperparameters
input_dimension = 226
learning_rate = 0.0001
momentum = 0.85
hidden_initializer = random_uniform(seed=SEED)
dropout_rate = 0.25

#load data
f1 = open('../data/FC/training_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f1)
X_train = []
y_train = []
for line in rdr:
    line = [float(i) for i in line]
    X_train.append(line[0:num_input])
    #x_data_val.append(line[29:43])
    y_train.append(line[-num_output:])
X_train = np.reshape(X_train, (-1, num_input))
y_train = np.reshape(y_train, (-1, num_output))
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

f2 = open('../data/FC/testing_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f2)
X_test = []
y_test = []
for line in rdr:
    line = [float(i) for i in line]
    X_test.append(line[1:num_input+1])
    #x_data_val.append(line[29:43])
    y_test.append(line[-num_output:])
X_test = np.reshape(X_test, (-1, num_input))
y_test = np.reshape(y_test, (-1, num_output))

f3 = open('../data/FC/validation_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f3)
X_val = []
y_val = []
for line in rdr:
    line = [float(i) for i in line]
    X_val.append(line[1:num_input+1])
    #x_data_val.append(line[29:43])
    y_val.append(line[-num_output:])
X_val = np.reshape(X_val, (-1, num_input))
y_val = np.reshape(y_val, (-1, num_output))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

print(X_train.shape[1:3])

# create model
model = Sequential()
#convolution 1st layer
model.add(Conv1D(32, kernel_size=6, input_shape=X_train.shape[1:3], padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D())
model.add(Dropout(dropout_rate))

#convolution 2nd layer
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D())
model.add(Dropout(dropout_rate))

#convolution 3rd layer
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D())
model.add(Dropout(dropout_rate))

#Fully connected 1st layer
model.add(Flatten()) #flatten serves as a connection between the convolution and dense layers
#model.add(Dense(256, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dense(256, kernel_initializer=hidden_initializer, activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

#fully connected final layer
#Dense(첫번째 인자 : 출력 뉴런의 수, input_dim : 입력 뉴런의 수, init :'uniform’ or ‘normal’)
model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dense(2, activation='softmax'))
#softmax makes the output sum up to 1 so interpreted as probabilities

sgd = SGD(lr=learning_rate, momentum=momentum)
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
CNN = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=total_epochs, batch_size=batch_size)
#predictions = model.predict_proba(X_test)

score = model.evaluate(X_test, y_test, batch_size=128)
print("loss & accuracy :" ,score)
elapsed_time = time.time() - start_time
print("elapsed time : ", elapsed_time)

# Plot training & validation accuracy values
plt.plot(CNN.history['acc'])
plt.plot(CNN.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(CNN.history['loss'])
plt.plot(CNN.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#save model
model.save('model/ttest.h5')
print("Saved model to disk")
  
