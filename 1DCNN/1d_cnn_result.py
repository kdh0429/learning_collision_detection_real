from keras.models import Sequential
from keras.layers import Dense, Activation,Conv1D,MaxPooling1D,Flatten,Dropout
from keras.optimizers import SGD, Adam
from keras.initializers import random_uniform
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import h5py
from keras.utils import plot_model

num_input = 42
num_output =2
#hyperparameters

loaded_model = load_model('model/test.h5')
print("Loaded model from disk")
print(loaded_model.summary())
plot_model(loaded_model, to_file='result/model.png', show_shapes=True)

for i in range(192):
    path = '../data/FC/TestingDivide/Testing_raw_data_' + str(i+1) + '.csv'
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_data_raw = []
    y_data_raw = []
    num_data = 0
    for line in rdr:
        line = [float(i) for i in line]
        x_data_raw.append(line[0:num_input])
        y_data_raw.append(line[-num_output:])
        num_data = num_data+1
        
    t = range(len(x_data_raw))
    t = np.reshape(t,(-1,1))
    x_data_raw = np.reshape(x_data_raw, (-1, num_input))
    y_data_raw = np.reshape(y_data_raw, (-1, num_output))

    x_data_raw = x_data_raw.reshape(x_data_raw.shape[0], x_data_raw.shape[1], 1)

    #predict_proba, predict_classes, predict
    prediction = loaded_model.predict(x_data_raw)
    score = loaded_model.evaluate(x_data_raw, y_data_raw, batch_size=128)
    print(prediction.shape)
    print("loss & accuracy :" ,score)
    plt.plot(t,y_data_raw[:,0], color='r', marker="o", label='real')
    #plt.plot(t, prediction, color='b',marker="x", label='prediction') #for predict_classes
    plt.plot(t, prediction[:,0], color='b',marker="x", label='prediction')
    plt.xlabel('time')
    plt.ylabel('Collision Probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('result/Figure_.' + str(i) + '.png')
    plt.clf()


