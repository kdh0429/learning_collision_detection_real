import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input_param = 2
time_step = 3
num_one_joint_data = time_step*num_input_param+1
num_joint = 2  #joint index
num_input = time_step*num_input_param
num_output = 1
false_negative = 0.0
false_positive = 0.0


for i in range(1056): #1056 joint 963 random
    path = '../data/random/TorqueLearning/TestingDivide/Testing_raw_data_' + str(i+1) + '.csv'
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_data_raw = []
    y_data_raw = []

    for line in rdr:
        line = [float(i) for i in line]
        x_data_raw.append(line[num_one_joint_data*(num_joint-1):num_one_joint_data*num_joint-1])
        y_data_raw.append(line[num_one_joint_data*num_joint-1])
    t = range(len(x_data_raw))
    t = np.reshape(t,(-1,1))
    x_data_raw = np.reshape(x_data_raw, (-1, num_input))
    y_data_raw = np.reshape(y_data_raw, (-1, num_output))

    tf.reset_default_graph()
    sess = tf.Session()

    new_saver = tf.train.import_meta_graph('./model/model_j1.ckpt.meta')
    new_saver.restore(sess, './model/model_j1.ckpt')

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("m1/input:0")
    y = graph.get_tensor_by_name("m1/output:0")
    keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
    is_train = graph.get_tensor_by_name("m1/is_train:0")
    pred = graph.get_tensor_by_name("m1/prediction:0")

    torque_est = sess.run(pred, feed_dict={x: x_data_raw, keep_prob: 1.0, is_train:False})

    # prediction = np.argmax(torque_est, 1)
    # correct_prediction = np.equal(prediction, np.argmax(y_data_raw, 1))

    accuracy = np.mean(torque_est/y_data_raw)

    print("Accuracy : %f" % accuracy)


    plt.plot(t,y_data_raw[:,0], color='r', marker="o", label='real')
    plt.plot(t,torque_est[:,0], color='b',marker="x", label='prediction')
    plt.xlabel('time')
    plt.ylabel('torquey')
    plt.legend()
    plt.ylim(0,1)
    plt.savefig('Figure_' + str(i)+'.png')
    plt.clf()
    #plt.show()