import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input_1st = 18
num_output_1st = 6
num_input_2nd = num_output_1st
num_output_2nd = 2


for i in range(192):
    path = '../data/Resi/TestingDivide/Testing_raw_data_' + str(i+1) + '.csv'
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_data_raw_1st = []
    y_data_raw_1st = []
    x_data_raw_2nd = []
    y_data_raw_2nd = []
    not_collision_num = 0
    for line in rdr:
        line = [float(i) for i in line]
        if line[42] == 0:
            not_collision_num += 1
        x_data_raw_1st.append(line[0:num_input_1st])
        y_data_raw_1st.append(line[num_input_1st:num_input_1st+num_output_1st])
        x_data_raw_2nd.append(line[num_input_1st:num_input_1st+num_output_1st])
        y_data_raw_2nd.append(line[-num_output_2nd:])
    t = range(len(x_data_raw_1st))
    t = np.reshape(t,(-1,1))
    x_data_raw_1st = np.reshape(x_data_raw_1st, (-1, num_input_1st))
    y_data_raw_1st = np.reshape(y_data_raw_1st, (-1, num_output_1st))
    x_data_raw_2nd = np.reshape(x_data_raw_2nd, (-1, num_input_2nd))
    y_data_raw_2nd = np.reshape(y_data_raw_2nd, (-1, num_output_2nd))

    tf.reset_default_graph()
    sess = tf.Session()

    new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    new_saver.restore(sess, 'model/model.ckpt')

    graph = tf.get_default_graph()
    x_1st = graph.get_tensor_by_name("m1/input_1st:0")
    hypo_1st = graph.get_tensor_by_name("m1/hypothesis_1st:0")
    keep_prob_1st = graph.get_tensor_by_name("m1/keep_prob_1st:0")
    it_train_1st = graph.get_tensor_by_name("m1/is_train_1st:0")
    x_2nd = graph.get_tensor_by_name("m1_1/input_2nd:0")
    hypo_2nd = graph.get_tensor_by_name("m1_1/hypothesis_2nd:0")
    keep_prob_2nd = graph.get_tensor_by_name("m1_1/keep_prob_2nd:0")

    hypo1st = sess.run(hypo_1st, feed_dict={x_1st: x_data_raw_1st, keep_prob_1st: 1.0, it_train_1st:False})
    resi = x_data_raw_2nd - hypo1st
    hypo2nd = sess.run(hypo_2nd, feed_dict={x_2nd: resi, keep_prob_2nd : 1.0})

    prediction = np.argmax(hypo2nd, 1)
    correct_prediction = np.equal(prediction, np.argmax(y_data_raw_2nd, 1))
    accuracy = np.mean(correct_prediction)

    print("Accuracy : %f" % accuracy)

    # for j in range(6):
    #     plt.subplot(6,1,j+1)
    #     plt.plot(t,x_data_raw_2nd[:,j], color='r', label='real')
    #     plt.plot(t,hypo1st[:,j], color='b', label='prediction')

    plt.plot(t,y_data_raw_2nd[:,0], color='r', marker="o", label='real')
    plt.plot(t,hypo2nd[:,0], color='b',marker="x", label='prediction')
    plt.xlabel('time')
    plt.ylabel('Collision Probability')
    plt.legend()
    plt.ylim(0,1)
    plt.savefig('Figure_' + str(i)+'.png')
    plt.clf()
    #plt.show()