import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import wandb
import os

wandb_use = True
start_time = time.time()
if wandb_use == True:
    wandb.init(project="real_residual_time", tensorboard=False)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_1st_network()
        self._build_2nd_network()

    def _build_1st_network(self):
        with tf.variable_scope(self.name):
            self.X_1st = tf.placeholder(tf.float32, shape=[None, num_input_1st], name = "input_1st")
            self.Y_1st = tf.placeholder(tf.float32, shape=[None, num_output_1st], name= "output_1st")
            self.keep_prob_1st = tf.placeholder(tf.float32, name="keep_prob_1st")
            self.is_train_1st = tf.placeholder(tf.bool, name = "is_train_1st")
            self.hidden_layers_1st = 0
            self.hidden_neurons_1st = 300

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W1_1st = tf.get_variable("W1_1st", shape=[num_input_1st, self.hidden_neurons_1st], initializer=tf.contrib.layers.xavier_initializer())
            b1_1st = tf.Variable(tf.random_normal([self.hidden_neurons_1st]))
            L1_1st = tf.matmul(self.X_1st, W1_1st) +b1_1st
            L1_1st = tf.nn.relu(L1_1st)
            L1_1st = tf.layers.batch_normalization(L1_1st, training=self.is_train_1st)
            L1_1st = tf.nn.dropout(L1_1st, keep_prob=self.keep_prob_1st)

            W2_1st = tf.get_variable("W2_1st", shape=[self.hidden_neurons_1st, self.hidden_neurons_1st], initializer=tf.contrib.layers.xavier_initializer())
            b2_1st = tf.Variable(tf.random_normal([self.hidden_neurons_1st]))
            L2_1st = tf.matmul(L1_1st, W2_1st) +b2_1st
            L2_1st = tf.nn.relu(L2_1st)
            L2_1st = tf.layers.batch_normalization(L2_1st, training=self.is_train_1st)
            L2_1st = tf.nn.dropout(L2_1st, keep_prob=self.keep_prob_1st)
            self.hidden_layers_1st += 1

            W3_1st = tf.get_variable("W3_1st", shape=[self.hidden_neurons_1st, self.hidden_neurons_1st], initializer=tf.contrib.layers.xavier_initializer())
            b3_1st = tf.Variable(tf.random_normal([self.hidden_neurons_1st]))
            L3_1st = tf.matmul(L2_1st, W3_1st) +b3_1st
            L3_1st = tf.nn.relu(L3_1st)
            L3_1st = tf.layers.batch_normalization(L3_1st, training=self.is_train_1st)
            L3_1st = tf.nn.dropout(L3_1st, keep_prob=self.keep_prob_1st)
            self.hidden_layers_1st += 1

            W4_1st = tf.get_variable("W4_1st", shape=[self.hidden_neurons_1st, self.hidden_neurons_1st], initializer=tf.contrib.layers.xavier_initializer())
            b4_1st = tf.Variable(tf.random_normal([self.hidden_neurons_1st]))
            L4_1st = tf.matmul(L3_1st, W4_1st) +b4_1st
            L4_1st = tf.nn.relu(L4_1st)
            L4_1st = tf.layers.batch_normalization(L4_1st, training=self.is_train_1st)
            L4_1st = tf.nn.dropout(L4_1st, keep_prob=self.keep_prob_1st)
            self.hidden_layers_1st += 1

            W5_1st = tf.get_variable("W5_1st", shape=[self.hidden_neurons_1st, self.hidden_neurons_1st], initializer=tf.contrib.layers.xavier_initializer())
            b5_1st = tf.Variable(tf.random_normal([self.hidden_neurons_1st]))
            L5_1st = tf.matmul(L4_1st, W5_1st) +b5_1st
            L5_1st = tf.nn.relu(L5_1st)
            L5_1st = tf.layers.batch_normalization(L5_1st, training=self.is_train_1st)
            L5_1st = tf.nn.dropout(L5_1st, keep_prob=self.keep_prob_1st)
            self.hidden_layers_1st += 1

            W6_1st = tf.get_variable("W6_1st", shape=[self.hidden_neurons_1st, self.hidden_neurons_1st], initializer=tf.contrib.layers.xavier_initializer())
            b6_1st = tf.Variable(tf.random_normal([self.hidden_neurons_1st]))
            L6_1st = tf.matmul(L5_1st, W6_1st) +b6_1st
            L6_1st = tf.nn.relu(L6_1st)
            L6_1st = tf.layers.batch_normalization(L6_1st, training=self.is_train_1st)
            L6_1st = tf.nn.dropout(L6_1st, keep_prob=self.keep_prob_1st)
            self.hidden_layers_1st += 1

            W7_1st = tf.get_variable("W7_1st", shape=[self.hidden_neurons_1st, num_output_1st], initializer=tf.contrib.layers.xavier_initializer())
            b7_1st = tf.Variable(tf.random_normal([num_output_1st]))
            self.hypothesis_1st = tf.matmul(L6_1st, W7_1st)+b7_1st
            self.hypothesis_1st = tf.identity(self.hypothesis_1st, "hypothesis_1st")

            # define cost/loss & optimizer
            self.cost_1st = tf.reduce_mean(tf.reduce_mean(tf.square(self.hypothesis_1st - self.Y_1st)))
            self.update_ops_1st = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops_1st):
                self.optimizer_1st = tf.train.AdamOptimizer(learning_rate= learning_rate_1st).minimize(self.cost_1st)
        
        self.mean_error_1st = tf.reduce_mean(tf.abs(self.Y_1st-self.hypothesis_1st))
        self.mean_error_1st = tf.identity(self.mean_error_1st, "mean_error_1st")

    def _build_2nd_network(self):
        with tf.variable_scope(self.name):
            self.X_2nd = tf.placeholder(tf.float32, shape=[None, num_input_2nd], name = "input_2nd")
            self.Y_2nd = tf.placeholder(tf.float32, shape=[None, num_output_2nd], name= "output_2nd")
            self.keep_prob_2nd = tf.placeholder(tf.float32, name="keep_prob_2nd")
            self.is_train_2nd = tf.placeholder(tf.bool, name = "is_train_2nd")
            self.hidden_layers_2nd = 0
            self.hidden_neurons_2nd = 300

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W1_2nd = tf.get_variable("W1_2nd", shape=[num_input_2nd, self.hidden_neurons_2nd], initializer=tf.contrib.layers.xavier_initializer())
            b1_2nd = tf.Variable(tf.random_normal([self.hidden_neurons_2nd]))
            L1_2nd = tf.matmul(self.X_2nd, W1_2nd) +b1_2nd
            L1_2nd = tf.nn.relu(L1_2nd)
            #L1_2nd = tf.layers.batch_normalization(L1_2nd, training=self.is_train_2nd)
            L1_2nd = tf.nn.dropout(L1_2nd, keep_prob=self.keep_prob_2nd)

            W2_2nd = tf.get_variable("W2_2nd", shape=[self.hidden_neurons_2nd, self.hidden_neurons_2nd], initializer=tf.contrib.layers.xavier_initializer())
            b2_2nd = tf.Variable(tf.random_normal([self.hidden_neurons_2nd]))
            L2_2nd = tf.matmul(L1_2nd, W2_2nd) +b2_2nd
            L2_2nd = tf.nn.relu(L2_2nd)
            #L2_2nd = tf.layers.batch_normalization(L2_2nd, training=self.is_train_2nd)
            L2_2nd = tf.nn.dropout(L2_2nd, keep_prob=self.keep_prob_2nd)
            self.hidden_layers_2nd += 1

            W3_2nd = tf.get_variable("W3_2nd", shape=[self.hidden_neurons_2nd, self.hidden_neurons_2nd], initializer=tf.contrib.layers.xavier_initializer())
            b3_2nd = tf.Variable(tf.random_normal([self.hidden_neurons_2nd]))
            L3_2nd = tf.matmul(L2_2nd, W3_2nd) +b3_2nd
            L3_2nd = tf.nn.relu(L3_2nd)
            #L3_2nd = tf.layers.batch_normalization(L3_2nd, training=self.is_train_2nd)
            L3_2nd = tf.nn.dropout(L3_2nd, keep_prob=self.keep_prob_2nd)
            self.hidden_layers_2nd += 1

            W4_2nd = tf.get_variable("W4_2nd", shape=[self.hidden_neurons_2nd, self.hidden_neurons_2nd], initializer=tf.contrib.layers.xavier_initializer())
            b4_2nd = tf.Variable(tf.random_normal([self.hidden_neurons_2nd]))
            L4_2nd = tf.matmul(L3_2nd, W4_2nd) +b4_2nd
            L4_2nd = tf.nn.relu(L4_2nd)
            #L4_2nd = tf.layers.batch_normalization(L4_2nd, training=self.is_train_2nd)
            L4_2nd = tf.nn.dropout(L4_2nd, keep_prob=self.keep_prob_2nd)
            self.hidden_layers_2nd += 1

            W5_2nd = tf.get_variable("W5_2nd", shape=[self.hidden_neurons_2nd, num_output_2nd], initializer=tf.contrib.layers.xavier_initializer())
            b5_2nd = tf.Variable(tf.random_normal([num_output_2nd]))
            self.logits = tf.matmul(L4_2nd,W5_2nd)+b5_2nd
            self.hypothesis_2nd = tf.nn.softmax(self.logits)
            self.hypothesis_2nd = tf.identity(self.hypothesis_2nd, "hypothesis_2nd")

            # define cost/loss & optimizer
            self.cost_2nd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels= self.Y_2nd))
            #self.update_ops_2nd = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #with tf.control_dependencies(self.update_ops_2nd):
            self.optimizer_2nd = tf.train.AdamOptimizer(learning_rate= learning_rate_2nd).minimize(self.cost_2nd)

        self.prediction = tf.argmax(self.hypothesis_2nd, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y_2nd, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_hypothesis_1st(self, x_test, keep_prop=1.0, is_train_1st=False):
        return self.sess.run(self.hypothesis_1st, feed_dict={self.X_1st: x_test, self.keep_prob_1st: keep_prop, self.is_train_1st:is_train_1st})
    def get_mean_error_hypothesis_1st(self, x_test, y_test, keep_prop=1.0, is_train_1st=False):
        return self.sess.run([self.mean_error_1st, self.hypothesis_1st, self.X_1st, self.Y_1st], feed_dict={self.X_1st: x_test, self.Y_1st: y_test, self.keep_prob_1st: keep_prop, self.is_train_1st:is_train_1st})
    def get_mean_error_hypothesis_2nd(self, x_test, y_test, keep_prop=1.0, is_train_2nd=False):
        return self.sess.run([self.accuracy, self.hypothesis_2nd, self.X_2nd, self.Y_2nd], feed_dict={self.X_2nd: x_test, self.Y_2nd: y_test, self.keep_prob_2nd: keep_prop, self.is_train_2nd:is_train_2nd})

    def train_1st(self, x_data, y_data, keep_prop=1.0, is_train_1st=True):
        return self.sess.run([self.mean_error_1st, self.optimizer_1st], feed_dict={
            self.X_1st: x_data, self.Y_1st: y_data, self.keep_prob_1st: keep_prop, self.is_train_1st: is_train_1st})
    def train_2nd(self, x_data, y_data, keep_prop=1.0, is_train_2nd=True):
        return self.sess.run([self.accuracy, self.optimizer_2nd], feed_dict={
            self.X_2nd: x_data, self.Y_2nd: y_data, self.keep_prob_2nd: keep_prop, self.is_train_2nd:is_train_2nd})

    def next_batch_1st(self, num, data):
        x_batch_1st_network = []
        y_batch_1st_network = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            if(line[42] == 0):
                x_batch_1st_network.append(line[0:num_input_1st])
                y_batch_1st_network.append(line[num_input_1st:num_input_1st+num_output_1st])
            i = i+1

            if i == num:
                break
        return [np.asarray(np.reshape(x_batch_1st_network, (-1, num_input_1st))), np.asarray(np.reshape(y_batch_1st_network,(-1,num_output_1st)))]
    
    def next_batch_2nd(self, num, data):
        x_batch_1st_network = []
        y_batch_1st_network = []
        x_batch_2nd_network = []
        y_batch_2nd_network = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            x_batch_1st_network.append(line[0:num_input_1st])
            y_batch_1st_network.append(line[num_input_1st:num_input_1st+num_output_1st])
            x_batch_2nd_network.append(line[num_input_1st:num_input_1st+num_output_1st])
            y_batch_2nd_network.append(line[-num_output_2nd:])
            i = i+1

            if i == num:
                break
        return [np.asarray(np.reshape(x_batch_1st_network, (-1, num_input_1st))), np.asarray(np.reshape(y_batch_1st_network,(-1,num_output_1st))), \
                np.asarray(np.reshape(x_batch_2nd_network, (-1, num_input_2nd))), np.asarray(np.reshape(y_batch_2nd_network,(-1,num_output_2nd)))]

    def get_hidden_number(self):
        return [self.hidden_layers_1st, self.hidden_neurons_1st, self.hidden_layers_2nd, self.hidden_neurons_2nd]

# input/output number
num_input_1st = 36*5
num_output_1st = 6
num_input_2nd = num_output_1st
num_output_2nd = 2

# parameters
learning_rate_1st = 0.00001 #0.000001
learning_rate_2nd = 0.00001
training_epochs = 1000
batch_size = 1000
total_batch = 224
drop_out = 0.85
regul_factor = 0.00000

# loading testing data
f_test = open('../data/ResiTime/testing_data_.csv', 'r', encoding='utf-8')
rdr_test = csv.reader(f_test)
x_data_test_1st = []
y_data_test_1st = []
x_data_test_1st_all =[]
y_data_test_1st_all=[]
x_data_test_2nd = []
y_data_test_2nd = []

for line in rdr_test:
    line = [float(i) for i in line]
    if line[42]==0 :
        x_data_test_1st.append(line[0:num_input_1st])
        y_data_test_1st.append(line[num_input_1st:num_input_1st+num_output_1st])
    x_data_test_1st_all.append(line[0:num_input_1st])
    y_data_test_1st_all.append(line[num_input_1st:num_input_1st+num_output_1st])

    x_data_test_2nd.append(line[num_input_1st:num_input_1st+num_output_1st])
    y_data_test_2nd.append(line[-num_output_2nd:])

x_data_test_1st = np.reshape(x_data_test_1st, (-1, num_input_1st))
y_data_test_1st = np.reshape(y_data_test_1st, (-1, num_output_1st))
x_data_test_1st_all = np.reshape(x_data_test_1st_all, (-1, num_input_1st))
y_data_test_1st_all = np.reshape(y_data_test_1st_all, (-1, num_output_1st))
x_data_test_2nd = np.reshape(x_data_test_2nd, (-1, num_input_2nd))
y_data_test_2nd = np.reshape(y_data_test_2nd, (-1, num_output_2nd))

# load validation data
f_val = open('../data/ResiTime/validation_data_.csv', 'r', encoding='utf-8')
rdr_val = csv.reader(f_val)
x_data_val_1st = []
y_data_val_1st = []
x_data_val_1st_all = []
y_data_val_1st_all = []
x_data_val_2nd = []
y_data_val_2nd = []
for line in rdr_val:
    line = [float(i) for i in line]
    if line[42]==0 :
        x_data_val_1st.append(line[0:num_input_1st])
        y_data_val_1st.append(line[num_input_1st:num_input_1st+num_output_1st])
    x_data_val_1st_all.append(line[0:num_input_1st])
    y_data_val_1st_all.append(line[num_input_1st:num_input_1st+num_output_1st])
    x_data_val_2nd.append(line[num_input_1st:num_input_1st+num_output_1st])
    y_data_val_2nd.append(line[-num_output_2nd:])

x_data_val_1st = np.reshape(x_data_val_1st, (-1, num_input_1st))
y_data_val_1st = np.reshape(y_data_val_1st, (-1, num_output_1st))
x_data_val_1st_all = np.reshape(x_data_val_1st_all, (-1, num_input_1st))
y_data_val_1st_all = np.reshape(y_data_val_1st_all, (-1, num_output_1st))
x_data_val_2nd = np.reshape(x_data_val_2nd, (-1, num_input_2nd))
y_data_val_2nd = np.reshape(y_data_val_2nd, (-1, num_output_2nd))


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())


if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate_1st = learning_rate_1st
    wandb.config.learning_rate_2nd = learning_rate_2nd
    wandb.config.drop_out = drop_out
    wandb.config.num_input_1st = num_input_1st
    wandb.config.num_output_1st = num_output_1st
    wandb.config.num_input_2nd = num_input_2nd
    wandb.config.num_output_2nd = num_output_2nd
    wandb.config.total_batch = total_batch
    wandb.config.activation_function = "ReLU"
    wandb.config.training_episode = 1200
    wandb.config.hidden_layers_1st, wandb.config.hidden_neurons_1st, wandb.config.hidden_layers_2nd, wandb.config.hidden_neurons_2nd = m1.get_hidden_number()
    wandb.config.L2_regularization = regul_factor 

# Train 1st model
train_mse_1st = np.zeros(training_epochs)
validation_mse_1st = np.zeros(training_epochs)

for epoch in range(training_epochs):
    eror_train_1st = 0
    f = open('../data/ResiTime/training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    for i in range(total_batch):
        batch_xs_1st, batch_ys_1st= m1.next_batch_1st(batch_size, rdr)
        c,_ = m1.train_1st(batch_xs_1st, batch_ys_1st)
        eror_train_1st += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1))
    print('1st Train error =', '{:.9f}'.format(eror_train_1st))

    [err_val, hypo, x_val, y_val] = m1.get_mean_error_hypothesis_1st(x_data_val_1st, y_data_val_1st)
    print('1st Validation error:', '{:.9f}'.format(err_val))

    train_mse_1st[epoch] = eror_train_1st
    validation_mse_1st[epoch] = err_val

    if wandb_use == True:
        wandb.log({'1st Training Error': eror_train_1st, '1st Validation Error': err_val})

        if epoch % 20 == 0:
            for var in tf.trainable_variables():
                name = var.name
                wandb.log({name: sess.run(var)})

print('1st Learning Finished!')
[err_test, hypo, x_test, y_test] = m1.get_mean_error_hypothesis_1st(x_data_test_1st, y_data_test_1st)
print('Test Error: ', err_test)


# Train 2nd model
train_acc_2nd = np.zeros(training_epochs)
validation_acc_2nd = np.zeros(training_epochs)

for epoch in range(training_epochs):
    acc_train_2nd = 0
    f = open('../data/Resi/training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    for i in range(total_batch):
        batch_xs_1st, batch_ys_1st, batch_xs_2nd, batch_ys_2nd= m1.next_batch_2nd(batch_size, rdr)
        hypo_1st = m1.get_hypothesis_1st(batch_xs_1st, drop_out)
        resi = batch_xs_2nd - hypo_1st
        c,_ = m1.train_2nd(resi, batch_ys_2nd, drop_out)
        acc_train_2nd += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1))
    print('2nd Train Accuracy =', '{:.9f}'.format(acc_train_2nd))


    hypo_val = m1.get_hypothesis_1st(x_data_val_1st_all, drop_out)
    resi_val = x_data_val_2nd - hypo_val
    [acc_val, hypo, x_val, y_val] = m1.get_mean_error_hypothesis_2nd(resi_val, y_data_val_2nd)
    print('2nd Validation Accuracy:', '{:.9f}'.format(acc_val))

    train_acc_2nd[epoch] = acc_train_2nd
    validation_acc_2nd[epoch] = acc_val

    if wandb_use == True:
        wandb.log({'Training Accuracy 2nd': acc_train_2nd, 'Validation Accuracy 2nd': acc_val})

        if epoch % 20 == 0:
            for var in tf.trainable_variables():
                name = var.name
                wandb.log({name: sess.run(var)})


print('2nd Learning Finished!')
hypo_test = m1.get_hypothesis_1st(x_data_test_1st_all, drop_out)
resi_test = x_data_test_2nd - hypo_test
[acc_test, hypo, x_test, y_test] = m1.get_mean_error_hypothesis_2nd(resi_test, y_data_test_2nd)
print('Test Accuracy: ', acc_test)



elapsed_time = time.time() - start_time
print(elapsed_time)

saver = tf.train.Saver()
saver.save(sess,'model/model.ckpt')

if wandb_use == True:
    saver.save(sess, os.path.join(wandb.run.dir, 'model/model.ckpt'))
    wandb.config.elapsed_time = elapsed_time

epoch = np.arange(training_epochs)
plt.plot(epoch, train_acc_2nd, 'r', label='train')
plt.plot(epoch, validation_acc_2nd, 'b', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('abs error')
plt.show()
