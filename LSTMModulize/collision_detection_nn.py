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
    wandb.init(project="real_LSTM_Modulize", tensorboard=False)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=[None, num_input], name = "input")
            self.Y = tf.placeholder(tf.int64, shape=[None, num_output], name= "output")
            self.is_train = tf.placeholder(tf.bool, name = "is_train")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.hidden_layers = 3
            self.hidden_neurons = 50

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

            lstms = [tf.nn.rnn_cell.LSTMCell(self.hidden_neurons) for size in range(self.hidden_layers)]
            drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=drop_out) for lstm in lstms]
            cell = tf.contrib.rnn.MultiRNNCell(drops)
            initial_state = cell.zero_state(batch_size, tf.float32)
            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, tf.reshape(self.X, [batch_size, 1,num_input]), initial_state=initial_state)
            lstm_outputs = tf.reshape(lstm_outputs, [batch_size,self.hidden_neurons])

            W5 = tf.get_variable("W5", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
            b5 = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L5 = tf.matmul(lstm_outputs, W5) +b5
            L5 = tf.layers.batch_normalization(L5, training=self.is_train)
            L5 = tf.nn.relu(L5)
            L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

            W6 = tf.get_variable("W6", shape=[self.hidden_neurons, num_output], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
            b6 = tf.Variable(tf.random_normal([num_output]))
            self.logits = tf.matmul(L5, W6) + b6
            tf.identity(self.logits, "logits")
            self.hypothesis = tf.nn.softmax(self.logits)
            self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            # define cost/loss & optimizer
            self.l2_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost + self.l2_reg)
        
        self.prediction = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_mean_error_hypothesis(self, x_test, y_test, keep_prop=1.0, is_train=False):
        return self.sess.run([self.accuracy,  self.l2_reg, self.cost], feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop, self.is_train: is_train})

    def train(self, x_data, y_data, keep_prop=1.0, is_train=True):
        return self.sess.run([self.accuracy, self.l2_reg, self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop, self.is_train: is_train})

    def next_batch(self, num, data):
        x_batch = []
        y_batch = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            x_batch.append(line[0:num_input])
            y_batch.append(line[-num_output:])
            i = i+1

            if i == num:
                break
        return [np.asarray(np.reshape(x_batch, (-1, num_input))), np.asarray(np.reshape(y_batch,(-1,num_output)))]
    def get_hidden_number(self):
        return [self.hidden_layers, self.hidden_neurons]

# input/output number
time_step = 1
num_one_joint_data = 6
num_joint = 6
num_input = num_one_joint_data*num_joint
num_output = 2
output_idx = 6

# parameters
learning_rate = 0.00002 #0.000001
training_epochs = 150
batch_size = 1000 
total_batch = 1307 # joint : 492, random : 1132/ 705 / 449 / 566
total_batch_val = 292 # joint: 105, random: 242/ 151 / 96/ 121
total_batch_test = 275 # joint: 105, random: 242/ 151 / 96 / 121
drop_out = 0.5
regul_factor = 0.000004#0.032
analog_clipping = 0.00


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())


if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out = drop_out
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.time_step = time_step
    wandb.config.total_batch = total_batch
    wandb.config.activation_function = "ReLU"
    wandb.config.training_episode = 1200
    wandb.config.hidden_layers, wandb.config.hidden_neurons = m1.get_hidden_number()
    wandb.config.L2_regularization = regul_factor 
    wandb.config.analog_clipping = analog_clipping

# train my model
train_mse = np.zeros(training_epochs)
validation_mse = np.zeros(training_epochs)

train_cost = np.zeros(training_epochs)
validation_cost = np.zeros(training_epochs)

for epoch in range(training_epochs):
    accu_train = 0
    accu_val = 0
    reg_train = 0
    reg_val = 0
    cost_train = 0
    cost_val = 0

    f = open('../data/random/LSTMModulize/Training_raw_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for i in range(total_batch):
        batch_xs, batch_ys = m1.next_batch(batch_size, rdr)
        c, reg_c, cost,_ = m1.train(batch_xs, batch_ys, drop_out)
        accu_train += c / total_batch
        reg_train += reg_c / total_batch
        cost_train += cost / total_batch

    f_val = open('../data/random/LSTMModulize/Validation_raw_data_.csv', 'r', encoding='utf-8')
    rdr_val = csv.reader(f_val)
    for i in range(total_batch_val):
        batch_xs_val, batch_ys_val = m1.next_batch(batch_size, rdr_val)
        c, reg_c, cost = m1.get_mean_error_hypothesis(batch_xs_val, batch_ys_val)
        accu_val += c / total_batch_val
        reg_val += reg_c / total_batch_val
        cost_val += cost / total_batch_val

    print('Epoch:', '%04d' % (epoch + 1))
    print('Train Accuracy =', '{:.9f}'.format(accu_train))
    print('Validation Accuracy =', '{:.9f}'.format(accu_val))
    print('Train Cost =', '{:.9f}'.format(cost_train), 'Train Regul =', '{:.9f}'.format(reg_train))
    print('Validation Cost =', '{:.9f}'.format(cost_val), 'Validation Regul =', '{:.9f}'.format(reg_val))

    train_mse[epoch] = accu_train
    validation_mse[epoch] = accu_val

    train_cost[epoch] = cost_train
    validation_cost[epoch] = cost_val

    if wandb_use == True:
        wandb.log({'training Accuracy': accu_train, 'validation Accuracy': accu_val})
        wandb.log({'training cost': cost_train, 'training reg': reg_train, 'validation cost': cost_val, 'validation l2_reg': reg_val})

        if epoch % 20 ==0:
            for var in tf.trainable_variables():
                name = var.name
                wandb.log({name: sess.run(var)})


print('Learning Finished!')

f_test = open('../data/random/LSTMModulize/Testing_raw_data_.csv', 'r', encoding='utf-8')
rdr_test = csv.reader(f_test)
accu_test = 0
reg_test = 0
cost_test = 0

for i in range(total_batch_test):
    batch_xs_test, batch_ys_test = m1.next_batch(batch_size, rdr_test)
    c, reg, cost  = m1.get_mean_error_hypothesis(batch_xs_test, batch_ys_test)
    accu_test += c / total_batch_test
    reg_test += reg / total_batch_test
    cost_test += cost / total_batch_test
print('Test Accuracy: ', accu_test)
print('Test Cost: ', cost_test)

elapsed_time = time.time() - start_time
print(elapsed_time)

saver = tf.train.Saver()
saver.save(sess,'model/model.ckpt')

if wandb_use == True:
    saver.save(sess, os.path.join(wandb.run.dir, 'model/model.ckpt'))
    wandb.config.elapsed_time = elapsed_time

#epoch = np.arange(training_epochs)
#plt.plot(epoch, train_mse, 'r', label='train')
#plt.plot(epoch, validation_mse, 'b', label='validation')
#plt.legend()
#plt.xlabel('epoch')
#plt.ylabel('abs error')
#plt.show()
