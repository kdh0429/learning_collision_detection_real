import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import wandb
import os

wandb_use = False
start_time = time.time()
if wandb_use == True:
    wandb.init(project="real_CNN_Modulize", tensorboard=False)

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
            self.hidden_layers = 0
            self.hidden_neurons = 32
            self.filt_conv1 = 15
            self.filt_conv2 = 10
            #self.X_image=tf.placeholder(tf.float32, [None, 5, 6], name="input_image")
            #self.X_image = tf.placeholder(tf.float32, shape=[None, time_step, 6], name = "input_image")

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            for i in range(6):
                with tf.variable_scope("Joint"+str(i)+"Net"):
                    X_image = tf.reshape(self.X[:, num_one_joint_data*i:num_one_joint_data*(i+1)], shape=[-1, 5, 6, 1])
                    # for j in range(time_step):
                    #     self.X_image = tf.concat([self.X_image_temp, self.X[:, num_one_joint_data*i + 6*j:num_one_joint_data*i + 6*(j+1)]], 0)
                       
                        ## make X data to the shape of image (5x6)
                        ##[t_i(k)       q_i(k)      q_dot_i(k)      q_d_i(k)    q_dot_d_i(k)    t_dyn_i(k)  ]
                        ##[t_i(k-1)     q_i(k-1)    q_dot_i(k-1)    q_d_i(k-1)  q_dot_d_i(k-1)  t_dyn_i(k-1)]
                        ##[t_i(k-2)     q_i(k-2)    q_dot_i(k-2)    q_d_i(k-2)  q_dot_d_i(k-2)  t_dyn_i(k-2)]
                        ##[t_i(k-3)     q_i(k-3)    q_dot_i(k-3)    q_d_i(k-3)  q_dot_d_i(k-3)  t_dyn_i(k-3)]
                        ##[t_i(k-4)     q_i(k-4)    q_dot_i(k-4)    q_d_i(k-4)  q_dot_d_i(k-4)  t_dyn_i(k-4)]
                    
                    #filt_conv1 = tf.Variable(tf.glorot_uniform_initializer()((3, 6, 1, 11)))
                    #b_conv1 = tf.Variable(tf.random_normal([11]))
                    #L1 = tf.nn.conv2d(X_image, filt_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
                    L1 = tf.layers.conv2d(inputs= X_image, filters= self.filt_conv1, kernel_size= [3, 6], padding="Valid", activation=tf.nn.relu, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), kernel_regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                    L1 = tf.layers.batch_normalization(L1, training=self.is_train)
                    L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                    L1 = tf.reshape(L1, shape=[-1, self.filt_conv1, 3, 1, 1]) #(batch, depth, height, width, channels)
                    self.hidden_layers += 1

                    # filt_conv2 = tf.Variable(tf.glorot_uniform_initializer()((3, 3, 1, 3)))
                    # b_conv2 = tf.Variable(tf.random_normal([3]))
                    # L2 = tf.nn.conv2d(L1, filt_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
                    L2 = tf.layers.conv3d(inputs= L1, filters= self.filt_conv2, kernel_size= [3, 3, 1], padding="Valid", activation=tf.nn.relu, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), kernel_regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                    L2 = tf.layers.batch_normalization(L2, training=self.is_train)
                    L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                    self.hidden_layers += 1
                    L2 = tf.reshape(L2, [-1, 130])
                  
                    # W1 = tf.get_variable("W1", shape=[num_one_joint_data, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                    # b1 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                    # L1 = tf.matmul(self.X[:, num_one_joint_data*i:num_one_joint_data*(i+1)], W1) +b1
                    # L1 = tf.nn.relu(L1)
                    # L1 = tf.layers.batch_normalization(L1, training=self.is_train)
                    # L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

                    # W2 = tf.get_variable("W2", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                    # b2 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                    # L2 = tf.matmul(L1, W2) +b2
                    # L2 = tf.nn.relu(L2)
                    # L2 = tf.layers.batch_normalization(L2, training=self.is_train)
                    # L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                    # self.hidden_layers += 1

                    # W3 = tf.get_variable("W3", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                    # b3 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                    # L3 = tf.matmul(L2, W3) +b3
                    # L3 = tf.nn.relu(L3)
                    # L3 = tf.layers.batch_normalization(L3, training=self.is_train)
                    # L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
                    # self.hidden_layers += 1

                    # W4 = tf.get_variable("W4", shape=[self.hidden_neurons, 1], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                    # b4 = tf.Variable(tf.random_normal([1]))
                    # L4 = tf.matmul(L3, W4) +b4
                    # L4 = tf.nn.relu(L4)
                    # L4 = tf.layers.batch_normalization(L4, training=self.is_train)
                    # L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob, name="L4")

                    if(i == 0):
                        self.LConcat = L2
                    else:
                        self.LConcat = tf.concat([self.LConcat, L2],1)

            with tf.variable_scope("ConcatenateNet"):
                W5 = tf.get_variable("W5", shape=[self.LConcat.shape[1], self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                b5 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                L5 = tf.matmul(self.LConcat, W5) +b5
                L5 = tf.nn.relu(L5)
                L5 = tf.layers.batch_normalization(L5, training=self.is_train)
                L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

                W6 = tf.get_variable("W6", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                b6 = tf.Variable(tf.random_normal([self.hidden_neurons]))
                L6 = tf.matmul(L5, W6) +b6
                L6 = tf.nn.relu(L6)
                L6 = tf.layers.batch_normalization(L6, training=self.is_train)
                L6 = tf.nn.dropout(L6, keep_prob=self.keep_prob)

                W7 = tf.get_variable("W7", shape=[self.hidden_neurons, num_output], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regul_factor))
                b7 = tf.Variable(tf.random_normal([num_output]))
                self.logits = tf.matmul(L6, W7) + b7
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

    def next_batch(self, size, iteration, data):
        x_batch = []
        y_batch = []

        # x_batch = data[size*(iteration-1):size, 
        i = 0

        for line in data:
            line = [float(i) for i in line]
            x_batch.append(line[0:num_input])
            y_batch.append(line[-num_output:])
            i = i+1

            if i == size:
                break
        return [np.asarray(np.reshape(x_batch, (-1, num_input))), np.asarray(np.reshape(y_batch,(-1,num_output)))]
    def get_hidden_number(self):
        return [self.hidden_layers, self.hidden_neurons]
    def get_cnn_filter_number(self):
        return [self.filt_conv1, self.filt_conv2]

# input/output number
time_step = 5
num_one_joint_data = 6*time_step #30
num_joint = 6
num_input = num_one_joint_data*num_joint #180
num_output = 2
output_idx = 6

# parameters
learning_rate = 0.000007 #0.000001
training_epochs = 1000
batch_size = 1000 
total_batch = 492 # joint : 492, random : 449
total_batch_val = 105 # joint: 105, random: 96
total_batch_test = 105 # joint: 105, random: 96
drop_out = 0.85
regul_factor = 0.035
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
    wandb.config.filt_conv1, wandb.config.filter_conv2 = m1.get_cnn_filter_number()
    wandb.config.L2_regularization = regul_factor 
    wandb.config.analog_clipping = analog_clipping
    

# train my model
train_mse = np.zeros(training_epochs)
validation_mse = np.zeros(training_epochs)

train_cost = np.zeros(training_epochs)
validation_cost = np.zeros(training_epochs)

# data_train = tuple(open('./data/random/FCModulize/training_data_.csv', 'r', encoding='utf-8'))
# data_val = tuple(open('./data/random/FCModulize/validation_data_.csv', 'r', encoding='utf-8'))

for epoch in range(training_epochs):
    accu_train = 0
    accu_val = 0
    reg_train = 0
    reg_val = 0
    cost_train = 0
    cost_val = 0
    f = open('./data/random/FCModulize/training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for i in range(total_batch):
        batch_xs, batch_ys = m1.next_batch(batch_size, i, rdr)
        c, reg_c, cost,_ = m1.train(batch_xs, batch_ys, drop_out)
        accu_train += c / total_batch
        reg_train += reg_c / total_batch
        cost_train += cost / total_batch


    f_val = open('./data/random/FCModulize/validation_data_.csv', 'r', encoding='utf-8')
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

f_test = open('./data/random/FCModulize/testing_data_.csv', 'r', encoding='utf-8')
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
saver.save(sess,'model/MCNNmodel.ckpt')

if wandb_use == True:
    saver.save(sess, os.path.join(wandb.run.dir, 'model/MCNNmodel.ckpt'))
    wandb.config.elapsed_time = elapsed_time

#epoch = np.arange(training_epochs)
#plt.plot(epoch, train_mse, 'r', label='train')
#plt.plot(epoch, validation_mse, 'b', label='validation')
#plt.legend()
#plt.xlabel('epoch')
#plt.ylabel('abs error')
#plt.show()
