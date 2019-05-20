import tensorflow as tf
import numpy as np

tf.reset_default_graph()
sess = tf.Session()

new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
new_saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
name = [n.name for n in tf.trainable_variables()]
np.savetxt('./name.txt', name, fmt='%s')
savefile_w = open('Weight.txt', 'w')
savefile_b = open('Bias.txt', 'w')
savefile_gm = open('Gamma.txt', 'w')
savefile_bt = open('Beta.txt', 'w')

# np.savetxt(namefile, [name], fmt="%s")

for n in name:
    print_value = graph.get_tensor_by_name(n)
    mat = print_value.eval(session=sess).transpose()
    # print_value.eval(session=sess).tofile(str(i)+'.txt', sep=' ')

    # np.savetxt(namefile, [n], fmt='%s')
    # namefile.write('\n')
    if 'Net/W' in n:
        for data_slice in mat:
            np.savetxt(savefile_w, [data_slice])
            # savefile_w.write('\n')

    elif 'Net/Variable' in n:
        for data_slice in mat:
            np.savetxt(savefile_b, [data_slice])
            # savefile_b.write('\n')         

    elif 'gamma' in n:
        for data_slice in mat:
            np.savetxt(savefile_gm, [data_slice])
            # savefile_gm.write('\n') 

    elif 'beta' in n:
        for data_slice in mat:
            np.savetxt(savefile_bt, [data_slice])
            # savefile_bt.write('\n') 

savefile_w.close()
savefile_b.close()
savefile_gm.close()
savefile_bt.close()

    # with open(str(i)+'.txt', 'w') as savefile:
    #     for data_slice in mat:
    #         print('shape:', data_slice.shape)
    #         np.savetxt(savefile, [data_slice])
    #         savefile.write('\n')

    # print(n, sess.run(print_value))
    # i = i+1
