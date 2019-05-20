import tensorflow as tf
import numpy as np

tf.reset_default_graph()
sess = tf.Session()

new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
new_saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
name = [n.name for n in tf.trainable_variables()]
for j in range(6):
    for i in range(4):
        if i == 0:
            mean_tensor_name = "m1/Joint"+str(j)+"Net/batch_normalization/moving_mean:0"
            variance_tensor_name = "m1/Joint"+str(j)+"Net/batch_normalization/moving_variance:0"
        else:
            mean_tensor_name = "m1/Joint"+str(j)+"Net/batch_normalization_"+str(i)+"/moving_mean:0"
            variance_tensor_name = "m1/Joint"+str(j)+"Net/batch_normalization_"+str(i)+"/moving_variance:0"
        name.append(mean_tensor_name)
        name.append(variance_tensor_name)
name.append("m1/ConcatenateNet/batch_normalization/moving_mean:0")
name.append("m1/ConcatenateNet/batch_normalization/moving_variance:0")

np.savetxt('./name.txt', name, fmt='%s')
savefile_w = open('Weight.txt', 'w')
savefile_b = open('Bias.txt', 'w')
savefile_gm = open('Gamma.txt', 'w')
savefile_bt = open('Beta.txt', 'w')
savefile_bm = open('Mean.txt', 'w')
savefile_bv = open('Variance.txt', 'w')
savefile_data_normal = open('InputMinMax.txt', 'w')

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

    if 'moving_mean' in n:
        np.savetxt(savefile_bm, mat)

    elif 'moving_variance' in n:
        np.savetxt(savefile_bv, mat)

Input_data_max = np.array([[14.1546000000000, 16.6524000000000, 9.01470000000000, 5.76740000000000, 2.41570000000000, 1.45500000000000],
                                [179.801900000000, 69.9101000000000, 89.8839000000000, 179.838500000000, 134.845000000000, 179.769200000000],
                                [89.3654000000000, 63.0918000000000, 65.7428000000000, 83.8778000000000, 79.5955000000000, 90.1153000000000],
                                [179.782200000000, 69.9044000000000, 89.8785000000000, 179.834900000000, 134.841000000000, 179.762600000000],
                                [88.6940000000000, 62.6816000000000, 65.1385000000000, 82.9356000000000, 79.4121000000000, 89.8661000000000],
                                [4.14100000000000, 81.6980000000000, 24.7633000000000, 4.23410000000000, 0.869800000000000, 0.0154000000000000]])
Input_data_min = np.array([[-11.0963000000000, -17.5971000000000, -9.36700000000000, -5.85700000000000, -2.68450000000000, -1.67120000000000],
                                [-179.937800000000, -69.8896000000000, -89.8450000000000, -179.941400000000, -134.935800000000, -179.479200000000],
                                [-86.0404000000000, -60.4827000000000, -64.3995000000000, -89.7486000000000, -82.2063000000000, -64.9344000000000],
                                [-179.937200000000, -69.8824000000000, -89.8339000000000, -179.938100000000, -134.932400000000, -179.474000000000],
                                [-85.1504000000000, -60.3310000000000, -64.0053000000000, -88.7053000000000, -81.9368000000000, -64.6215000000000],
                                [-4.00330000000000, -82.5923000000000, -25.3464000000000, -4.26440000000000, -0.786900000000000, -0.0115000000000000]])
Input_data_max_min = np.concatenate((Input_data_max.T, Input_data_min.T), axis=0)
np.savetxt(savefile_data_normal, Input_data_max_min, fmt='%f')


savefile_w.close()
savefile_b.close()
savefile_gm.close()
savefile_bt.close()
savefile_bm.close()
savefile_bv.close()
savefile_data_normal.close()
