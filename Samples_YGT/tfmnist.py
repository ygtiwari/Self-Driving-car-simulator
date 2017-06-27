import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def create_conv_layer(input_data, input_channel, output_channel, filter_shape, pool_shape, name):  # input_channel = 1 in case of greyscale, output_channel = no. of filters

    #convulation dot product starts
    conv_shape = [filter_shape[0], filter_shape[1], input_channel, output_channel]

    weight = tf.Variable(tf.truncated_normal(conv_shape, stddev = 0.03), name =name + "_w")
    bias = tf.Variable(tf.truncated_normal([output_channel]), name = name + "_b")

    out_layer = tf.nn.conv2d(input_data, weight, [1, 1, 1, 1], padding = "SAME")  # [1, 1, 1, 1] is the stride, padding = SAME ensures no loss of dimensions

    out_layer += bias

    out_layer = tf.nn.relu(out_layer) # activation function
    #convulation dot product ends

    #now pooling operation starts

    pool_size = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]

    out_layer = tf.nn.max_pool(out_layer, ksize = pool_size, strides = strides, padding = "SAME")

    return out_layer


def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    learning_rate = 0.0001
    epochs = 10
    batch_size = 50

    x = tf.placeholder(tf.float32, [None, 784])
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])

    y = tf.placeholder(tf.float32, [None, 10])

    layer1 = create_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], "layer1")
    layer2 = create_conv_layer(layer1, 32, 64, [5, 5], [2, 2], "layer2")

    flattened = tf.flatten(layer2, [-1, 3136])

    wd1 = tf.Variable(tf.truncated_normal([3136, 1000], stddev = 0.03), name = "wd1")
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev = 0.03), name = "bd1")

    out_layer1 = tf.matmul(flattened, wd1) + bd1
    out_layer1 = tf.nn.relu(out_layer1)

    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev = 0.03), name = "wd2")
    bd2 = tf.Variable(tf.truncated_normal([10], stddev = 0.03), name = "bd2")

    out_layer2 = tf.matmul(out_layer1, wd2) + bd2
    y_ = tf.nn.softmax(out_layer2)
    

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out_layer2, labels = y))

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()

    #rest of the code is not written because it is basically just the calling of cross_entropy and optimizer by using with session loop
                       
    
    
                         
