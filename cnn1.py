import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
display_step = 10

# Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):  
    with tf.variable_scope(name) as scope:
        shape = [filter_size, filter_size, num_input_channels, num_filters] # Shape of the filter-weights for the convolution      
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05)) # Create new weights (filters) with the given shape
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))# Create new biases, one for each filter

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')# TensorFlow operation for convolution
        layer += biases# Add the biases to the results of the convolution.
        return layer, weights

def new_pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')# TensorFlow operation for convolution
        return layer

def new_relu_layer(input, name):
    with tf.variable_scope(name) as scope: # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        return layer

def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:        
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))# Create new weights and biases.
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        layer = tf.matmul(input, weights) + biases# Multiply the input and weights, and then add the bias-values.
        return layer


layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6, name ="conv1")#Convolutional Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")#Pooling Layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")#RelU layer 1
layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")#Convolutional Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")#Pooling Layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")#RelU layer 2
num_features = layer_relu2.get_shape()[1:4].num_elements()#Flatten Layer
layer_flat = tf.reshape(layer_relu2, [-1, num_features])
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")#Fully-Connected Layer 1
layer_relu3 = new_relu_layer(layer_fc1, name="relu3")#RelU layer 3
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")#Fully-Connected Layer 2


with tf.variable_scope("Softmax"):#Use Softmax function to normalize the output
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

with tf.name_scope("cross_ent"):#Use Cross entropy cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope("optimizer"):#Use Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

with tf.name_scope("accuracy"):#Accuracy
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_epochs = 3
batch_size = 200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in range(num_epochs):# Loop over number of epochs       
        train_accuracy = 0
        for batch in range(0, int(len(data.train.labels)/batch_size)):
            x_batch, y_true_batch = data.train.next_batch(batch_size)# Get a batch of images and labels
            sess.run(optimizer, feed_dict={x: x_batch, y_true: y_true_batch})# Run the optimizer using this batch of training data.
            if epoch % display_step == 0 or epoch == 1:
              acc = sess.run([ accuracy], feed_dict={x: x_batch, y_true: y_true_batch})
              print("Step " + str(epoch) + ", Training Accuracy= " + "{:3f}".format(float(acc)))
        print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256]}))