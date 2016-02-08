# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import cPickle as pickle
import numpy as np
import tensorflow as tf

pickle_file = '/home/citnaj/Desktop/tensorflow/tensorflow/tensorflow/examples/udacity/notMNIST.pickle'

image_size = 28
num_labels = 10
train_subset = 10000
batch_size = 128
num_hidden_params = 1024 # 1st layer num features
num_inputs = image_size * image_size # MNIST data input
learningRate = 0.05
num_steps = 3001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

def multilayerNetwork(_X, _weights, _biases):
    hidden1 = tf.nn.relu(tf.matmul(_X, _weights['h1']) + _biases['h1'])
    outputLayer = tf.matmul(hidden1, _weights['out']) + _biases['out']
    return outputLayer

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels



with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print 'Training set', train_dataset.shape, train_labels.shape
    print 'Validation set', valid_dataset.shape, valid_labels.shape
    print 'Test set', test_dataset.shape, test_labels.shape




train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Test set', test_dataset.shape, test_labels.shape


graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_inputs))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = {
        'h1':tf.Variable(tf.truncated_normal([num_inputs, num_hidden_params])),
        'out':tf.Variable(tf.truncated_normal([num_hidden_params, num_labels]))
    }

    biases = {
        'h1':tf.Variable(tf.zeros([num_hidden_params])),
        'out':tf.Variable(tf.zeros([num_labels]))
    }
    # Training computation.

    trainingNetwork = multilayerNetwork(tf_train_dataset, weights, biases)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(trainingNetwork, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(trainingNetwork)
    valid_prediction = tf.nn.softmax(multilayerNetwork(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(multilayerNetwork(tf_test_dataset, weights, biases))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print "Initialized"
    for step in xrange(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (np.random.randint(1, train_subset) * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print "Minibatch loss at step", step, ":", l
            print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
            print "Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels)

    print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)