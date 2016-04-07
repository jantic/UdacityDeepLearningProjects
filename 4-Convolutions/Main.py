

import cPickle as pickle
from numpy.ma import sqrt

import numpy as np
import tensorflow as tf

pickle_file = '/home/citnaj/Desktop/tensorflow/tensorflow/tensorflow/examples/udacity/notMNIST.pickle'

_imageSize = 28
_numLabels = 10
_batchSize = 16
_patchSize = 5
_depth = 16
_numChannels = 1 #grayscale
_numHidden = 64
_numSteps = 1001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, _imageSize, _imageSize, _numChannels)).astype(np.float32)
    labels = (np.arange(_numLabels) == labels[:, None]).astype(np.float32)
    return dataset, labels

#source:  http://arxiv.org/pdf/1502.01852v1.pdf
def calculateOptimalWeightStdDev(numPreviousLayerParams):
    return sqrt(2.0/numPreviousLayerParams)



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
    # Input data.
    tf_train_dataset = tf.placeholder(
    tf.float32, shape=(_batchSize, _imageSize, _imageSize, _numChannels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(_batchSize, _numLabels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [_patchSize, _patchSize, _numChannels, _depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([_depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
      [_patchSize, _patchSize, _depth, _depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[_depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
      [_imageSize // 4 * _imageSize // 4 * _depth, _numHidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[_numHidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
      [_numHidden, _numLabels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[_numLabels]))

  # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(_numSteps):
    offset = (step * _batchSize) % (train_labels.shape[0] - _batchSize)
    batch_data = train_dataset[offset:(offset + _batchSize), :, :, :]
    batch_labels = train_labels[offset:(offset + _batchSize), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))