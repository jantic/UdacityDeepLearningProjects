

import _pickle as pickle
from numpy.ma import sqrt

import numpy as np
import tensorflow as tf

pickle_file = 'C:/Users/Jason/Documents/GitHub/tensorflow/tensorflow/examples/udacity/notMNIST.pickle'

_imageSize = 28
_numLabels = 10
_batchSize = 16
_patchSize = 5
_depth = 16
_numChannels = 1 #grayscale
_numParamsPerHiddenLayer = 2000
_numSteps = 300001
_dropoutKeepProb = 0.5
_regularizationRate = 0.00001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, _imageSize, _imageSize, _numChannels)).astype(np.float32)
    labels = (np.arange(_numLabels) == labels[:, None]).astype(np.float32)
    return dataset, labels

#source:  http://arxiv.org/pdf/1502.01852v1.pdf
def calculateOptimalWeightStdDev(numPreviousLayerParams):
    return sqrt(2.0/numPreviousLayerParams)

def validateNumHiddenLayers(numHiddenLayers):
    if numHiddenLayers < 1:
        raise ValueError('Number of hidden layers must be >= 1')

def generateRegularizers(weights, biases, numHiddenLayers):
    validateNumHiddenLayers(numHiddenLayers)
    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['h1'])

    for layerNum in range(numHiddenLayers+1):
        if layerNum > 1:
            regularizers = regularizers + tf.nn.l2_loss(weights['h' + str(layerNum)]) + tf.nn.l2_loss(biases['h' + str(layerNum)])

    regularizers = regularizers + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])
    return regularizers

def generateLossCalc(weights, biases, numHiddenLayers, trainingNetwork, trainingLabels, regularizationRate):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(trainingNetwork, trainingLabels))
    regularizers = generateRegularizers(weights, biases, numHiddenLayers)
    loss += regularizationRate * regularizers
    return loss

def generateHiddenLayerKey(layerNum):
    return 'h' + str(layerNum)


def generateWeights(hiddenLayers, numInputs, numLabels):
    numHiddenLayers = hiddenLayers.__len__()
    validateNumHiddenLayers(numHiddenLayers)
    weights = {}

    numHiddenFeatures = hiddenLayers[0]
    stddev = calculateOptimalWeightStdDev(numInputs)
    weights[generateHiddenLayerKey(1)] = tf.Variable(tf.truncated_normal([numInputs, numHiddenFeatures], 0, stddev))

    for layerNum in range(numHiddenLayers+1):
        if layerNum > 1:
            previousNumHiddenFeatures = numHiddenFeatures
            numHiddenFeatures = hiddenLayers[layerNum-1]
            stddev = calculateOptimalWeightStdDev(previousNumHiddenFeatures)
            weights[generateHiddenLayerKey(layerNum)] = tf.Variable(tf.truncated_normal([previousNumHiddenFeatures, numHiddenFeatures], 0, stddev))

    stddev = calculateOptimalWeightStdDev(numHiddenFeatures)
    weights['out'] = tf.Variable(tf.truncated_normal([numHiddenFeatures, numLabels], 0, stddev))
    return weights

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)




train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


graph = tf.Graph()
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(_batchSize, _imageSize, _imageSize, _numChannels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(_batchSize, _numLabels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    keep_prob = tf.placeholder(tf.float32)


    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([_patchSize, _patchSize, _numChannels, _depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([_depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([_patchSize, _patchSize, _depth, _depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[_depth]))
    layer3InputSize = _imageSize // 4 * _imageSize // 4 * _depth
    layer3_weights = tf.Variable(tf.truncated_normal([layer3InputSize, _numParamsPerHiddenLayer], stddev=calculateOptimalWeightStdDev(layer3InputSize)))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[_numParamsPerHiddenLayer]))
    layer4_weights = tf.Variable(tf.truncated_normal([_numParamsPerHiddenLayer, _numLabels], stddev=calculateOptimalWeightStdDev(_numParamsPerHiddenLayer)))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[_numLabels]))

  # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv + layer1_biases)
        hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
        pooled = tf.nn.max_pool(hidden1_drop,[1, 2, 2, 1],[1, 2, 2, 1],padding='SAME')
        conv = tf.nn.conv2d(pooled, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv + layer2_biases)
        hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
        pooled = tf.nn.max_pool(hidden2_drop,[1, 2, 2, 1],[1, 2, 2, 1],padding='SAME')
        shape = pooled.get_shape().as_list()
        reshape = tf.reshape(pooled, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        hidden3_drop = tf.nn.dropout(hidden3, keep_prob)
        return tf.matmul(hidden3_drop, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    weights = {}
    weights[generateHiddenLayerKey(1)] = layer1_weights;
    weights[generateHiddenLayerKey(2)] = layer2_weights;
    weights[generateHiddenLayerKey(3)] = layer3_weights;
    weights['out'] = layer4_weights;

    biases = {}
    biases[generateHiddenLayerKey(1)] = layer1_biases;
    biases[generateHiddenLayerKey(2)] = layer2_biases;
    biases[generateHiddenLayerKey(3)] = layer3_biases;
    biases['out'] = layer4_biases;

    numHiddenLayers = weights.__len__()-1;
    loss = generateLossCalc(weights, biases, numHiddenLayers, logits, tf_train_labels, _regularizationRate)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(_numSteps):
    offset = (step * _batchSize) % (train_labels.shape[0] - _batchSize)
    batch_data = train_dataset[offset:(offset + _batchSize), :, :, :]
    batch_labels = train_labels[offset:(offset + _batchSize), :]
    train_feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: _dropoutKeepProb}
    validation_feed_dict = {keep_prob: _dropoutKeepProb}
    test_feed_dict = {keep_prob: 1.0}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=train_feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(validation_feed_dict), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(test_feed_dict), test_labels))