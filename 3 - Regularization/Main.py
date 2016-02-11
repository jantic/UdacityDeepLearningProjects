# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import cPickle as pickle
import numpy as np
import tensorflow as tf

pickle_file = '/home/citnaj/Desktop/tensorflow/tensorflow/tensorflow/examples/udacity/notMNIST.pickle'

_imageSize = 28
_numLabels = 10
_trainSubset = 60000
_batchSize = 128
_hiddenLayers = [1024,128]
_numInputs = _imageSize * _imageSize
_learningRate = 0.00005
_numSteps = 10000
_regularizationRate = 0.0005
_dropoutKeepRate = 0.5

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

def validateNumHiddenLayers(numHiddenLayers):
    if numHiddenLayers < 1:
        raise ValueError('Number of hidden layers must be >= 1')

def generateHiddenLayerKey(layerNum):
    return 'h' + str(layerNum)

def generateHiddenLayer(layerNum, previousLayer, weights, biases, training):
    key = generateHiddenLayerKey(layerNum)
    hiddenLayer = tf.nn.relu(tf.matmul(previousLayer, weights[key]) + biases[key])
    if training:
        hiddenLayer = tf.nn.dropout(hiddenLayer, _dropoutKeepRate)
    return hiddenLayer

def multilayerNetwork(inputs, weights, biases, numHiddenLayers, training):
    validateNumHiddenLayers(numHiddenLayers)

    hiddenLayer = generateHiddenLayer(1, inputs, weights, biases, training)

    for layerNum in xrange(numHiddenLayers+1):
        if layerNum > 1:
            hiddenLayer = generateHiddenLayer(layerNum, hiddenLayer, weights, biases, training)

    return tf.matmul(hiddenLayer, weights['out']) + biases['out']

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, _imageSize * _imageSize)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(_numLabels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def generateWeights(hiddenLayers, numInputs, numLabels):
    numHiddenLayers = hiddenLayers.__len__()
    validateNumHiddenLayers(numHiddenLayers)
    weights = {}

    numHiddenFeatures = hiddenLayers[0]
    weights[generateHiddenLayerKey(1)] = tf.Variable(tf.truncated_normal([numInputs, numHiddenFeatures]))

    for layerNum in xrange(numHiddenLayers+1):
        if layerNum > 1:
            previousNumHiddenFeatures = numHiddenFeatures
            numHiddenFeatures = hiddenLayers[layerNum-1]
            weights[generateHiddenLayerKey(layerNum)] = tf.Variable(tf.truncated_normal([previousNumHiddenFeatures, numHiddenFeatures]))

    weights['out'] = tf.Variable(tf.truncated_normal([numHiddenFeatures, numLabels]))
    return weights

def generateBiases(hiddenLayers,  numLabels):
    numHiddenLayers = hiddenLayers.__len__()
    validateNumHiddenLayers(numHiddenLayers)
    biases = {}

    numHiddenFeatures = hiddenLayers[0]
    biases[generateHiddenLayerKey(1)] = tf.Variable(tf.zeros([numHiddenFeatures]))

    for layerNum in xrange(numHiddenLayers+1):
        if layerNum > 1:
            numHiddenFeatures = hiddenLayers[layerNum-1]
            biases[generateHiddenLayerKey(layerNum)] = tf.Variable(tf.zeros([numHiddenFeatures]))

    biases['out'] = tf.Variable(tf.zeros([numLabels]))
    return biases

def generateRegularizers(weights, biases, numHiddenLayers):
    validateNumHiddenLayers(numHiddenLayers)
    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['h1'])

    for layerNum in xrange(numHiddenLayers+1):
        if layerNum > 1:
            regularizers = regularizers + tf.nn.l2_loss(weights['h' + str(layerNum)]) + tf.nn.l2_loss(biases['h' + str(layerNum)])

    regularizers = regularizers + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])
    return regularizers

def generateLossCalc(weights, biases, numHiddenLayers, trainingNetwork, trainingLabels, regularizationRate):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(trainingNetwork, trainingLabels))
    regularizers = generateRegularizers(weights, biases, numHiddenLayers)
    loss += regularizationRate * regularizers
    return loss

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
    tf_train_dataset = tf.placeholder(tf.float32, shape=(_batchSize, _numInputs))
    tf_train_labels = tf.placeholder(tf.float32, shape=(_batchSize, _numLabels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    numHiddenLayers = _hiddenLayers.__len__()
    weights = generateWeights(_hiddenLayers, _numInputs, _numLabels)
    biases = generateBiases(_hiddenLayers, _numLabels)
    trainingNetwork = multilayerNetwork(tf_train_dataset, weights, biases, numHiddenLayers, True)
    loss = generateLossCalc(weights, biases, numHiddenLayers, trainingNetwork, tf_train_labels, _regularizationRate)
    optimizer = tf.train.GradientDescentOptimizer(_learningRate).minimize(loss)

    train_prediction = tf.nn.softmax(trainingNetwork)
    valid_prediction = tf.nn.softmax(multilayerNetwork(tf_valid_dataset, weights, biases, numHiddenLayers, False))
    test_prediction = tf.nn.softmax(multilayerNetwork(tf_test_dataset, weights, biases, numHiddenLayers, False))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print "Initialized"
    for step in xrange(_numSteps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (np.random.randint(1, _trainSubset) * _batchSize) % (train_labels.shape[0] - _batchSize)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + _batchSize), :]
        batch_labels = train_labels[offset:(offset + _batchSize), :]
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