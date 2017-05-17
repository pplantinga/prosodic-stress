import numpy as np
import re
import string
import tensorflow as tf
from tensorflow.contrib import rnn
from model_new import Seq2SeqModel
from random import shuffle, seed
from helpers import levenshtein, errorRate, line_accuracy, convert2int, readDict

def splitWords(dictionary, testProportion=0.2):
    np.random.seed(120)

    trainX = []
    trainY = []
    testX = []
    testY = []
    testLabels = []

    for word in dictionary:
        if np.random.uniform() < testProportion:
            testX.append(word)
            testY.append(dictionary[word][0])
            testLabels.append(dictionary[word])
        else:
            for pronunciation in dictionary[word]:
                trainX.append(word)
                trainY.append(pronunciation)

    return trainX, trainY, testX, testY, testLabels


########
# Main #
########

dictionary, phoneList, text = readDict("cmudict-0.7b.utf8", stress=False)

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars, 1))
indices_char = dict((i, c) for i, c in enumerate(chars, 1))

label_markers = sorted(list(set(phoneList)))
label_indices = dict((c, i) for i, c in enumerate(label_markers, 2))
indices_label = dict((i, c) for i, c in enumerate(label_markers, 2))

# With some probability, put example in test set
trainX, trainY, testX, testY, testLabels = splitWords(dictionary, testProportion = 0.02)

max_line_len = max([len(i) for i in trainX])
max_label_len = max([len(i) for i in trainY])

trainX = convert2int(trainX, max_line_len, char_indices, backwards=True)
trainY = convert2int(trainY, max_label_len, label_indices, backwards=False)
testX = convert2int(testX, max_line_len, char_indices, backwards=True)
testY = convert2int(testY, max_label_len, label_indices, backwards=False)

seed(100)
z = list(zip(trainX, trainY))
shuffle(z)
trainX, trainY = zip(*z)

# Define network architecture
layer_size = 512
layers = 3
dropout = 0.3
keep_prob = tf.placeholder(tf.float32)
encoder_cell = rnn.DropoutWrapper(rnn.LSTMCell(layer_size), output_keep_prob=keep_prob)
decoder_cell = rnn.DropoutWrapper(rnn.LSTMCell(layer_size * 2), output_keep_prob=keep_prob)
if layers > 1:
    encoder_cell = rnn.MultiRNNCell([encoder_cell] * layers)
    decoder_cell = rnn.MultiRNNCell([decoder_cell] * layers)

model = Seq2SeqModel(encoder_cell=encoder_cell,
        decoder_cell=decoder_cell,
        vocab_size=len(chars) + 3,
        embedding_size=16,
        layers=layers,
        keep_prob=keep_prob,
        attention=True,
        bidirectional=True,
        debug=False)

batch_size = 256
learning_rate = 0.001
last_epoch_accuracy = 0
max_accuracy = 0
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        avgLoss = 0
        count = 1
        Z = list(zip(trainX, trainY))
        shuffle(Z)
        trainX, trainY = zip(*Z)
        for i in range(int((len(trainX)) / batch_size)):
            X = trainX[i*batch_size:(i+1)*batch_size]
            Y = trainY[i*batch_size:(i+1)*batch_size]
            fd = model.make_train_inputs(X, Y, learning_rate, keep_prob=1.-dropout)
            _, l = sess.run([model.train_op, model.loss], fd)

            avgLoss += (l - avgLoss) / (count)
            count += 1

            if i%100 == 0:
                print("Avg loss: " + str(avgLoss))
                avgLoss = 0
                count = 1

        X = trainX[i*batch_size:]
        Y = trainY[i*batch_size:]
        fd = model.make_train_inputs(X, Y, learning_rate, keep_prob=1.-dropout)
        _, l = sess.run([model.train_op, model.loss], fd)
        avgLoss += (l - avgLoss) / (count)
        count += 1
            
        fd = model.make_train_inputs(testX, testY, learning_rate, keep_prob=1.)
        predictions, validLoss = sess.run([model.decoder_prediction_inference, model.loss], fd)
        predictions = [[indices_label[i] for i in p if i in indices_label] for p in predictions.T]

        # Calculate error rate and accuracy
        PER = errorRate(testLabels, predictions)
        accuracy = line_accuracy(testLabels, predictions)
 
        print("Epoch " + str(epoch) + " validation loss: " + str(validLoss))
        print("PER = " + str(PER))
        print("line accuracy = " + str(accuracy))

       
        # If accuracy is less than last epoch's, reduce the learning rate
        if accuracy < last_epoch_accuracy:
            learning_rate *= 0.8
        elif accuracy > max_accuracy:
            max_accuracy = accuracy
            filename = "saved_models_3layer512/char2phone" + str(epoch) + ".ckpt"
            print("Saving in " + filename)
            saver.save(sess, filename)

        last_epoch_accuracy = accuracy

