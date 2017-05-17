
from g2p.helpers import trainTestSplit
from re import sub
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell as LSTM, DropoutWrapper as Dropout, MultiRNNCell as Multi
import numpy as np

class ProsodicStressModel:

    UNKNOWN = "UNK"
    PAD = ""
    train = {}
    test = {}

    def __init__(self, lines, phones, lex, labels,
            position = 0,
            test_fraction = 0.1,
            embedding_size = 15,
            embedding_dropout = 0.4,
            hidden_is_recurrent = True,
            hidden_layer_size = 100,
            hidden_layers = 2,
            hidden_dropout = 0.4,
            output_dropout = 0.2):

        # Embedding params
        self.embed_size = embedding_size
        self.embed_dropout = embedding_dropout

        # Hidden layer params
        self.hidden_is_recurrent = hidden_is_recurrent
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = hidden_layers
        self.hidden_dropout = hidden_dropout

        # Output layer params
        self.output_dropout = output_dropout

        # _initialize stuffs
        self._make_inputs(lines, phones, lex, labels, position, test_fraction)
        self._make_graph()

    # Organize the data into one-hot vectors
    def _make_inputs(self, lines, phones, lex, labels, position, test_fraction):
        
        graphs = [self._split_line(i, clean=False) for i in lines]
        lines = [self._split_line(i, clean=True) for i in lines]

        # separate lines and labels (per-line stuff)
        self.train['lines'], self.test['lines'] = trainTestSplit(lines, test_fraction, position)
        self.train['lex'], self.test['lex'] = trainTestSplit(lex, test_fraction, position)
        self.train['labels'], self.test['labels'] = trainTestSplit(labels, test_fraction, position)

        # Calculate lengths (test length cannot be longer than longest train length)
        self.train['lengths'] = np.array([len(line) for line in self.train['lines']])
        self.max_line_len = max(self.train['lengths'])
        self.test['lengths'] = [min(len(line), self.max_line_len) for line in self.test['lines']]
        
        # Only use the first label for each training example.
        self.train['labels'] = [labels[0] for labels in self.train['labels']]

        # separate graphs and phones (per-syllable stuff)
        self.train['graphs'], self.test['graphs'] = trainTestSplit(graphs, test_fraction, position)
        self.train['phones'], self.test['phones'] = trainTestSplit(phones, test_fraction, position)
        
        # pad each line to max_line_len and reshape to 1-d
        self.train['graphs'] = self._pad(self.train['graphs'], self.max_line_len)
        self.test['graphs']  = self._pad(self.test['graphs'], self.max_line_len)
        self.train['phones'] = self._pad(self.train['phones'], self.max_line_len)
        self.test['phones']  = self._pad(self.test['phones'], self.max_line_len)

        # Calculate lengths
        self.train['graph_lengths'] = [len(graph) for graph in self.train['graphs']]
        self.max_graph_len = max(self.train['graph_lengths'])
        self.test['graph_lengths']  = [min(len(graph), self.max_graph_len) for graph in self.test['graphs']]

        self.train['phone_lengths'] = [len(phone) for phone in self.train['phones']]
        self.max_phone_len = max(self.train['phone_lengths'])
        self.test['phone_lengths']  = [min(len(phone), self.max_phone_len) for phone in self.test['phones']]

        # Create a lookup table for converting from tokens to indexes, and vice-versa
        self.syl_indices,   self.indices_syl   = self._vocab(self.train['lines'])
        self.lex_indices,   self.indices_lex   = self._vocab(self.train['lex'])
        self.graph_indices, self.indices_graph = self._vocab(self.train['graphs'])
        self.phone_indices, self.indices_phone = self._vocab(self.train['phones'])
        self.label_indices, self.indices_label = self._vocab(self.train['labels'])

        # Convert all inputs to integer indexes
        self.train['lines']  = self._convert2index(self.train['lines'], self.max_line_len, self.syl_indices)
        self.train['lex']    = self._convert2index(self.train['lex'], self.max_line_len, self.lex_indices)
        self.train['graphs'] = self._convert2index(self.train['graphs'], self.max_graph_len, self.graph_indices)
        self.train['phones'] = self._convert2index(self.train['phones'], self.max_phone_len, self.phone_indices)
        self.test['lines']   = self._convert2index(self.test['lines'], self.max_line_len, self.syl_indices)
        self.test['lex']     = self._convert2index(self.test['lex'], self.max_line_len, self.lex_indices)
        self.test['graphs']  = self._convert2index(self.test['graphs'], self.max_graph_len, self.graph_indices)
        self.test['phones']  = self._convert2index(self.test['phones'], self.max_phone_len, self.phone_indices)

        # Reshape back to 1 entry per line
        self.train['graphs'] = np.reshape(self.train['graphs'], [-1, self.max_line_len, self.max_graph_len])
        self.train['phones'] = np.reshape(self.train['phones'], [-1, self.max_line_len, self.max_phone_len])
        self.test['graphs'] = np.reshape(self.test['graphs'], [-1, self.max_line_len, self.max_graph_len])
        self.test['phones'] = np.reshape(self.test['phones'], [-1, self.max_line_len, self.max_phone_len])
        self.train['graph_lengths'] = np.reshape(self.train['graph_lengths'], [-1, self.max_line_len])
        self.train['phone_lengths'] = np.reshape(self.train['phone_lengths'], [-1, self.max_line_len])
        self.test['graph_lengths'] = np.reshape(self.test['graph_lengths'], [-1, self.max_line_len])
        self.test['phone_lengths'] = np.reshape(self.test['phone_lengths'], [-1, self.max_line_len])

        # Convert labels to indices
        self.train['labels'] = self._convert2index(self.train['labels'], self.max_line_len, self.label_indices)

    # Split a line into syllables. _if clean is true, remove punctuation and caps
    def _split_line(self, line, clean=True):
        
        if clean:
            line = sub("[^a-z ']", ' ', line.lower())

        else:
            line = line.replace('"', '')
            line = line.replace(",", ", ")
            line = sub("(\s*-\s*)+", " -", line)
            line = sub("\s*-$", "", line)

        return line.split()


    # _pad each line to length and reshape to 1-d
    def _pad(self, data, length):
        return [x[i] if i < len(x) else self.PAD for x in data for i in range(length)]

    # Generate a mapping from data to indexes and vice versa
    def _vocab(self, data):

        # Count all instances of each token, 
        counts = {self.UNKNOWN: 0}
        for x in data:
            for token in x:
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1

        # Word types with only one token are excluded from the vocab (count as unknown)
        vocab = [wordType for wordType in counts if counts[wordType] != 1]

        # Create mappings
        vocab_indices = {c: i for i, c in enumerate(vocab)}
        indices_vocab = {i: c for i, c in enumerate(vocab)}

        return vocab_indices, indices_vocab

    def _convert2onehot(self, data, length, indices):
        index = np.zeros([len(data), length, len(indices)], dtype=np.float32)

        for i, line in enumerate(data):
            for j, token in enumerate(line):
                if j < length:
                    if token in indices:
                        index[i, j, indices[token]] = 1.
                    else:
                        index[i, j, indices[self.UNKNOWN]] = 1.

        return index

    # Convert data array to indexes
    def _convert2index(self, data, length, indices):
        index = np.zeros([len(data), length], dtype=np.int32)

        for i, line in enumerate(data):
            for j, token in enumerate(line):
                if j < length:
                    if token in indices:
                        index[i, j] = indices[token]
                    else:
                        index[i, j] = indices[self.UNKNOWN]

        return index

    # Create the computation graph
    def _make_graph(self):

        self._make_placeholders()
        
        with tf.variable_scope("lex"):
            lex_rep = self._make_embedding(self.lex, len(self.lex_indices))

        with tf.variable_scope("graph"):
            graph_rep = self._make_embedding(self.graphs, len(self.graph_indices))
            graph_rep = self._make_recurrent_embedding(graph_rep, self.graph_lengths)
        
        with tf.variable_scope("phone"):
            phone_rep = self._make_embedding(self.phones, len(self.phone_indices))
            phone_rep = self._make_recurrent_embedding(phone_rep, self.phone_lengths)

        with tf.variable_scope("lines"):
            lines_rep = self._make_embedding(self.lines, len(self.syl_indices))
        
        # Concatenate all representations to a single feature vector
        features = tf.concat([graph_rep, lex_rep, phone_rep, lines_rep], -1)
        #features = tf.concat([graph_rep, lines_rep], -1)
        #features = self.lines

        with tf.variable_scope("hidden"):
            hidden_output = self._make_hidden(features)

        # output will be used for predictions
        with tf.variable_scope("output"):
            #x = tf.reshape(features, [-1, len(self.syl_indices)])
            #self.output = self._make_output(x)
            self.output = self._make_output(hidden_output)
            
        self._make_train_op()


    # This will hold the data at compute time
    def _make_placeholders(self):
        
        # Data inputs
        self.lines  = tf.placeholder(tf.int32, [None, self.max_line_len], name="lines")
        #self.lines  = tf.placeholder(tf.float32, [None, self.max_line_len, len(self.syl_indices)], name="lines")
        self.lex    = tf.placeholder(tf.int32, [None, self.max_line_len], name="lex")
        self.graphs = tf.placeholder(tf.int32, [None, self.max_graph_len], name="graphs")
        self.phones = tf.placeholder(tf.int32, [None, self.max_phone_len], name="phones")
        self.line_lengths  = tf.placeholder(tf.int32, shape=[None], name="line_lengths")
        self.graph_lengths = tf.placeholder(tf.int32, shape=[None], name="graph_lengths")
        self.phone_lengths = tf.placeholder(tf.int32, shape=[None], name="phone_lengths")

        # And outputs
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

        # Dropout input
        self.embed_keep_prob  = tf.placeholder(tf.float32, name="embed_keep_prob")
        self.hidden_keep_prob = tf.placeholder(tf.float32, name="hidden_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

        # _learning rate
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")


    def _make_embedding(self, inputs, vocab_size):

        embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, self.embed_size], -1.0, 1.0))

        embedding = tf.nn.embedding_lookup(embedding_matrix, inputs)
        embedding = tf.nn.dropout(embedding, keep_prob = self.embed_keep_prob)
        return embedding


    def _make_recurrent_embedding(self, inputs, input_lengths):

        fw_rnn_cell = Dropout(LSTM(self.embed_size), self.embed_keep_prob)
        bw_rnn_cell = Dropout(LSTM(self.embed_size), self.embed_keep_prob)

        _, ((_, out_fw), (_, out_bw)) = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_cell,
                bw_rnn_cell,
                inputs,
                sequence_length = input_lengths,
                dtype = tf.float32)

        output = tf.concat((out_fw, out_bw), -1)
        return tf.reshape(output, [-1, self.max_line_len, 2 * self.embed_size])

    def _make_hidden(self, features):

        if self.hidden_is_recurrent:
            fw_rnn_cell = Multi([Dropout(LSTM(self.hidden_layer_size), self.hidden_keep_prob) for _ in range(self.hidden_layers)])
            bw_rnn_cell = Multi([Dropout(LSTM(self.hidden_layer_size), self.hidden_keep_prob) for _ in range(self.hidden_layers)])

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_rnn_cell,
                    bw_rnn_cell,
                    features,
                    sequence_length = self.line_lengths,
                    dtype = tf.float32)

            output = tf.concat(outputs, 2)

            return tf.reshape(output, [-1, 2 * self.hidden_layer_size])

        else:
            input_size = features.get_shape().as_list()[-1]

            x = tf.reshape(features, [-1, input_size])

            for i in range(self.hidden_layers):

                W = tf.get_variable(
                        "W" + str(i),
                        shape = [input_size, self.hidden_layer_size],
                        dtype = tf.float32,
                        initializer = tf.contrib.layers.xavier_initializer(uniform=False))
                b = tf.get_variable(
                        "b" + str(i),
                        shape = [self.hidden_layer_size],
                        dtype = tf.float32,
                        initializer = tf.zeros_initializer())
                x = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W) + b), keep_prob = self.hidden_keep_prob)
                input_size = self.hidden_layer_size

            return x

    def _make_output(self, hidden_output):

        input_size = hidden_output.get_shape().as_list()[-1]

        W = tf.get_variable(
                "W",
                shape = [input_size, len(self.label_indices)],
                dtype = tf.float32,
                initializer = tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(
                "b",
                shape = [len(self.label_indices)],
                dtype = tf.float32,
                initializer = tf.zeros_initializer())

        output = tf.nn.dropout(tf.matmul(hidden_output, W) + b, self.output_keep_prob)
        
        return tf.reshape(output, [-1, self.max_line_len, len(self.label_indices)])

    def _make_train_op(self):

        # CRF decoding
        log_likelihood, self.transition_params = \
            tf.contrib.crf.crf_log_likelihood(self.output, self.labels, self.line_lengths)
        loss = tf.reduce_mean(-log_likelihood)

        # Adam optimization with gradient clipping
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)

    # Create a list of feed dictionaries, one per batch
    def batchify(self, batch_size, lr):

        train_count = len(self.train['lines'])

        # Randomize entries
        indexes = np.arange(train_count)
        np.random.shuffle(indexes)
        self.train['lines']    = self.train['lines'][indexes]
        self.train['phones']   = self.train['phones'][indexes]
        self.train['graphs']   = self.train['graphs'][indexes]
        self.train['lex']      = self.train['lex'][indexes]
        self.train['lengths']  = self.train['lengths'][indexes]
        self.train['labels']   = self.train['labels'][indexes]
        self.train['phone_lengths'] = self.train['phone_lengths'][indexes]
        self.train['graph_lengths'] = self.train['graph_lengths'][indexes]
        
        # Go over train lines in batch_size jumps
        iterator = range(0, train_count, batch_size)

        return [self._make_fd(self.train, start, batch_size, lr) for start in iterator]

    # Create the feed dictionary for each batch
    def _make_fd(self, data, start = 0, batch_size = None, train = True, lr = 0.002):

        if batch_size:
            stop = min(start + batch_size, len(data['lines']))
        else:
            stop = len(data['lines'])

        #embed_start = start * self.max_line_len
        #embed_stop = stop * self.max_line_len
        phones = np.reshape(data['phones'][start : stop], [-1, self.max_phone_len])
        graphs = np.reshape(data['graphs'][start : stop], [-1, self.max_graph_len])
        phone_lengths = np.reshape(data['phone_lengths'][start : stop], [-1])
        graph_lengths = np.reshape(data['graph_lengths'][start : stop], [-1])

        fd = {  self.lines:         data['lines'][start : stop],
                self.line_lengths:  data['lengths'][start : stop],
                self.phones:        phones,
                self.phone_lengths: phone_lengths,
                self.graphs:        graphs,
                self.graph_lengths: graph_lengths,
                self.lex:           data['lex'][start : stop],
                }

        if train:
            fd[self.labels]           = data['labels'][start : stop]
            fd[self.embed_keep_prob]  = 1. - self.embed_dropout
            fd[self.hidden_keep_prob] = 1. - self.hidden_dropout
            fd[self.output_keep_prob] = 1. - self.output_dropout
            fd[self.learning_rate]    = lr
        else:
            fd[self.embed_keep_prob]  = 1.
            fd[self.hidden_keep_prob] = 1.
            fd[self.output_keep_prob] = 1.

        return fd

    # Make all predictions on the test set
    def predict(self, sess):

        predictions = []

        fd = self._make_fd(self.test, train = False)
        logits, trans_param = sess.run([self.output, self.transition_params], fd)
        for logit, sequence_length in zip(logits, self.test['lengths']):
            logit = logit[:sequence_length]
            sequence, _ = tf.contrib.crf.viterbi_decode(logit, trans_param)
            predictions.append([self.indices_label[i] for i in sequence if i in self.indices_label])

        return predictions
