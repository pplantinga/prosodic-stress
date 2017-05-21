
from re import sub
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell as LSTM, DropoutWrapper as Dropout, MultiRNNCell as Multi
import numpy as np

class ProsodicStressModel:

    UNKNOWN = "UNK"
    PAD = ""
    train = {}
    test = {}

    def __init__(self, 
            embedding_size = 16,
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


    # Compute sizes needed to build the graph
    def compute_size(self, lines, lex, phones, graphs, labels):

        # Calculate max lengths
        self.max_line_len = max(len(line) for line in lines)

        graphs = self._pad(graphs, self.max_line_len)
        phones = self._pad(phones, self.max_line_len)

        self.max_graph_len = max(len(graph) for graph in graphs)
        self.max_phone_len = max(len(phone) for phone in phones)

        # Create a lookup table so we know how big to make matrices
        syl_indices,   _ = self._vocab(lines)
        lex_indices,   _ = self._vocab(lex)
        graph_indices, _ = self._vocab(graphs)
        phone_indices, _ = self._vocab(phones)
        label_indices, _ = self._vocab([label[0] for label in labels])

        # Compute vocab sizes
        self.syl_vocab_size = len(syl_indices)
        self.lex_vocab_size = len(lex_indices)
        self.graph_vocab_size = len(graph_indices)
        self.phone_vocab_size = len(phone_indices)
        self.label_vocab_size = len(label_indices)

    # Change the data into numeric form for computation
    def _make_3D_input(self, data, max_len, indices):

        # Convert to indexes
        lengths = np.array([len(line) for line in data])
        data = self._convert2index(data, max_len, indices)

        # Convert back to 3-d
        data = np.reshape(data, [-1, self.max_line_len, max_len])
        lengths = np.reshape(lengths, [-1, self.max_line_len])

        return data, lengths

    # Organize the inputs into matrix form
    def make_inputs(self, train, test):

        self.train = {}
        self.test = {}

        # Only use the first label for each training example.
        train['labels'] = [labels[0] for labels in train['labels']]

        # Pad 3-d data and convert to 2-d
        train['graphs'] = self._pad(train['graphs'], self.max_line_len)
        test['graphs']  = self._pad(test['graphs'], self.max_line_len)
        train['phones'] = self._pad(train['phones'], self.max_line_len)
        test['phones']  = self._pad(test['phones'], self.max_line_len)

        # Create vocabulary for each input
        self.syl_indices,   self.indices_syl   = self._vocab(train['lines'])
        self.lex_indices,   self.indices_lex   = self._vocab(train['lex'])
        self.graph_indices, self.indices_graph = self._vocab(train['graphs'])
        self.phone_indices, self.indices_phone = self._vocab(train['phones'])
        self.label_indices, self.indices_label = self._vocab(train['labels'])

        # Calculate line lengths
        self.train['lengths'] = np.array([len(line) for line in train['lines']])
        self.test['lengths']  = np.array([len(line) for line in test['lines']])

        # Convert 2-d inputs to matrix form
        self.train['lines']  = self._convert2index(train['lines'], self.max_line_len, self.syl_indices)
        self.test['lines']   = self._convert2index(test['lines'], self.max_line_len, self.syl_indices)
        self.train['lex']    = self._convert2index(train['lex'], self.max_line_len, self.lex_indices)
        self.test['lex']     = self._convert2index(test['lex'], self.max_line_len, self.lex_indices)
        self.train['labels'] = self._convert2index(train['labels'], self.max_line_len, self.label_indices)
        self.test['labels']  = test['labels']

        # Convert 3-d inputs to matrix form
        self.train['graphs'], self.train['graph_lengths'] = \
                self._make_3D_input(train['graphs'], self.max_graph_len, self.graph_indices)
        self.test['graphs'], self.test['graph_lengths'] = \
                self._make_3D_input(test['graphs'], self.max_graph_len, self.graph_indices)
        self.train['phones'], self.train['phone_lengths'] = \
                self._make_3D_input(train['phones'], self.max_phone_len, self.phone_indices)
        self.test['phones'], self.test['phone_lengths'] = \
                self._make_3D_input(test['phones'], self.max_phone_len, self.phone_indices)

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

    def _convert2onehot(self, data, length, width, indices):
        index = np.zeros([len(data), length, width], dtype=np.float32)

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
    def make_graph(self):

        self._make_placeholders()
        
        with tf.variable_scope("lex"):
            lex_rep = self._make_embedding(self.lex, self.lex_vocab_size)

        with tf.variable_scope("graph"):
            graph_rep = self._make_embedding(self.graphs, self.graph_vocab_size)
            graph_rep = self._make_recurrent_embedding(graph_rep, self.graph_lengths)
        
        with tf.variable_scope("phone"):
            phone_rep = self._make_embedding(self.phones, self.phone_vocab_size)
            phone_rep = self._make_recurrent_embedding(phone_rep, self.phone_lengths)

        with tf.variable_scope("lines"):
            lines_rep = self._make_embedding(self.lines, self.syl_vocab_size)
        
        # Concatenate all representations to a single feature vector
        features = tf.concat([graph_rep, lex_rep, phone_rep, lines_rep], -1)
        #features = tf.concat([graph_rep, lines_rep], -1)
        #features = lines_rep#self.lines

        with tf.variable_scope("hidden"):
            hidden_output = self._make_hidden(features)

        # output will be used for predictions
        with tf.variable_scope("output"):
            #x = tf.reshape(features, [-1, self.embed_size])#self.syl_vocab_size])
            #self.output = self._make_output(x)
            self.output = self._make_output(hidden_output)
            
        with tf.variable_scope("train_op"):
            self._make_train_op()


    # This will hold the data at compute time
    def _make_placeholders(self):
        
        # Data inputs
        self.lines  = tf.placeholder(tf.int32, [None, self.max_line_len], name="lines")
        #self.lines  = tf.placeholder(tf.float32, [None, self.max_line_len, self.syl_vocab_size], name="lines")
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

        fw_rnn_cell = Dropout(LSTM(self.embed_size / 2), self.embed_keep_prob)
        bw_rnn_cell = Dropout(LSTM(self.embed_size / 2), self.embed_keep_prob)

        _, ((_, out_fw), (_, out_bw)) = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_cell,
                bw_rnn_cell,
                inputs,
                sequence_length = input_lengths,
                dtype = tf.float32)

        output = tf.concat((out_fw, out_bw), -1)
        return tf.reshape(output, [-1, self.max_line_len, self.embed_size])

    def _hidden_cell(self):
        return Dropout(LSTM(self.hidden_layer_size), self.hidden_keep_prob)

    def _make_hidden(self, features):

        if self.hidden_is_recurrent:
            fw_rnn_cell = Multi([self._hidden_cell() for _ in range(self.hidden_layers)])
            bw_rnn_cell = Multi([self._hidden_cell() for _ in range(self.hidden_layers)])

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
                shape = [input_size, self.label_vocab_size],
                dtype = tf.float32,
                initializer = tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(
                "b",
                shape = [self.label_vocab_size],
                dtype = tf.float32,
                initializer = tf.zeros_initializer())

        output = tf.nn.dropout(tf.matmul(hidden_output, W) + b, self.output_keep_prob)
        
        return tf.reshape(output, [-1, self.max_line_len, self.label_vocab_size])

    def _make_train_op(self):

        # CRF decoding
        log_likelihood, self.transition_params = \
            tf.contrib.crf.crf_log_likelihood(self.output, self.labels, self.line_lengths)
        loss = tf.reduce_mean(-log_likelihood)

        # Adam optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss)

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
