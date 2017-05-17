import numpy as np
import re
from glob import glob

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1] 

def errorRate(actual, predicted):
    totalErrors = 0
    count = 0
    for i in range(len(actual)):
        minErrors = 100
        length = 100
        for actualString in actual[i]:
            errors = levenshtein(actualString, predicted[i])
            if errors < minErrors:
                minErrors = errors
                length = len(actualString)

        totalErrors += minErrors
        count += length

    return totalErrors / count * 100

def line_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        equivalent = False
        for actualString in actual[i]:
            if tuple(actualString) == tuple(predicted[i]):
                equivalent = True

        if equivalent:
            correct += 1

    return correct / len(actual)

def readDict(filename, stress=False):
    dictionary = {}
    text = ""
    phoneList = []
    with open(filename) as f:
        for line in f:

            #if not re.match(r'^[A-Z]', line[0]):
            if line[:3] == ";;;":
                continue

            word, phones = line.split("  ")
            word = re.sub(r'\(\d\)', r'', word)
            if stress:
                phones = re.sub(r'([012])', r' \1', phones).split()
            else:
                phones = re.sub(r'[012]', r'', phones).split()

            if word not in dictionary:
                dictionary[word] = [phones]
            else:
                dictionary[word].append(phones)

            text += word
            for phone in phones:
                phoneList.append(phone)

    return dictionary, phoneList, text

def readLabeledPoems(dirname, filenames):
    X = []
    Y = []
    text = []

    for filename in glob(dirname + "/" + filenames):
        for line in open(filename):
            line = line.split("\t")
            if len(line) < 3:
                continue

            label = re.sub('[() ]', '', line[1])
            label = label.replace("^", "-")

            X.append(line[2])
            Y.append(label.split("|"))

            text.extend(line[2])

    return X, Y, text

def convert2int(stringList, max_line_len, char_indices, backwards=False):
    if backwards:
        return [[char_indices[char] for char in line[::-1]] for line in stringList]
    else:
        return [[char_indices[char] for char in line] for line in stringList]

def trainTestSplit(data, fraction, position):
    start = int(len(data) * fraction) * position
    end = int(len(data) * fraction) * (position + 1)
    train = list(data[:start]) + list(data[end:])
    test = list(data[start:end])
    return train, test


