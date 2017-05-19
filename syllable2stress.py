
from psm import ProsodicStressModel
import tensorflow as tf
from g2p.helpers import trainTestSplit, errorRate, line_accuracy, trainTestSplit
from random import seed, shuffle
from re import sub

def readPoems(filename):
    lines = []
    phones = []
    labels = []
    with open(filename) as f:
        for line in f:
            label, phone, text = line.split("\t")

            labels.append(label.split("|"))
            phones.append([p.split("-") for p in phone.split()])
            lines.append(text)

    return lines, phones, labels

def findFirstStress(phones):
    for phone in phones:
        if phone in ["0", "1", "2"]:
            return phone

    return "NUL"

def splitPhones(phoneStress):
    phones = []
    lex = []

    for line in phoneStress:
        phones.append([[p for p in syl if p not in ["0", "1", "2"]] for syl in line])
        lex.append([findFirstStress(syl) for syl in line])

    return phones, lex

# Split a line into syllables. _if clean is true, remove punctuation and caps
def split_line(line, clean=True):
    
    if clean:
        line = sub("[^a-z ']", ' ', line.lower())

    else:
        line = line.replace('"', '')
        line = line.replace(",", ", ")
        line = sub("(\s*-\s*)+", " -", line)
        line = sub("\s*-$", "", line)

    return line.split()


########
# Main #
########

randoSeed = 131009
seed(randoSeed)

# Read in labeled poems
lines, phones, labels = readPoems("linePhonesWithStress.txt")

# Shuffle data randomly
Z = list(zip(lines, phones, labels))
shuffle(Z)
lines, phones, labels = zip(*Z)

# Split pronunciations into phones and lexical stress
phones, lex = splitPhones(phones)

# Generate graphs and lines from the text
graphs = [split_line(i, clean=False) for i in lines]
lines = [split_line(i, clean=True) for i in lines]

# Training parameters
lr = 0.002
batch_size = 32

avg_SER = 0
avg_acc = 0

model = ProsodicStressModel()#hidden_is_recurrent=False)

# 10-fold CV
folds = 10
for position in range(folds):

    train = {}
    test = {}
    train['phones'], test['phones'] = trainTestSplit(phones, 1/folds, position)
    train['graphs'], test['graphs'] = trainTestSplit(graphs, 1/folds, position)
    train['lines'],  test['lines']  = trainTestSplit(lines, 1/folds, position)
    train['labels'], test['labels'] = trainTestSplit(labels, 1/folds, position)
    train['lex'],    test['lex']    = trainTestSplit(lex, 1/folds, position)
    model.make_inputs(train, test)

    # On first iteration, initialize graph and start session
    if position == 0:
        model.make_graph()
        sess = tf.Session()

    last_acc = 0

    # Train for 50 epochs
    sess.run(tf.global_variables_initializer())
    for epoch in range(50):

        batches = model.batchify(batch_size, lr)

        for fd in batches:
            sess.run(model.train_op, fd)

        predictions = model.predict(sess)

        # Check out error rate and accuracy
        SER = errorRate(model.test['labels'], predictions)
        acc = line_accuracy(model.test['labels'], predictions)

        if (epoch + 1) == 50:
            print("Epoch: " + str(epoch))
            print("SER = " + str(SER))
            print("acc = " + str(acc*100))

        # Reduce learning rate by 30% whenever accuracy decreases
        if acc < last_acc:
            lr *= 0.7

        last_acc = acc

    avg_SER += SER / 10
    avg_acc += acc / 10

print("Average SER: " + str(avg_SER))
print("Average acc: " + str(avg_acc * 100))
