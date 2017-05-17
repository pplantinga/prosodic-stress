from helpers import readDict, readLabeledPoems, convert2int
import re
import tensorflow as tf
from model_new import Seq2SeqModel
from tensorflow.contrib import rnn
import string
import sys


# Convert a line of text into a list of words, each word is composed of syllables
def convert2words(line):
    line = line.replace('"', "")
    line = line.replace(",", ", ")
    line = re.sub("-\s*-", " ", line)
    line = re.sub("(\w)-(\w)", r'\1 \2', line)
    line = re.sub("[^A-Z '-]", " ", line.upper())

    # separate into words
    words = []
    word = []
    for syllable in reversed(line.split()):

        # If it starts with a dash, its a later syllable
        if syllable[0] == '-':
            word.append(syllable[1:])

        # If not, its the beginning of a word, so append
        else:
            word.append(syllable)
            words.append(list(reversed(word)))
            word = []

    return list(reversed(words))

def isVowel(letter, vowels):
    if vowels:
        return letter[0] in ["A", "E", "I", "O", "U"]
    else:
        return letter[0] not in ["A", "E", "I", "O", "U"]

# Eat all consonants/vowels
def eat(letters, index, vowels = True):
    letterList = []
    while index < len(letters) and isVowel(letters[index], vowels):
        letterList.append(letters[index])
        index += 1
        
    if index < len(letters) and vowels and letters[index][0] == "W":
        letterList.append(letters[index])
        index += 1

    return letterList, index

# Divide the phones of a word into syllables based on the text syllables
def syllablize(phones, syllables):

    if len(syllables) == 1:
        return [phones]

    phone_syllables = []
    phone_syllable = []
    phone_index = 0

    for syllable in syllables:
        letter_index = 0

        # Eat all consonants
        consonants, letter_index = eat(syllable, letter_index, vowels=False)
        con_phones, phone_index = eat(phones, phone_index, vowels=False)

        phone_syllable.extend(con_phones)

        # Eat next vowel
        vowels, letter_index = eat(syllable, letter_index, vowels=True)
        #if phone_index < len(phones) and  isVowel(phones[phone_index], vowels=True):
        #    phone_syllable.append(phones[phone_index])
        #    phone_index += 1
            
        vow_phones, phone_index = eat(phones, phone_index, vowels=True)
        if phone_index < len(phones) and phones[phone_index] in ["0", "1", "2"]:
            vow_phones.append(phones[phone_index])
            phone_index += 1

        phone_syllable.extend(vow_phones)

        # Eat same number of consonants off the end of the syllable and phones
        while letter_index < len(syllable) and phone_index < len(phones) and not isVowel(phones[phone_index], True):
            phone_syllable.append(phones[phone_index])
            phone_index += 1
            if letter_index + 1 < len(syllable) and syllable[letter_index:letter_index+2] in ["TH", "CH", "CK", "SH"]:
                letter_index += 2
            else:
                letter_index += 1

        phone_syllables.append(phone_syllable)
        phone_syllable = []

    # append any remaining phones onto the last syllable
    while phone_index < len(phones):
        phone_syllables[-1].append(phones[phone_index])
        phone_index += 1

    return phone_syllables

########
# Main #
########

dictionary, phoneList, text = readDict("cmudict-0.7b.utf8", False)

if len(sys.argv) > 1 and sys.argv[1] == "hymns":
    phoneFile = "hymnPhones.txt"
    X = []
    Y = []
    with open("hymnLabels.txt") as f:
        for line in f:
            label, words = line.split("\t")
            Y.append(label)
            X.append(words)
else:        
    phoneFile = "linePhones.txt"
    X, Y, _ = readLabeledPoems("poems", "*.txt")

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars, 1))
indices_char = dict((i, c) for i, c in enumerate(chars, 1))

phone_markers = sorted(list(set(phoneList)))
phone_indices = dict((c, i) for i, c in enumerate(phone_markers, 2))
indices_phone = dict((i, c) for i, c in enumerate(phone_markers, 2))

# Load char2phone model
layer_size = 512
layers = 3
dropout = 0.3
keep_prob = tf.placeholder(tf.float32)
encoder_cell = rnn.DropoutWrapper(rnn.LSTMCell(layer_size), output_keep_prob=keep_prob)
decoder_cell = rnn.DropoutWrapper(rnn.LSTMCell(layer_size * 2), output_keep_prob=keep_prob)
if layers > 1:
    encoder_cell = rnn.MultiRNNCell([encoder_cell] * layers)
    decoder_cell = rnn.MultiRNNCell([decoder_cell] * layers)

with tf.Session() as session:
    phoneModel = Seq2SeqModel(encoder_cell=encoder_cell,
            decoder_cell=decoder_cell,
            vocab_size=60,
            embedding_size=16,
            layers=layers,
            keep_prob=keep_prob,
            attention=True,
            bidirectional=True,
            debug=False)

    saver = tf.train.Saver()

    saver.restore(session, tf.train.latest_checkpoint("saved_models_3layer512/"))

    # Convert words to phones
    maxlen = max([len(i) for i in X])
    phone_X = []
    for i in range(len(X)):
        linePhones = []
        for word in convert2words(X[i]):
            wordString = "".join(word)
            if wordString not in dictionary and wordString != "":
                inputs = convert2int([wordString], maxlen, char_indices, backwards=True)
                fd = phoneModel.make_inference_inputs(inputs, keep_prob=1.)
                prediction = session.run(phoneModel.decoder_prediction_inference, fd).T
                dictionary[wordString] = [[indices_phone[i] for i in prediction[0] if i in indices_phone]]

            if wordString in dictionary:
                phones = syllablize(dictionary[wordString][0], word)
                linePhones.extend(phones)

        phone_X.append(linePhones)

with open(phoneFile, "w") as w:
    for i in range(len(phone_X)):
        if isinstance(Y[i], str):
            w.write(Y[i])
        else:
            w.write("|".join(Y[i]))

        phones = ["-".join(j) for j in phone_X[i]]
        for j in range(len(phones)):
            if not phones[j]:
                phones[j] = "NUL"

        w.write("\t" + " ".join(phones))

        w.write("\t" + X[i])

