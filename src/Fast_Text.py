from gensim.models.fasttext import FastText, load_facebook_model
from gensim.test.utils import get_tmpfile, datapath
from Preprocessing import Preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_SEQUENCE_LENGTH = 100

class Language_Model:

    def __init__(self):
        self.preprocessing = Preprocessing()
        fname = datapath("/usr/src/app/fasttext_classification_big.model")
        self.model = FastText.load(fname)

    def encode_text(self, text):

        text, vocab = self.preprocessing.process_text(text, lower=True)
        print("The vocabulary contains {} unique tokens".format(len(vocab)))

        # encode text
        for x in range(len(text)):
            for i in range(len(text[x])):
                if text[x][i] in self.model.wv.vocab:
                    arr = (self.model.wv.vocab[text[x][i]].index + 1)
                    text[x][i] = arr
                else:
                    text[x][i] = 0

        data = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH,
                          padding="pre", truncating="post")

        return data

    def now2(text):
        preprocessing = Preprocessing()
        text, vocab = preprocessing.process_text(text, lower=True)
        print("The vocabulary contains {} unique tokens".format(len(vocab)))
        fname = datapath("fasttext123347.model")
        model = FastText.load(fname)
        #
        # # YOU HAVE TO FIX HEREEEEEEEEEEEEEEEEEEEEE
        for x in range(len(text)):
            for i in range(len(text[x])):
                if text[x][i] in model.wv.vocab:
                    arr = (model.wv.vocab[text[x][i]].index + 1)
                    #print(arr.shape)
                    text[x][i] = arr
                else:
                    text[x][i] = 0

        data = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH,
                          padding="pre", truncating="post")

        return data
