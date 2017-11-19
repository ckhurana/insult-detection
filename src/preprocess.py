from sklearn.base import BaseEstimator
import numpy as np


class CleanTransformer(BaseEstimator):

    def __init__(self):
        with open('badlist.txt') as file:
            badwords = [line.strip() for line in file.readlines()]
        self.badwords = badwords

    def fit(self):
        return self

    def transform(self, docs):
        n_words = []
        n_chars = []
        caps_ratio = []

        for doc in docs:
            n_words.append(len(doc.split()))
            n_chars.append(len(doc))

            caps_ratio.append(np.sum([c.isupper() for c in doc]) / len(doc))

        return np.array([n_words, n_chars, caps_ratio]).T