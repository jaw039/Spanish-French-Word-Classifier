
from collections import Counter
import numpy as np

def extract_features(word):
    features = Counter()
    word = word.lower()
    for i in range(len(word) - 1):
        bigram = word[i:i+2]
        features[bigram] += 1
    return features

class NaiveBayesClassifier:
    def __init__(self):
        self.spanish_counts = Counter()
        self.french_counts = Counter()
        self.spanish_total = 0
        self.french_total = 0
        self.vocab = set()

    def train(self, words, labels):
        for word, label in zip(words, labels):
            feats = extract_features(word)
            if label == "spanish":
                self.spanish_counts.update(feats)
                self.spanish_total += sum(feats.values())
            else:
                self.french_counts.update(feats)
                self.french_total += sum(feats.values())
            self.vocab.update(feats.keys())

    def predict(self, word):
        feats = extract_features(word)
        vocab_size = len(self.vocab)
        spanish_prob = np.log(0.5)
        french_prob = np.log(0.5)
        for bigram, count in feats.items():
            sp_count = self.spanish_counts.get(bigram, 0) + 1
            spanish_prob += count * np.log(sp_count / (self.spanish_total + vocab_size))
            fr_count = self.french_counts.get(bigram, 0) + 1
            french_prob += count * np.log(fr_count / (self.french_total + vocab_size))
        return "spanish" if spanish_prob > french_prob else "french"

def classify(train_words, train_labels, test_words):
    classifier = NaiveBayesClassifier()
    classifier.train(train_words, train_labels)
    predictions = [classifier.predict(word) for word in test_words]
    return predictions
