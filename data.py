###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os, helper, json
from collections import Counter


class Dictionary(object):
    """Dictionary class that stores all words of train/dev corpus."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad_token = '<p>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1

    def build_dict(self, instances, max_words):
        word_count = Counter()
        for instance in instances:
            word_count.update(instance.sentence1)
            word_count.update(instance.sentence2)

        most_common = word_count.most_common(max_words) if max_words > 0 else word_count.most_common()
        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Instance(object):
    """Instance that represent a sample of train/dev/test corpus."""

    def __init__(self):
        self.sentence1 = []
        self.sentence2 = []
        self.label = -1

    def add_sentence(self, sentence, tokenize, sentence_no = -1):
        words = ['<s>'] + helper.tokenize(sentence, tokenize) + ['</s>']
        if sentence_no == 1:
            self.sentence1 = words
        elif sentence_no == 1:
            self.sentence2 = words
        elif sentence_no == -1:
            self.sentence1 = words
            self.sentence2 = words


    def add_label(self, label):
        self.label = label

    def print_(self):
        print (' sen 1: ', self.sentence1)
        print (' sen 2: ', self.sentence2)
        print (' label: ', self.label)


class Corpus(object):
    """Corpus class which contains all information about train/dev/test corpus."""

    def __init__(self, _tokenize):
        self.tokenize = _tokenize
        self.data = []

    def parse(self, in_file, task_name, max_example=None):
        """Parses the content of a file."""
        if 'IMDB' in task_name:
            for line in in_file:
                instance = Instance()
                instance.add_sentence(line['text'], self.tokenize, -1)
                instance.add_label(int(line['y']))
                self.data.append(instance)
                if len(self.data) == max_example:
                    break
        elif 'tweet' in task_name:
            with open(in_file, 'r') as f:
                for single_line in f:
                    line = json.loads(single_line)
                    instance = Instance()
                    instance.add_sentence(line['title'], self.tokenize, -1)
                    l = int(line['sentiment'])
                    if l==-1:l=0
                    instance.add_label(l)
                    self.data.append(instance)
                    if len(self.data) == max_example:
                        break
        else:
            assert os.path.exists(in_file)
            with open(in_file, 'r') as f:
                for line in f:
                    tokens = line.strip().split('\t')
                    instance = Instance()
                    if task_name == 'quora':
                        instance.add_label(int(tokens[2]))
                    elif task_name == 'snli' or task_name == 'multinli':
                        if tokens[4] == 'entailment':
                            instance.add_label(0)
                        elif tokens[4] == 'neutral':
                            instance.add_label(1)
                        elif tokens[4] == 'contradiction':
                            instance.add_label(2)
                    self.data.append(instance)
                    if len(self.data) == max_example:
                        break
