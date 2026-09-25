from collections import Counter


class Vocabulary:

    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        self.word_to_idx = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3
        }

        self.idx_to_word = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>"
        }

    def build(self, captions):

        counter = Counter()

        for caption in captions:
            words = caption.lower().split()

            for word in words:
                counter[word] += 1

        for word, frequency in counter.items():

            if frequency >= self.min_freq:

                index = len(self.word_to_idx)

                self.word_to_idx[word] = index
                self.idx_to_word[index] = word

    def numericalize(self, caption):

        words = caption.lower().split()

        tokens = [self.word_to_idx["<START>"]]

        for word in words:

            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx["<UNK>"])

        tokens.append(self.word_to_idx["<END>"])

        return tokens

    def __len__(self):
        return len(self.word_to_idx)