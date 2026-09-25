from app.utils.vocabulary import Vocabulary


captions = [
    "a dog is running",
    "a dog is playing",
    "a cat is running"
]


vocab = Vocabulary(min_freq=1)

vocab.build(captions)


print("Vocabulary size:", len(vocab))

print("Word to index:")
print(vocab.word_to_idx)

caption = "a dog is running"

tokens = vocab.numericalize(caption)

print("Caption:", caption)
print("Tokens:", tokens)