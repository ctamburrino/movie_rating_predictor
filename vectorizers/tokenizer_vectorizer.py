from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TokenizerVectorizerWrapper:
    def __init__(self, num_words=10000, max_length=100, oov_token="<OOV>"):
        self.num_words = num_words
        self.max_length = max_length
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    def fit_transform(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post" )
        return padded
    
    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post" )
        return padded
        