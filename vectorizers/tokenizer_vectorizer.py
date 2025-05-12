from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TokenizerVectorizerWrapper:
    """
    A wrapper class for Tokenizer to provide a consistent interface
    for vectorizing text, regardless of the vectorizer used.
    """
    def __init__(self, num_words=10000, max_length=100, oov_token="<OOV>"):
        """
        Initializes the Tokenizer.

        Parameters:
        - num_words (int): The maximum number of features (vocabulary size).
        - max_length (int): The maximum length of the sequence.
        - oov_token (str): The token for out-of-vocabulary words.
        """
        self.num_words = num_words
        self.max_length = max_length
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    def fit_transform(self, texts):
        """
        Fits the Tokenizer on the texts and transforms them into sequences.

        Parameters:
        - texts (List[str]): A list of input strings.

        Returns:
        - numpy.ndarray: Padded sequences of shape (n_samples, max_length)
        """
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post" )
        return padded
    
    def transform(self, texts):
        """
        Transforms the texts into sequences.

        Parameters:
        - texts (List[str]): A list of input strings.

        Returns:
        - numpy.ndarray: Padded sequences of shape (n_samples, max_length)
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post" )
        return padded
        