from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMClassifier:
    """
    A class for a long short-term memory (LSTM) classifier.
    """
    def __init__(self, vocab_size=10000, max_seq_len=50, embedding_dim=64):
        """
        Initializes the LSTMClassifier.

        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - max_seq_len (int): The maximum sequence length.
        - embedding_dim (int): The dimension of the embedding vectors.
        """
        self.model = Sequential([
            Input(shape=(max_seq_len,)),
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2), 
            Dense(5, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss = "categorical_crossentropy",
            metrics=["accuracy"]
        )
    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the LSTMClassifier on the training data.

        Parameters:
        - X_train (numpy.ndarray): The training input data.
        - y_train (numpy.ndarray): The training output data.
        - X_val (numpy.ndarray): The validation input data.
        - y_val (numpy.ndarray): The validation output data.
        """
        self.model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val))
    def evaluate(self, X_test, y_test):
        """
        Evaluates the LSTMClassifier on the test data.

        Parameters:
        - X_test (numpy.ndarray): The test input data.
        - y_test (numpy.ndarray): The test output data.
        """
        return self.model.evaluate(X_test, y_test)