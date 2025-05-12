from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from keras.optimizers import Adam

class CNNClassifier:
    """
    A class for a convolutional neural network (CNN) classifier.
    """
    def __init__(self, vocab_size=10000, embedding_dim=64, max_length=100, num_classes=5):
        """
        Initializes the CNNClassifier.

        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - embedding_dim (int): The dimension of the embedding vectors.
        """
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            Conv1D(filters=128, kernel_size=5, activation="relu"),
            GlobalMaxPooling1D(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the CNNClassifier on the training data.

        Parameters:
        - X_train (numpy.ndarray): The training input data.
        - y_train (numpy.ndarray): The training output data.
        - X_val (numpy.ndarray): The validation input data.
        - y_val (numpy.ndarray): The validation output data.
        """
        self.model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val))
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the CNNClassifier on the test data.

        Parameters:
        - X_test (numpy.ndarray): The test input data.
        - y_test (numpy.ndarray): The test output data.
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X_test):
        """
        Predicts the class labels for the test data.

        Parameters:
        - X_test (numpy.ndarray): The test input data.
        """
        return self.model.predict(X_test)
        