from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class MLPClassifier:
    """
    A class for a multi-layer perceptron (MLP) classifier.
    """
    def __init__(self, input_dim):
        """
        Initializes the MLPClassifier.

        Parameters:
        - input_dim (int): The dimension of the input data.
        """
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(5, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the MLPClassifier on the training data.

        Parameters:
        - X_train (numpy.ndarray): The training input data.
        - y_train (numpy.ndarray): The training output data.
        - X_val (numpy.ndarray): The validation input data.
        - y_val (numpy.ndarray): The validation output data.
        """
        self.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

    def evaluate(self, X_test, y_test):
        """
        Evaluates the MLPClassifier on the test data.

        Parameters:
        - X_test (numpy.ndarray): The test input data.
        - y_test (numpy.ndarray): The test output data.
        """
        return self.model.evaluate(X_test, y_test)
        
        
