from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from keras.optimizers import Adam

class CNNClassifier:
    def __init__(self, vocab_size=10000, embedding_dim=64, max_length=100, num_classes=5):
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
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
        self.model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val))
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
        