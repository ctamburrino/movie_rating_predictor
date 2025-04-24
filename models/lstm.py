from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMClassifier: 
    def __init__(self, vocab_size=10000, max_seq_len=50, embedding_dim=64):
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
        self.model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val))
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)