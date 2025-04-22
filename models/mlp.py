from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input,Dense, Dropout
from tensorflow.keras.optimizers import Adam

class MLPClassifier:
    def __init__(self, input_dim):
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128,activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(5, activation="softmax")
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
        
        
