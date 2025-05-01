import tensorflow as tf
from transformers import TFBertModel

class BERTLSTMClassifier(tf.keras.Model):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        x = self.lstm(sequence_output)
        x = self.dropout(x)
        return self.classifier(x)

    def train(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=3):
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        self.fit(X_train, y_train, validation_data=(X_val, y_val),
                 batch_size=batch_size, epochs=epochs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)

