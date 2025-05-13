"""
Implements a BERT LSTM classifier.

Authors:
- Cory Tamburrino
- David Kujawinski
- Dinh Troung

Date Last Modified: 5/12/2025
"""

import os
import warnings
import logging as py_logging
from transformers import TFBertModel, logging as hf_logging
import tensorflow as tf

# Suppress Hugging Face and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)
py_logging.getLogger('tensorflow').setLevel(py_logging.ERROR)
hf_logging.set_verbosity_error()


class BERTLSTMClassifier(tf.keras.Model):
    """
    A class for a BERT LSTM classifier.
    """
    def __init__(self, num_classes, dropout_rate=0.3):
        """
        Initializes the BERT LSTM classifier.

        Parameters:
        - num_classes (int): The number of classes.
        - dropout_rate (float): The dropout rate.
        """
        super(BERTLSTMClassifier, self).__init__()
        self.bert = TFBertModel.from_pretrained(
            'bert-base-uncased',
            add_pooling_layer=False  # disables unused pooler to avoid gradient warnings
        )
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        """
        Calls the BERT LSTM classifier.

        Parameters:
        - inputs (dict): A dictionary containing the input data.

        Returns:
        - A tensor of the predicted class labels.
        """
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        x = self.lstm(sequence_output)
        x = self.dropout(x)
        return self.classifier(x)

    def train(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=3):
        """
        Trains the BERT LSTM classifier.

        Parameters:
        - X_train (dict): Training inputs (input_ids + attention_mask).
        - y_train (np.ndarray): Training labels.
        - X_val (dict): Validation inputs.
        - y_val (np.ndarray): Validation labels.
        - batch_size (int): Batch size.
        - epochs (int): Number of epochs.
        """
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs
        )

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the BERT LSTM classifier.

        Returns:
        - A tuple of (loss, accuracy).
        """
        return super().evaluate(*args, **kwargs)
