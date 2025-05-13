"""
Implements a BERT tokenizer wrapper.

Authors:
- Cory Tamburrino
- David Kujawinski
- Dinh Troung

Date Last Modified: 5/12/2025
"""

from transformers import BertTokenizer
import numpy as np

class BertTokenizerWrapper:
    """
    A wrapper class for BertTokenizer to provide a consistent interface
    for vectorizing text, regardless of the vectorizer used.
    """
    def __init__(self, model_name='bert-base-uncased', max_len=128):
        """
        Initializes the BertTokenizerWrapper.

        Parameters:
        - model_name (str): The name of the BERT model to use.
        - max_len (int): The maximum length of the input text.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def fit_transform(self, texts):
        """
        Fits the tokenizer to the texts and returns the transformed texts.

        Parameters:
        - texts (list): A list of texts to transform.

        Returns:
        - A dictionary containing the input IDs and attention masks.
        """
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='np'
            )
            input_ids.append(encoded['input_ids'][0])
            attention_masks.append(encoded['attention_mask'][0])

        return {
            'input_ids': np.stack(input_ids),
            'attention_mask': np.stack(attention_masks)
        }

    def transform(self, texts):
        """
        Transforms the texts and returns the transformed texts.

        Parameters:
        - texts (list): A list of texts to transform.

        Returns:
        - A dictionary containing the input IDs and attention masks.
        """
        return self.fit_transform(texts)
