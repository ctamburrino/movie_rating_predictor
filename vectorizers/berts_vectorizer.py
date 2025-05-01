# berts_vectorizer.py
from transformers import BertTokenizer
import numpy as np

class BertTokenizerWrapper:
    def __init__(self, model_name='bert-base-uncased', max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def fit_transform(self, texts):
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
        return self.fit_transform(texts)
