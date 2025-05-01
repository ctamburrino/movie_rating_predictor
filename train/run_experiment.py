import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.parse import parse_json, combine_text_fields
from vectorizer_factory import get_vectorizer
from model_factory import get_model

file_path = "Movies_and_TV.json"

vectorizer_name = "tfidf" # "tfidf", "tokenizer", or "berts"
model_name = "lstm" # "mlp", "cnn", "lstm", or "bert"

print("Parsing data...")
data = parse_json(file_path, limit=10000)
texts, labels = combine_text_fields(data)

print("Vectorizing text...")
vectorizer = get_vectorizer(vectorizer_name)
X = vectorizer.fit_transform(texts)

print("Encoding labels...")
y = np.array(labels).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

print("Splitting data...")
if model_name == "bert":
    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.2, random_state=42
    )
    X_train_raw = {
        'input_ids': X['input_ids'][idx_train],
        'attention_mask': X['attention_mask'][idx_train]
    }
    X_test_raw = {
        'input_ids': X['input_ids'][idx_test],
        'attention_mask': X['attention_mask'][idx_test]
    }

    idx_train2, idx_val, y_train, y_val = train_test_split(
        np.arange(len(y_train)), y_train, test_size=0.2, random_state=42
    )
    X_train = {
        'input_ids': X_train_raw['input_ids'][idx_train2],
        'attention_mask': X_train_raw['attention_mask'][idx_train2]
    }
    X_val = {
        'input_ids': X_train_raw['input_ids'][idx_val],
        'attention_mask': X_train_raw['attention_mask'][idx_val]
    }
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val = None  # not used for non-BERT models

input_dim = X_train['input_ids'].shape[1] if model_name == "bert" else X.shape[1]

print(f"Training model: {model_name}...")
model = get_model(model_name, input_dim, num_classes=y.shape[1])

if model_name == "bert":
    model.train(X_train, y_train, X_val, y_val)
else:
    model.train(X_train, y_train, X_test, y_test)

print("Evaluating model...")
if model_name == "bert":
    loss, accuracy = model.evaluate(X_test_raw)
else:
    loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

