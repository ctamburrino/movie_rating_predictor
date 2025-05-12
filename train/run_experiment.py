import sys
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.parse import parse_json, combine_text_fields
from vectorizer_factory import get_vectorizer
from model_factory import get_model

file_path = "Movies_and_TV.json"

vectorizer_name = "tfidf" # "tfidf", "tokenizer", or "berts"
model_name = "mlp" # "mlp", "cnn", "lstm", or "bert"

# Dictionary to store timing information
timings = {}

print("Parsing data...")
start_time = time.time()
data = parse_json(file_path, limit=100000)
texts, labels = combine_text_fields(data)
timings['parsing'] = time.time() - start_time

print("Vectorizing text...")
start_time = time.time()
vectorizer = get_vectorizer(vectorizer_name)
X = vectorizer.fit_transform(texts)
timings['vectorizing'] = time.time() - start_time

print("Encoding labels...")
start_time = time.time()
y = np.array(labels).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)
timings['encoding'] = time.time() - start_time

print("Splitting data...")
start_time = time.time()
if model_name == "bert":
    # BERT-specific data splitting
    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.2, random_state=42, stratify=y
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
        np.arange(len(y_train)), y_train, test_size=0.2, random_state=42, stratify=y_train
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
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Second split: separate validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
timings['splitting'] = time.time() - start_time

input_dim = X_train['input_ids'].shape[1] if model_name == "bert" else X.shape[1]

print(f"Training model: {model_name}...")
start_time = time.time()
model = get_model(model_name, input_dim, num_classes=y.shape[1])

# Train the model
model.train(X_train, y_train, X_val, y_val)
timings['training'] = time.time() - start_time

print("Evaluating model...")
start_time = time.time()
if model_name == "bert":
    loss, accuracy = model.evaluate(X_test_raw)
else:
    loss, accuracy = model.evaluate(X_test, y_test)
timings['evaluation'] = time.time() - start_time

print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Get predictions for detailed metrics
if model_name == "bert":
    y_pred = model.model.predict(X_test_raw)
else:
    y_pred = model.model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nDetailed Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

print("\nTiming Information:")
print(f"Data Parsing: {timings['parsing']:.2f} seconds")
print(f"Text Vectorization: {timings['vectorizing']:.2f} seconds")
print(f"Label Encoding: {timings['encoding']:.2f} seconds")
print(f"Data Splitting: {timings['splitting']:.2f} seconds")
print(f"Model Training: {timings['training']:.2f} seconds")
print(f"Model Evaluation: {timings['evaluation']:.2f} seconds")
print(f"Total Time: {sum(timings.values()):.2f} seconds")

