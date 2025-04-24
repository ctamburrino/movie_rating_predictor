import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.parse import parse_json, combine_text_fields
from vectorizer_factory import get_vectorizer
from model_factory import get_model
from vectorizers.tfidf_vectorizer import TfidfVectorizerWrapper
from models.mlp import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

file_path = "Movies_and_TV.json"

# tfidf or tokenizer
vectorizer_name = "tokenizer"
# mlp or cnn
model_name = "lstm"

print("Parsing data...")
data = parse_json(file_path,10000)
texts, labels = combine_text_fields(data)

# tokenizer vectorizer - num_words and max_length are optional parameters
# tfidf vectorizer - max_features is optional parameter
print("Vectorizing text...")
vectorizer = get_vectorizer(vectorizer_name)
X = vectorizer.fit_transform(texts)

print("Encoding labels...")
y = np.array(labels).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training MLP...")
model = get_model(model_name, X.shape[1])
model.train(X_train, y_train, X_test, y_test)

print("Evaluating MLP...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")