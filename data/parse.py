# Function to parse overall, reviewText, and summaryText from json file
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np

file_path = "Movies_and_TV.json"

def parse_json(file_path, limit=None):
    reviews = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if limit and i >= limit:
                break
            try:
                review = json.loads(line)
                text = review.get("reviewText", "")
                summary = review.get("summary", "")
                overall = review.get("overall", "")
                if text and overall:
                    reviews.append({
                        "text": text,
                        "summary": summary,
                        "overall": int(overall)
                    })
            except json.JSONDecodeError:    
                continue
    return reviews

def combine_text_fields(reviews):
    texts = [(review["summary"] + " " + review["text"].strip()) for review in reviews]
    labels = [review["overall"] for review in reviews]
    return texts, labels


# Vectorize overall
def vectorize_overall(overall_ratings):
    
    y = np.array(overall_ratings).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False, categories="auto")
    y_encoded = encoder.fit_transform(y)

    return y_encoded, encoder

if __name__ == "__main__":
    file_path = "Movies_and_TV.json"
    data = parse_json(file_path)

    # Combine text fields
    texts = [combine_text_fields(review) for review in data]
    labels = [review["overall"] for review in data]

    # Vectorize text
    X, vectorizer = vectorize_text_tfidf(texts)
    y = labels
    # Vectorize overall
    y, encoder = vectorize_overall(labels)

    print(f"Vectorized {X.shape[0]} samples with {X.shape[1]} features.")