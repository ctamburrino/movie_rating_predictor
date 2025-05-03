# Function to parse overall, reviewText, and summaryText from json file
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import pickle
import os

def parse_json(file_path, limit=None):
    """
    Parse reviews from a JSON file using an index file for efficient random sampling.
    
    Args:
        file_path (str): Path to the JSON file containing reviews.
        limit (int, optional): Maximum number of reviews to return. If None, returns all reviews.
        
    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - text (str): The review text
            - summary (str): The review summary
            - overall (int): The review rating
            
    Raises:
        FileNotFoundError: If the index file is not found.
    """
    reviews = []
    index_file = "Movies_and_TV.index"
    
    # Load the index file
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file {index_file} not found. Please run create_index.py first.")
    
    with open(index_file, 'rb') as f:
        positions = pickle.load(f)
    
    # Randomly select line numbers
    if limit:
        selected_indices = random.sample(range(len(positions)), min(limit, len(positions)))
    else:
        selected_indices = range(len(positions))
        random.shuffle(selected_indices)
    
    # Read the selected lines
    with open(file_path, 'r') as f:
        for idx in selected_indices:
            f.seek(positions[idx])
            line = f.readline()
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
    """
    Combine the summary and text fields from reviews into a single text field.
    
    Args:
        reviews (list): List of review dictionaries containing 'text' and 'summary' fields.
        
    Returns:
        tuple: A tuple containing:
            - texts (list): List of combined text strings
            - labels (list): List of review ratings
    """
    texts = [(review["summary"] + " " + review["text"].strip()) for review in reviews]
    labels = [review["overall"] for review in reviews]
    return texts, labels


# Vectorize overall
def vectorize_overall(overall_ratings):
    """
    Convert overall ratings into one-hot encoded vectors.
    
    Args:
        overall_ratings (list): List of integer ratings.
        
    Returns:
        tuple: A tuple containing:
            - y_encoded (numpy.ndarray): One-hot encoded ratings
            - encoder (OneHotEncoder): Fitted OneHotEncoder instance
    """
    y = np.array(overall_ratings).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False, categories="auto")
    y_encoded = encoder.fit_transform(y)

    return y_encoded, encoder