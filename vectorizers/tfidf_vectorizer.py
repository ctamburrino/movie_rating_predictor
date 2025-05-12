from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfVectorizerWrapper:
    """
    A wrapper class for TfidfVectorizer to provide a consistent interface
    for vectorizing text, regardless of the vectorizer used.
    """
    def __init__(self, max_features=10000):
        """
        Initializes the TF-IDF vectorizer.

        Parameters:
        - max_features (int): The maximum number of features (vocabulary size).
                              Keeps only the top `max_features` most frequent words.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True,  # Apply sublinear scaling to term frequencies
            analyzer='word',
            token_pattern=r'(?u)\b\w\w+\b'  # Match words with 2 or more characters
        )

    def fit_transform(self, texts):
        """
        Fits the vectorizer to the text and transforms it into TF-IDF vectors.

        Parameters:
        - texts (List[str]): A list of input strings.

        Returns:
        - scipy.sparse.csr_matrix: The TF-IDF matrix of shape (n_samples, max_features).
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Transforms texts into TF-IDF vectors using the fitted vocabulary.

        Parameters:
        - texts (List[str]): A list of input strings.

        Returns:
        - scipy.sparse.csr_matrix: The transformed TF-IDF matrix.
        """
        return self.vectorizer.transform(texts)
