from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerWrapper:
    """
    A wrapper class for TfidfVectorizer to provide a consistent interface
    for vectorizing text, regardless of the vectorizer used.
    """
    def __init__(self, max_features):
        """
        Initializes the TF-IDF vectorizer.

        Parameters:
        - max_features (int): The maximum number of features (vocabulary size).
                              Keeps only the top `max_features` most frequent words.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english')

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
