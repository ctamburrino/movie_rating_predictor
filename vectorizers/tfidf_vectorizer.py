from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerWrapper:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
