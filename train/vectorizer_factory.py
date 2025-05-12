def get_vectorizer(vectorizer_name, **kwargs):
    """
    Factory function to get a vectorizer based on the vectorizer name.

    Parameters:
        vectorizer_name (str): The name of the vectorizer to get.
        **kwargs: Additional keyword arguments to pass to the vectorizer.

    Returns:
        A vectorizer object.
    """
    if vectorizer_name == "tfidf":
        from vectorizers.tfidf_vectorizer import TfidfVectorizerWrapper
        return TfidfVectorizerWrapper(**kwargs)
    elif vectorizer_name == "tokenizer":
        from vectorizers.tokenizer_vectorizer import TokenizerVectorizerWrapper
        return TokenizerVectorizerWrapper(**kwargs)
    elif vectorizer_name == "berts":
        from vectorizers.berts_vectorizer import BertTokenizerWrapper
        return BertTokenizerWrapper(**kwargs)
    else:
        raise ValueError(f"Vectorizer {vectorizer_name} not found")
