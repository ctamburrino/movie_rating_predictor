def get_vectorizer(vectorizer_name, **kwargs):
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
