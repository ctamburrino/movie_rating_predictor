def get_model(model_name, input_dim, **kwargs):
    """
    Factory function to get a model based on the model name.

    Parameters:
    - model_name (str): The name of the model to get.
    - input_dim (int): The dimension of the input data.
    - **kwargs: Additional keyword arguments for the model.

    Returns:
    - model: The model object.
    """
    if model_name == "mlp":
        from models.mlp import MLPClassifier
        return MLPClassifier(input_dim)
    elif model_name == "cnn":
        from models.cnn import CNNClassifier
        return CNNClassifier(
            vocab_size=kwargs.get("vocab_size", 10000),
            embedding_dim=kwargs.get("embedding_dim", 64),
            max_length=input_dim,
            num_classes=kwargs.get("num_classes", 5)
        )
    elif model_name == "lstm":
        from models.lstm import LSTMClassifier
        return LSTMClassifier()
    elif model_name == "bert":
        from models.bert import BERTLSTMClassifier
        return BERTLSTMClassifier(num_classes=kwargs.get("num_classes", 5))
    else:
        raise ValueError(f"Model {model_name} not found")
