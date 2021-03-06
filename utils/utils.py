def get_class_label_from_path(file_path: str) -> int:
    """Obtain class label for training data
    Returns: int
        class label based on file path
    """
    return int(file_path.split('/')[-2])
