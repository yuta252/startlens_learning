import os

import settings


IMAGE_SIZE = 224
EMBEDDING_DIM = 50
PATH_CSV_DIR = os.path.join(settings.base_dir, 'tmp', 'csv')
PATH_KNN_DIR = os.path.join(settings.base_dir, 'tmp', 'knn')
