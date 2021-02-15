from collections import defaultdict
import logging

import numpy as np
from PIL import Image

from fetch.resource import S3Object
import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/train/input_generator.log')
logger.addHandler(handler)


BATCH_SIZE = 8
IMAGE_SIZE = 224


class GenerateSample(object):
    """GenerateSample class"""
    def __init__(self, file_class_mapping):
        """Instantiate generate sample class
        
        Parameters
        ----------
        file_class_mapping: dict[str: int]
            the mapping of file path as key and classification label as value.
            ex. {"uploads/1/1/xxxx.jpg": 1, "uploads/1/3/yyyy.jpg": 3}
        """
        self.file_class_mapping = file_class_mapping
        # the mapping of class label as key and array of file paths as value
        self.class_to_list_files = defaultdict(list)
        self.list_all_files = list(self.file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in self.file_class_mapping.items():
            self.class_to_list_files[class_].append(file)

        # unique label of classification class
        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes = range(len(self.list_classes))
        # ratio of each classification class
        self.class_number = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = self.class_number / np.sum(self.class_number)

    def get_sample(self):
        """Get 2 positive examples as an anchor and a negative example

        Select a classification class by the ratio of each class and get 3 training examples
        including two examples in a selected class and a exmaple in an unselected class.
        """
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)

        positive_example_1, positive_example_2 = \
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]],\
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]
        positive_class = self.list_classes[class_idx]

        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] == self.file_class_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
            negative_class = self.file_class_mapping[negative_example]
        return positive_example_1, negative_example, positive_example_2, positive_class, negative_class

    def read_and_resize(self, file_path: str):
        """Read the file from file path and resize to squire(224x224)

        Parameter
        ---------
        file_path: str
            the file path of image file
        
        Returns: ndarray
            image data converted to RGB array(shape: 224x224x3)
        """
        s3_object = S3Object(file_path, aws_access_key_id=settings.aws_access_key_id, aws_secret_access_key=settings.aws_secret_access_key)
        io_image = s3_object.get_bytes_image_on_memory()
        pil_image = Image.open(io_image).convert('RGB')
        pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.array(pil_image, dtype="float32")

    def augment(self, image_array):
        """Flip left and right with a 90% possibility to augment sample data
        
        Parameter
        --------
        image_array: ndarray
            image array converted to numpy
        """
        if np.random.uniform(0, 1) > 0.9:
            image_array = np.fliplr(image_array)
        return image_array

    def generate(self):
        """Generator for the inputs of model

        Returns: dict[str: ndarray]
            generate 3 sample data (including an anchor, a positive and a negative sample) as generator
        """
        while True:
            logger.debug({'action': 'generate', 'status': 'start', 'message': 'start to generate batch samples'})
            list_positive_examples_1 = []
            list_negative_examples = []
            list_positive_examples_2 = []

            for i in range(BATCH_SIZE):
                positive_example_1, negative_example, positive_example_2, positive_class, negative_class = self.get_sample()

                positive_example_1_img = self.read_and_resize(positive_example_1)
                negative_example_img = self.read_and_resize(negative_example)
                positive_example_2_img = self.read_and_resize(positive_example_2)

                positive_example_1_img = self.augment(positive_example_1_img)
                negative_example_img = self.augment(negative_example_img)
                positive_example_2_img = self.augment(positive_example_2_img)

                list_positive_examples_1.append(positive_example_1_img)
                list_negative_examples.append(negative_example_img)
                list_positive_examples_2.append(positive_example_2_img)
                logger.debug({'action': 'generate', 'positive_class': positive_class, 'negative_class': negative_class, 'batch': i})

            label = None
            logger.debug({'action': 'generate', 'status': 'end', 'message': 'finish to generate batch sampels'})
            yield ({
                'anchor_input': np.array(list_positive_examples_1),
                'positive_input': np.array(list_positive_examples_2),
                'negative_input': np.array(list_negative_examples)}, label)
    
    def generate_image(self, batch_size=8):
        """Generator for triplet network prediction

        Parameters
        ----------
        batch_size: int
            batch size to input triplet network

        Returns: tuple(list, list)
            generate 2 lists, the list of classification class and the list of image converted to RGB array
        """
        i = 0
        images = []
        classes = []
        for file_path, class_ in self.file_class_mapping.items():
            if i == 0:
                images = []
                classes = []
            i += 1
            image = self.read_and_resize(file_path)
            images.append(image)
            classes.append(class_)
            if i == batch_size:
                i = 0
                images = np.array(images)
                yield classes, images
        # If finish to extract from map and batch data remains, output the rest of the data
        if i != 0:
            images = np.array(images)
            yield classes, images
