import datetime
import logging
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from app.model.knn import Hdf5Model, CsvModel
import constants
from fetch.resource import S3Object
import settings
from train.input_generator import GenerateSample
from utils.utils import get_class_label_from_path


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/train/triplet_loss.log')
logger.addHandler(handler)


now = datetime.datetime.now()
strf_time = now.strftime("%Y%m%d_%H%M%S")

EMBEDDING_DIM = 50
EPOCH = 6
# steps_per_epoch is total number of batch samples genereated by the generator between an epoch and next epoch
# unique samples data divided by batch size in general
STEPS_PER_EPOCH = 4
VALIDATION_STEPS = 2

PATH_TRAIN = os.path.join(settings.base_dir, 'tmp', 'train')
PATH_MODEL_CHECKPINT = os.path.join(settings.base_dir, 'tmp', 'checkpoint', f'triplet_chkpt_{strf_time}.hdf5')
PATH_MODEL_TMP = os.path.join(settings.base_dir, 'tmp', 'hdf5', f'embedding_{strf_time}.hdf5')
PATH_TFMODEL_TMP = os.path.join(settings.base_dir, 'tmp', 'tflite', f'embedding_{strf_time}.tflite')
PATH_MODEL_DIST = os.path.join('hdf5', f'embedding_{strf_time}.hdf5')
PATH_TFMODEL_DIST = os.path.join('tflite', f'embedding_{strf_time}.tflite')


class TripletLoss(object):
    """TripletLoss model class"""
    def triplet_loss(self, inputs, dist='sqeuclidean', margin='maxplus'):
        """triplet loss function
        Use 3 samples (anchor, positive, negative) data.
        Caluculate the distance between an anchor sample and a positive sample and also between an anchor samples and negative samples.
        Set the loss function such as the same class are much closer and different one are much father

        Parameters
        ----------
        inputs: list
            3 output list converted through model
        dist: str
            How to measure metric space.
        margin: str
            Output layer settings
        """
        anchor, positive, negative = inputs
        positive_distance = K.square(anchor - positive)
        negative_distance = K.square(anchor - negative)

        if dist == 'euclidean':
            positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
            negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
        elif dist == 'sqeuclidean':
            positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
            negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
        loss = positive_distance - negative_distance
        if margin == 'maxplus':
            loss = K.maximum(0.0, 1 + loss)
        elif margin == 'softplus':
            loss = K.log(1 + K.exp(loss))
        return K.mean(loss)

    def get_model(self):
        """create triplet model
        Use pretrained model, MobileNetV2

        Returns: tupple
            embedding model and model with triplet loss function
        """
        base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max')
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Dropout(0.6)(x)
        x = Dense(EMBEDDING_DIM)(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        embedding_model = Model(base_model.input, x, name='embedding')

        input_shape = (int(constants.IMAGE_SIZE), int(constants.IMAGE_SIZE), 3)
        anchor_input = Input(input_shape, name='anchor_input')
        positive_input = Input(input_shape, name='positive_input')
        negative_input = Input(input_shape, name='negative_input')
        anchor_embedding = embedding_model(anchor_input)
        positive_embedding = embedding_model(positive_input)
        negative_embedding = embedding_model(negative_input)

        inputs = [anchor_input, positive_input, negative_input]
        outputs = [anchor_embedding, positive_embedding, negative_embedding]

        triplet_model = Model(inputs, outputs)
        triplet_model.add_loss(K.mean(self.triplet_loss(outputs)))
        return embedding_model, triplet_model

    def convert_to_tflite(self, embedding_model):
        """Convert from hdf5 to tflite
        Create a tflite file to make inferences from mobile edge devices
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
        tflite_model = converter.convert()

        logger.info({'action': 'convert_to_tflite', 'status': 'start'})
        with open(PATH_TFMODEL_TMP, mode='wb') as f:
            f.write(tflite_model)
        logger.info({'action': 'convert_to_tflite', 'status': 'success'})

    def train(self, file_paths: list):
        """Train model
        Train triplet loss model and save in hdf5, tflite as a result
        """
        train_paths, test_paths = train_test_split(file_paths, train_size=0.7, random_state=1337)
        file_class_mapping_train = {train_path: get_class_label_from_path(train_path) for train_path in train_paths}
        file_class_mapping_test = {test_path: get_class_label_from_path(test_path) for test_path in test_paths}
        train_samples = GenerateSample(file_class_mapping_train)
        test_samples = GenerateSample(file_class_mapping_test)

        checkpoint = ModelCheckpoint(PATH_MODEL_CHECKPINT, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
        callbacks_list = [checkpoint, early]

        embedding_model, triplet_model = self.get_model()
        # show layers
        for i, layer in enumerate(embedding_model.layers):
            print(i, layer.name, layer.trainable)

        # TODO: adjust parameters to be flozen gradually
        for layer in embedding_model.layers[72:]:
            layer.trainable = True
        for layer in embedding_model.layers[:72]:
            layer.trainable = False
            if "bn" in layer.name:
                layer.trainable = True
        logger.info({'action': 'train', 'network summary': embedding_model.summary()})
        
        triplet_model.compile(loss=None, optimizer=Adam(lr=0.0001))
        logger.info({'action': 'train', 'status': 'start training'})
        history = triplet_model.fit_generator(train_samples.generate(), validation_data=test_samples.generate(), epochs=EPOCH, verbose=1, workers=1,
                                              steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS,
                                              use_multiprocessing=False, callbacks=callbacks_list)
        logger.info({'action': 'train', 'train_loss': history.history['loss']})
        logger.info({'action': 'train', 'val_loss': history.history['val_loss']})
        logger.info({'action': 'train', 'epoch': EPOCH, 'steps_per_epoch': STEPS_PER_EPOCH, 'validation_steps': VALIDATION_STEPS})
        logger.info({'action': 'train', 'history': history.history})

        embedding_model.save(PATH_MODEL_TMP)
        s3_object = S3Object(PATH_MODEL_DIST, aws_access_key_id=settings.aws_access_key_id, aws_secret_access_key=settings.aws_secret_access_key)
        is_saved_model = s3_object.upload_file(PATH_MODEL_TMP)

        if is_saved_model:
            self.convert_to_tflite(embedding_model)
            s3_object.file_path = PATH_TFMODEL_DIST
            is_saved_tflite = s3_object.upload_file(PATH_TFMODEL_TMP)
        
        if is_saved_model and is_saved_tflite:
            logger.info({'action': 'train', 'status': 'success to train model and save it'})
        else:
            logger.info({'action': 'train', 'status': 'failed to save model'})
        logger.info({'action': 'train', 'status': 'end training'})

    def predict(self, file_paths: list, spot_id: int, is_exhibit: bool):
        file_class_mapping_train = {file_path: get_class_label_from_path(file_path) for file_path in file_paths}
        
        embedding_model, triplet_model = self.get_model()
        hdf5_model = Hdf5Model()
        embedding_model = load_model(hdf5_model.file_path)

        samples = GenerateSample(file_class_mapping_train)
        
        logger.info({'action': 'predict', 'status': 'start', 'spot_id': spot_id, 'message': 'start to write csv file'})
        
        result_predicts = []
        result_classes = []
        for classes, images in tqdm(samples.generate_image(batch_size=8)):
            predicts = embedding_model.predict(images)
            predicts = predicts.tolist()
            result_predicts += predicts
            result_classes += classes

        csv_model = CsvModel(spot_id=spot_id)
        if not is_exhibit:
            csv_model.save_all_data(classes=result_classes, images=result_predicts)
        else:
            csv_model.add_new_data(classes=result_classes, images=result_predicts)
