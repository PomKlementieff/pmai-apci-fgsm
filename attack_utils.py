import os
import numpy as np
import tensorflow as tf
from models.inceptionv4 import create_model as create_inceptionv4_model
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.convnext import ConvNeXtBase
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.nasnet import NASNetMobile
#from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image

# Constants
MODEL_INPUT_SIZES = {
    'Con-B': (224, 224),
    'Den-121': (224, 224),
    'Den-169': (224, 224),
    'Den-201': (224, 224),
    'Eff-v2_b0': (224, 224),
    'Inc-v3': (299, 299),
    'Inc-v3_ens3': (299, 299),
    'Inc-v3_ens4': (299, 299),
    'Inc-v4': (299, 299),
    'IncRes-v2': (299, 299),
    'IncRes-v2_ens': (299, 299),
    'Mob-v3_L': (224, 224),
    'Mob-v3_S': (224, 224),
    'Nas-M': (224, 224),
    'Res-50': (224, 224),
    'Res-101': (224, 224),
    'Res-152': (224, 224),
    'Xce': (299, 299),
}

# Model loading functions
def load_model(model_name):
    if model_name == 'Con-B':
        return ConvNeXtBase(weights='imagenet', classes=1000)
    elif model_name == 'Den-121':
        return DenseNet121(weights='imagenet', classes=1000)
    elif model_name == 'Den-169':
        return DenseNet169(weights='imagenet', classes=1000)
    elif model_name == 'Den-201':
        return DenseNet201(weights='imagenet', classes=1000)
    elif model_name == 'Eff-v2_b0':
        return EfficientNetV2B0(weights='imagenet', classes=1000)
    elif model_name == 'Inc-v3':
        return InceptionV3(weights='imagenet', classes=1000)
    elif model_name == 'Inc-v4':
        return create_inceptionv4_model(weights='imagenet', num_classes=1000, include_top=True)
    elif model_name == 'IncRes-v2':
        return InceptionResNetV2(weights='imagenet', classes=1000)
    elif model_name in ['Inc-v3_ens3', 'Inc-v3_ens4', 'IncRes-v2_ens']:
        return load_ensemble_model(model_name)
    elif model_name == 'Mob-v3_L':
        return MobileNetV3Large(weights='imagenet', classes=1000)
    elif model_name == 'Mob-v3_S':
        return MobileNetV3Small(weights='imagenet', classes=1000)
    elif model_name == 'Nas-M':
        return NASNetMobile(weights='imagenet', classes=1000)
    elif model_name == 'Res-50':
        return ResNet50V2(weights='imagenet', classes=1000)
    elif model_name == 'Res-101':
        return ResNet101V2(weights='imagenet', classes=1000)
    elif model_name == 'Res-152':
        return ResNet152V2(weights='imagenet', classes=1000)
    elif model_name == 'Xce':
        return Xception(weights='imagenet', classes=1000)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def load_ensemble_model(model_name):
    if model_name == 'Inc-v3_ens3':
        ckpt_path = 'models/ens3_adv_inception_v3.ckpt-1'
        base_model = InceptionV3(weights=None, classes=1000)
    elif model_name == 'Inc-v3_ens4':
        ckpt_path = 'models/ens4_adv_inception_v3.ckpt-1'
        base_model = InceptionV3(weights=None, classes=1000)
    elif model_name == 'IncRes-v2_ens':
        ckpt_path = 'models/ens_adv_inception_resnet_v2.ckpt-1'
        base_model = InceptionResNetV2(weights=None, classes=1000)
    else:
        raise ValueError(f"Unknown ensemble model: {model_name}")

    reader = tf.train.load_checkpoint(ckpt_path)
    shape_from_key = reader.get_variable_to_shape_map()
    
    for var in base_model.trainable_variables:
        if var.name.split(':')[0] in shape_from_key:
            var.assign(reader.get_tensor(var.name.split(':')[0]))
    
    return base_model

# Utility functions
def get_input_size(model_name):
    if model_name in MODEL_INPUT_SIZES:
        return MODEL_INPUT_SIZES[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def resize_batch(batch, target_size):
    return tf.image.resize(batch, target_size)

# Image processing functions
def load_and_preprocess_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x

# Dataset loading function
def load_dataset(data_dir, batch_size=32, image_size=(299, 299)):
    np.random.seed(42)  # Set fixed seed for reproducibility
    
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(data_dir)
        for file in files
        if file.endswith('.JPEG')
    ]
    
    image_paths.sort()
    image_paths = image_paths[:1000]  # Select first 1000 images
    
    def data_generator():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [load_and_preprocess_image(path, target_size=image_size) for path in batch_paths]
            yield np.vstack(batch_images)
    
    return tf.data.Dataset.from_generator(
        data_generator,
        output_types=tf.float32,
        output_shapes=(None, image_size[0], image_size[1], 3)
    )
