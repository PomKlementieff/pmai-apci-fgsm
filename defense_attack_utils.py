import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import logging
from third_party.HGD.denoise_defense_models import create_denoise_inception
from third_party.RS.rs import RandomSmoothingModel
from third_party.Bit_Red.bit_red import BitDepthReduce, suppress_tensorflow_warnings
from third_party.FD.feature_distillation import FeatureDistillation
from third_party.JPEG.jpeg import JPEGCompression
from third_party.NIPS_r3.nips_r3 import (NIPSDefenseModel, create_model)
from third_party.ComDefend.compression_imagenet import ComDefend

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
suppress_tensorflow_warnings()

MODEL_PATHS = {
    'jpeg_defense': {
        'path': 'third_party/JPEG/ens3_adv_inception_v3.ckpt-1',
        'module': 'JPEG.jpeg',
        'class': JPEGCompression
    },
    'feature_distill': {
        'path': 'third_party/FD/ens3_adv_inception_v3.ckpt-1',
        'module': 'FD.feature_distillation',
        'class': FeatureDistillation
    },
    'bit_red': {
        'path': 'third_party/Bit_Red/ens3_adv_inception_v3.ckpt-1',
        'module': 'Bit_Red.bit_red',
        'class': BitDepthReduce
    }
}

MODEL_INPUT_SIZES = {
    'Inc-v3': (299, 299),
    'hgd': (299, 299),
    'rs': (224, 224),
    'r_p': (299, 299),
    'bit_red': (299, 299),
    'feature_distill': (299, 299),
    'jpeg_defense': (299, 299),
    'nips_r3': (299, 299),
    'comdefend': (299, 299),
    'purify': (299, 299)
}

class PurifyWithClassifier(tf.keras.Model):
    def __init__(self):
        super(PurifyWithClassifier, self).__init__()
        try:
            self.purifier = tf.keras.models.load_model('third_party/NRP/tf_models/nrp_model')
            self.classifier = tf.keras.applications.InceptionV3(weights='imagenet')
        except Exception as e:
            raise ValueError(f"Failed to load models: {str(e)}")

    def purify_batch(self, images, dynamic=True):
        images = tf.cast(images, tf.float32)
        if tf.reduce_max(images) > 1.0:
            images = images / 255.0
        
        images = tf.image.resize(images, (256, 256))
        
        if dynamic:
            eps = 16/255
            noise = tf.random.normal(tf.shape(images), mean=0, stddev=0.05)
            images_noisy = images + noise
            images_noisy = tf.clip_by_value(images_noisy, images - eps, images + eps)
            images_noisy = tf.clip_by_value(images_noisy, 0.0, 1.0)
        else:
            images_noisy = images
        
        purified = self.purifier(images_noisy, training=False)
        purified = tf.clip_by_value(purified, 0.0, 1.0)
        purified = tf.image.resize(purified, (299, 299))
        
        return purified

    @tf.function
    def call(self, inputs, training=False):
        purified = self.purify_batch(inputs, dynamic=True)
        purified = purified * 2.0 - 1.0
        return self.classifier(purified)

class ComDefendWithClassifier(tf.keras.Model):
    def __init__(self):
        super(ComDefendWithClassifier, self).__init__()
        try:
            self.defend = ComDefend()
            self.defend.model = tf.keras.models.load_model('third_party/ComDefend/tf2_weights/comdefend_model')
            self.classifier = tf.keras.applications.InceptionV3(weights='imagenet')
        except Exception as e:
            raise ValueError(f"Failed to initialize ComDefend: {str(e)}")
    
    @tf.function
    def call(self, inputs, training=False):
        defended = self.defend(inputs, training=False)
        defended = defended * 2.0 - 1.0
        return self.classifier(defended)

def verify_model(model, model_name, sample_input=None):
    if sample_input is None:
        input_size = get_input_size(model_name)
        if model_name == 'feature_distill':
            sample_input = tf.random.uniform((1, input_size[0], input_size[1], 3), -1, 1)
        elif model_name == 'nips_r3':
            sample_input = tf.random.uniform((1, input_size[0], input_size[1], 3), 0, 255)  
        else:
            sample_input = tf.random.uniform((1, input_size[0], input_size[1], 3), 0, 1)
   
    try:
        if model_name == 'purify':
            purified = model.purify_batch(sample_input)
            _ = model(sample_input, training=False)
        elif model_name == 'jpeg_defense':
            from third_party.JPEG.jpeg import verify_jpeg_compression
            verify_jpeg_compression(model, sample_input)
        elif model_name == 'feature_distill':
            from third_party.FD.feature_distillation import verify_feature_distillation
            verify_feature_distillation(model, sample_input)
        elif model_name == 'bit_red':
            from third_party.Bit_Red.bit_red import verify_bit_depth_reduction
            verify_bit_depth_reduction(model, sample_input)
        elif model_name == 'nips_r3':
            inc_images, tcd_images = model.preprocess_images(sample_input)
            _ = model.inception_v3(inc_images)
            _ = model.inception_v3(tcd_images)
            _ = model(sample_input, training=False)
        else:
            _ = model(sample_input, training=False)
        return True
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        return False

def load_model(model_name, sigma=0.25, verify=True):
    save_options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)
    
    if model_name in MODEL_PATHS:
        try:
            model_info = MODEL_PATHS[model_name]
            module = __import__(f'third_party.{model_info["module"]}', 
                              fromlist=['create_defended_model'])
            model = module.create_defended_model(model_info['path'])
            
            temp_path = f'temp_{model_name}'
            model.save(temp_path, options=save_options, include_optimizer=False)
            model = tf.keras.models.load_model(
                temp_path, 
                custom_objects={model_info['class'].__name__: model_info['class']}
            )
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            
            if os.path.exists(temp_path):
                import shutil
                shutil.rmtree(temp_path)
                
            if verify:
                verify_model(model, model_name)
            return model
                
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            raise ValueError(f"Failed to load {model_name} model: {str(e)}")
    
    if model_name == 'purify':
        try:
            model = PurifyWithClassifier()
            if verify:
                verify_model(model, model_name)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load Purify model: {str(e)}")
    
    if model_name == 'comdefend':
        try:
            model = ComDefendWithClassifier()
            if verify:
                verify_model(model, model_name)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load ComDefend model: {str(e)}")

    if model_name == 'nips_r3':
        try:
            model = create_model()
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                policy = tf.keras.mixed_precision.Policy('float32')
                tf.keras.mixed_precision.set_global_policy(policy)
            if verify:
                verify_model(model, model_name)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load NIPS Defense model: {str(e)}")

    if model_name == 'r_p':
        from third_party.R_P.inception_resnet_v2_tf2 import InceptionResNetV2
        model = InceptionResNetV2(num_classes=1001)
        _ = model(tf.random.normal((1, 299, 299, 3)))
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore('third_party/R_P/ens_adv_inception_resnet_v2.ckpt')
        status.expect_partial()
        return model
    
    if model_name == 'rs':
        try:
            model = tf.keras.applications.ResNet50(include_top=True, weights=None, 
                                                 input_shape=(224, 224, 3), classes=1000)
            _ = model(tf.random.normal((1, 224, 224, 3)))
            model.load_weights('third_party/RS/tf_model_weights.h5', by_name=True)
            return RandomSmoothingModel(model, sigma=sigma)
        except Exception as e:
            raise ValueError(f"Failed to load defense model weights: {str(e)}")
    
    if model_name == 'hgd':
        try:
            model = create_denoise_inception()
            checkpoint = tf.train.Checkpoint(model=model)
            status = checkpoint.restore('third_party/HGD/denoise_incepv3_012')
            status.expect_partial()
            return model
        except Exception as e:
            raise ValueError(f"Failed to load HGD model: {str(e)}")
    
    standard_models = {
        'Inc-v3': lambda: tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
    }
    
    if model_name in standard_models:
        return standard_models[model_name]()
    
    raise ValueError(f"Unknown model: {model_name}")

def get_input_size(model_name):
    if model_name in MODEL_INPUT_SIZES:
        return MODEL_INPUT_SIZES[model_name]
    base_name = model_name.split('_')[0] + '_' + model_name.split('_')[1]
    if base_name in MODEL_INPUT_SIZES:
        return MODEL_INPUT_SIZES[base_name]
    raise ValueError(f"Unknown model: {model_name}")

def resize_batch(batch, target_size):
    return tf.image.resize(batch, target_size)

def load_and_preprocess_image(img_path, target_size=(299, 299), model_name=None):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    if model_name == 'comdefend':
        x = tf.clip_by_value(x / 255.0, 0.0, 1.0)
    elif model_name == 'nips_r3':
        x = tf.cast(x, tf.float32)
    elif model_name == 'feature_distill':
        x = (x / 127.5) - 1.0
    elif model_name == 'jpeg_defense':
        x = tf.cast(x, tf.float32) / 255.0
    else:
        x = tf.keras.applications.inception_v3.preprocess_input(x)
    
    return x

def load_dataset(data_dir, batch_size=32, image_size=(299, 299), model_name=None):
    np.random.seed(42)
    batch_size = 16 if model_name == 'nips_r3' else batch_size
    
    image_paths = []
    for root, _, files in os.walk(data_dir):
        image_paths.extend([os.path.join(root, f) for f in files if f.endswith('.JPEG')])
    
    image_paths.sort()
    image_paths = image_paths[:1000]
    
    def data_generator():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [load_and_preprocess_image(path, target_size=image_size, model_name=model_name) 
                          for path in batch_paths]
            batch = np.vstack(batch_images)
            
            if model_name == 'feature_distill':
                batch = tf.clip_by_value(batch, -1.0, 1.0)
            elif model_name == 'jpeg_defense':
                batch = tf.clip_by_value(batch, 0.0, 1.0)
            
            yield batch
    
    return tf.data.Dataset.from_generator(
        data_generator,
        output_types=tf.float32,
        output_shapes=(None, image_size[0], image_size[1], 3)
    )
