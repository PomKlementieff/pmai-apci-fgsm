import logging
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import get_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

def preprocess_input(x):
    return tf.keras.applications.inception_v3.preprocess_input(x)

def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same', use_bias=False):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                      kernel_regularizer=tf.keras.regularizers.l2(0.00004),
                      kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal'))(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = layers.Activation('relu')(x)
    return x

def inception_stem(input_tensor):
    x = conv2d_bn(input_tensor, 32, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    
    branch_0 = layers.MaxPooling2D(3, strides=(2, 2), padding='valid')(x)
    branch_1 = conv2d_bn(x, 96, 3, strides=(2, 2), padding='valid')
    x = layers.Concatenate(axis=-1)([branch_0, branch_1])
    
    branch_0 = conv2d_bn(x, 64, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, padding='valid')
    branch_1 = conv2d_bn(x, 64, 1)
    branch_1 = conv2d_bn(branch_1, 64, (1, 7))
    branch_1 = conv2d_bn(branch_1, 64, (7, 1))
    branch_1 = conv2d_bn(branch_1, 96, 3, padding='valid')
    x = layers.Concatenate(axis=-1)([branch_0, branch_1])
    
    branch_0 = conv2d_bn(x, 192, 3, strides=(2, 2), padding='valid')
    branch_1 = layers.MaxPooling2D(3, strides=(2, 2), padding='valid')(x)
    x = layers.Concatenate(axis=-1)([branch_0, branch_1])
    
    return x

def inception_a(x):
    branch_0 = conv2d_bn(x, 96, 1)
    
    branch_1 = conv2d_bn(x, 64, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3)
    
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    
    branch_3 = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_3 = conv2d_bn(branch_3, 96, 1)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    return x

def inception_b(x):
    branch_0 = conv2d_bn(x, 384, 1)
    
    branch_1 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(branch_1, 224, (1, 7))
    branch_1 = conv2d_bn(branch_1, 256, (7, 1))
    
    branch_2 = conv2d_bn(x, 192, 1)
    branch_2 = conv2d_bn(branch_2, 192, (7, 1))
    branch_2 = conv2d_bn(branch_2, 224, (1, 7))
    branch_2 = conv2d_bn(branch_2, 224, (7, 1))
    branch_2 = conv2d_bn(branch_2, 256, (1, 7))
    
    branch_3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_3 = conv2d_bn(branch_3, 128, 1)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    return x

def inception_c(x):
    branch_0 = conv2d_bn(x, 256, 1)
    
    branch_1 = conv2d_bn(x, 384, 1)
    branch_10 = conv2d_bn(branch_1, 256, (1, 3))
    branch_11 = conv2d_bn(branch_1, 256, (3, 1))
    branch_1 = layers.Concatenate(axis=-1)([branch_10, branch_11])
    
    branch_2 = conv2d_bn(x, 384, 1)
    branch_2 = conv2d_bn(branch_2, 448, (3, 1))
    branch_2 = conv2d_bn(branch_2, 512, (1, 3))
    branch_20 = conv2d_bn(branch_2, 256, (1, 3))
    branch_21 = conv2d_bn(branch_2, 256, (3, 1))
    branch_2 = layers.Concatenate(axis=-1)([branch_20, branch_21])
    
    branch_3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_3 = conv2d_bn(branch_3, 256, 1)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    return x

def reduction_a(x):
    branch_0 = conv2d_bn(x, 384, 3, strides=(2, 2), padding='valid')
    
    branch_1 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=(2, 2), padding='valid')
    
    branch_2 = layers.MaxPooling2D(3, strides=(2, 2), padding='valid')(x)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    return x

def reduction_b(x):
    branch_0 = conv2d_bn(x, 192, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, strides=(2, 2), padding='valid')
    
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, (1, 7))
    branch_1 = conv2d_bn(branch_1, 320, (7, 1))
    branch_1 = conv2d_bn(branch_1, 320, 3, strides=(2, 2), padding='valid')
    
    branch_2 = layers.MaxPooling2D(3, strides=(2, 2), padding='valid')(x)
    
    x = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    return x

def inception_v4(input_shape=(299, 299, 3), num_classes=1001, dropout_rate=0.2, include_top=True, weights=None):
    inputs = layers.Input(shape=input_shape)
    
    x = inception_stem(inputs)
    
    for _ in range(4):
        x = inception_a(x)
    
    x = reduction_a(x)
    
    for _ in range(7):
        x = inception_b(x)
    
    x = reduction_b(x)
    
    for _ in range(3):
        x = inception_c(x)
    
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = x
    
    model = models.Model(inputs, outputs, name='inception_v4')
    
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('inception-v4_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='9fe79d77f793fe874470d84ca6ba4a3b')
        else:
            weights_path = get_file('inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='9296b46b5971573064d12e4669110969')
        
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        
        if include_top and num_classes != 1001:
            model.layers[-1].set_weights([
                tf.random.normal((1536, num_classes)),
                tf.zeros((num_classes,))
            ])
        
    elif weights is not None:
        model.load_weights(weights)
    
    return model

def create_model(num_classes=1001, dropout_prob=0.2, weights=None, include_top=True):
    return inception_v4(num_classes=num_classes, dropout_rate=dropout_prob, weights=weights, include_top=include_top)

if __name__ == "__main__":
    model = create_model(weights='imagenet', num_classes=1000)
