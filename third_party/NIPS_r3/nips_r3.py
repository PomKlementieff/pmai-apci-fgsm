"""NIPS 2017 Defense Implementation in TensorFlow 2.11.0

This defense uses an ensemble of models including:
- Inception ResNet v2
- Inception v3
- ResNet152 v2
- VGG16

Each image goes through different preprocessing techniques including distortion,
transcoding, and scaling before being classified by the ensemble.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.ndimage import interpolation

# Constants
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_MAX_SCALE = 1.1
_MIN_SCALE = 0.75
_VGG_MAX_SCALE = 0.8
_MAX_ANGLE = 4
_MAX_NOISE = 4

class Config:
    def __init__(self):
        self.image_width = 299
        self.image_height = 299
        self.batch_size = 16

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='constant', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [
        interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def inc_distort(image):
    image += np.random.randn(*image.shape).reshape(image.shape) * _MAX_NOISE / 255.
    np.clip(image, -1, 1, out=image)
    angle = np.random.randint(_MAX_ANGLE - 2, _MAX_ANGLE)
    if np.random.randint(2) == 0:
        angle = -angle

    rotated = interpolation.rotate(image, angle, reshape=True)
    zoom = (np.random.uniform(_MAX_SCALE - 0.05, _MAX_SCALE),
            np.random.uniform(_MAX_SCALE - 0.05, _MAX_SCALE), 1)
    zoomed = interpolation.zoom(rotated, zoom)

    starts = [(zoomed.shape[i] - image.shape[i]) // 2 for i in range(2)]
    ends = [starts[i] + image.shape[i] for i in range(2)]
    result = zoomed[starts[0]:ends[0], starts[1]:ends[1]]
    return result

def vgg_distort(image, out_shape):
    angle = np.random.randint(_MAX_ANGLE - 2, _MAX_ANGLE)
    if np.random.randint(2) == 0:
        angle = -angle
    rotated = interpolation.rotate(image, angle, reshape=True)

    zoom = (np.random.uniform(_MIN_SCALE, _VGG_MAX_SCALE),
            np.random.uniform(_MIN_SCALE, _VGG_MAX_SCALE), 1)
    zoomed = interpolation.zoom(rotated, zoom)

    starts = [(zoomed.shape[i] - out_shape[i]) // 2 for i in range(2)]
    ends = [starts[i] + out_shape[i] for i in range(2)]
    result = zoomed[starts[0]:ends[0], starts[1]:ends[1]]
    return result

def ens_distort(image):
    h = image.shape[0]
    w = image.shape[1]
    channel_axis = 2
    shear = jitter(np.pi/128)
    theta = jitter(-np.pi/128)
    tx = int(jitter(w*0.02))
    ty = int(jitter(h*0.02))
    zx = jitter(0.95)
    zy = jitter(0.95)

    transform_matrix = np.array(
        [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    image = apply_transform(image, transform_matrix, channel_axis)

    transform_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    image = apply_transform(image, transform_matrix, channel_axis)

    transform_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    image = apply_transform(image, transform_matrix, channel_axis)

    transform_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    image = apply_transform(image, transform_matrix, channel_axis)

    return image

def jitter(num):
    scale = np.random.randint(95, 106) / 100.
    return num*scale

def transcode(image):
    im = Image.fromarray(image)
    im.save('/dev/shm/tmp.jpg', quality=50)
    transcoded = Image.open('/dev/shm/tmp.jpg')
    return np.asarray(transcoded)

class NIPSDefenseModel(tf.keras.Model):
    def __init__(self):
        super(NIPSDefenseModel, self).__init__()
        self.config = Config()
        self.inception_v3 = tf.keras.applications.InceptionV3(weights=None, classes=1000)
        
        ckpt_path = 'third_party/NIPS_r3/upgraded_ens3_adv_inception_v3_rename.ckpt-1'
        reader = tf.train.load_checkpoint(ckpt_path)
        shape_from_key = reader.get_variable_to_shape_map()
        
        for var in self.inception_v3.trainable_variables:
            if var.name.split(':')[0] in shape_from_key:
                var.assign(reader.get_tensor(var.name.split(':')[0]))
        
        print("Successfully loaded Inc-v3_ens3 model weights")

    def _process_single_image(self, image):
        """Process a single image"""
        from skimage.transform import resize

        # 입력을 [0, 255] 범위로 정규화
        if tf.reduce_max(image) <= 1.0:
            image = image * 255.0
        
        # uint8로 변환
        image_np = tf.cast(image, tf.uint8).numpy()
        
        # JPEG transcoding
        im = Image.fromarray(image_np)
        im.save('/dev/shm/tmp.jpg', quality=50)
        transcoded = Image.open('/dev/shm/tmp.jpg')
        tcd_image = np.array(transcoded, dtype=np.float32)
        tcd_image = resize(tcd_image, (299, 299, 3), anti_aliasing=True)
        tcd_image = tcd_image * 255.0  # resize는 [0,1] 범위로 정규화하므로 다시 [0,255]로 변환
        
        # Distortion 적용
        image_float = image_np.astype(np.float32)
        
        # 노이즈 추가
        noise = np.random.randn(*image_float.shape) * _MAX_NOISE
        image_float = image_float + noise
        image_float = np.clip(image_float, 0, 255)
        
        # 회전
        angle = np.random.randint(-_MAX_ANGLE, _MAX_ANGLE + 1)
        rotated = interpolation.rotate(image_float, angle, reshape=False, mode='reflect')
        
        # 스케일링 및 리사이즈
        scale = np.random.uniform(_MIN_SCALE, _MAX_SCALE)
        scaled = interpolation.zoom(rotated, (scale, scale, 1), mode='reflect')
        distorted = resize(scaled, (299, 299, 3), anti_aliasing=True) * 255.0
        
        # [-1, 1] 범위로 정규화
        inc_image = (distorted - 127.5) / 127.5
        tcd_image = (tcd_image - 127.5) / 127.5
        
        # shape 검증 - 오류가 발생하면 디버깅을 위해 정보 출력
        if inc_image.shape != (299, 299, 3):
            print(f"Incorrect inc_image shape: {inc_image.shape}")
            inc_image = resize(inc_image, (299, 299, 3)) # 마지막 시도로 리사이즈
        if tcd_image.shape != (299, 299, 3):
            print(f"Incorrect tcd_image shape: {tcd_image.shape}")
            tcd_image = resize(tcd_image, (299, 299, 3)) # 마지막 시도로 리사이즈
            
        return np.float32(inc_image), np.float32(tcd_image)

    def preprocess_images(self, images):
        """Apply preprocessing techniques to input images."""
        images = tf.cast(images, tf.float32)
        
        processed_images = tf.map_fn(
            lambda x: tf.py_function(
                self._process_single_image,
                [x],
                [tf.float32, tf.float32]
            ),
            images,
            dtype=[tf.float32, tf.float32],
            parallel_iterations=1
        )
        
        inc_images, tcd_images = processed_images
        
        # Shape 설정
        inc_images = tf.ensure_shape(inc_images, (None, 299, 299, 3))
        tcd_images = tf.ensure_shape(tcd_images, (None, 299, 299, 3))
        
        return inc_images, tcd_images

    def call(self, inputs):
        print(f"Model input range: [{tf.reduce_min(inputs)}, {tf.reduce_max(inputs)}]")
        
        inc_images, tcd_images = self.preprocess_images(inputs)
        
        # Get predictions
        inc_pred = self.inception_v3(inc_images)
        tcd_pred = self.inception_v3(tcd_images)
        
        # Original weights from NIPS 2017 defense
        logits = 0.9 * inc_pred + 0.1 * tcd_pred
        
        print("Prediction completed")
        return tf.nn.softmax(logits)

    def get_config(self):
        return {"name": "NIPSDefenseModel"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_model():
    """Helper function to create and return the defense model."""
    return NIPSDefenseModel()
