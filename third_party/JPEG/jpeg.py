import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import os
import logging

# TensorFlow 워닝 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL만 표시
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def suppress_tensorflow_warnings():
    """TensorFlow 관련 워닝 메시지 억제"""
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)

def load_ensemble_model(ckpt_path):
    """InceptionV3 ensemble 모델을 로드합니다."""
    base_model = InceptionV3(weights=None, classes=1000)
    
    # checkpoint에서 가중치 로드
    reader = tf.train.load_checkpoint(ckpt_path)
    shape_from_key = reader.get_variable_to_shape_map()
    
    for var in base_model.trainable_variables:
        if var.name.split(':')[0] in shape_from_key:
            var.assign(reader.get_tensor(var.name.split(':')[0]))
    
    # Compile 워닝 제거를 위한 컴파일
    base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    return base_model

class JPEGCompression(tf.keras.layers.Layer):
    """JPEG Compression 레이어"""
    def __init__(self, x_min=0.0, x_max=1.0, quality=95, **kwargs):
        super(JPEGCompression, self).__init__(**kwargs)
        self.x_min = x_min
        self.x_max = x_max
        self.quality = quality

    @tf.function
    def compress_single_image(self, img):
        # Convert to uint8 [0, 255] range
        img_uint8 = tf.cast(img, tf.uint8)
        # Encode and decode with JPEG
        encoded = tf.image.encode_jpeg(img_uint8, quality=self.quality)
        decoded = tf.image.decode_jpeg(encoded)
        return decoded

    def call(self, inputs):
        # Scale input images to [0, 255] range
        imgs_scaled = tf.cast((inputs - self.x_min) / ((self.x_max - self.x_min) / 255.0), tf.uint8)
        
        # Process each image in the batch
        imgs_jpeg = tf.map_fn(
            self.compress_single_image,
            imgs_scaled,
            fn_output_signature=tf.uint8
        )
        
        # Ensure output shape matches input
        imgs_jpeg.set_shape(inputs.shape)
        
        # Scale back to original range
        imgs_final = tf.cast(imgs_jpeg, inputs.dtype) / (255.0 / (self.x_max - self.x_min)) + self.x_min
        
        return imgs_final

    def get_config(self):
        config = super(JPEGCompression, self).get_config()
        config.update({
            'x_min': self.x_min,
            'x_max': self.x_max,
            'quality': self.quality
        })
        return config

def create_defended_model(ckpt_path='ens3_adv_inception_v3.ckpt-1'):
    """방어 모델을 생성합니다."""
    suppress_tensorflow_warnings()
    
    # 기본 모델 로드
    base_model = load_ensemble_model(ckpt_path)
    
    # 새로운 모델 구성
    input_layer = tf.keras.Input(shape=(299, 299, 3))
    jpeg_layer = JPEGCompression()
    defended_input = jpeg_layer(input_layer)
    output = base_model(defended_input)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # 컴파일하여 워닝 제거
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    return model

def verify_jpeg_compression(model, images):
    """모델의 JPEG Compression 적용을 검증합니다."""
    # 수동으로 JPEG Compression 적용
    jpeg_layer = model.layers[1]  # JPEG compression layer
    manual_compressed = jpeg_layer(images)
    
    # 모델 예측과 수동 압축 후 예측 비교
    model_output = model(images, training=False)
    manual_output = model.layers[-1](manual_compressed, training=False)
    
    output_diff = tf.reduce_max(tf.abs(model_output - manual_output))
    print(f"\nMaximum difference between model and manual compression: {output_diff:.10f}")
    
    # 압축된 이미지 분석
    compressed_images = manual_compressed.numpy()
    unique_values = np.unique(compressed_images)
    print(f"Number of unique values in compressed output: {len(unique_values)}")
    print(f"Min value: {np.min(unique_values):.6f}")
    print(f"Max value: {np.max(unique_values):.6f}")

# 모델 저장 시 사용할 옵션
save_options = tf.saved_model.SaveOptions(
    experimental_custom_gradients=True,
    save_debug_info=False
)

# 사용 예시
if __name__ == "__main__":
    suppress_tensorflow_warnings()
    
    # 모델 생성
    defended_model = create_defended_model()
    
    # 모델 저장 (워닝 없이)
    defended_model.save('defended_jpeg_model', 
                       options=save_options,
                       include_optimizer=False)
    
    # 모델 로드 (워닝 없이)
    loaded_model = tf.keras.models.load_model(
        'defended_jpeg_model', 
        custom_objects={'JPEGCompression': JPEGCompression},
        compile=False  # 컴파일 워닝 제거
    )
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # 테스트
    test_images = tf.random.uniform((10, 299, 299, 3), 0, 1)
    verify_jpeg_compression(loaded_model, test_images)
