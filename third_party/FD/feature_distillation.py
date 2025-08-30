"""Feature Distillation Defense Implementation"""
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import os
import logging

# TensorFlow 워닝 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def suppress_tensorflow_warnings():
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)

def load_ensemble_model(ckpt_path):
    """InceptionV3 ensemble 모델을 로드합니다."""
    base_model = InceptionV3(weights=None, classes=1000)
    reader = tf.train.load_checkpoint(ckpt_path)
    shape_from_key = reader.get_variable_to_shape_map()
    
    for var in base_model.trainable_variables:
        if var.name.split(':')[0] in shape_from_key:
            var.assign(reader.get_tensor(var.name.split(':')[0]))
    
    base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return base_model

class FeatureDistillation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeatureDistillation, self).__init__(**kwargs)
        self.block_size = 8
        
        # Quantization tables
        self.q_table = tf.constant(
            np.ones((self.block_size, self.block_size)) * 30, 
            dtype=tf.float32
        )
        self.q_table = tf.tensor_scatter_nd_update(
            self.q_table,
            indices=[[i, j] for i in range(4) for j in range(4)],
            updates=tf.ones(16) * 25
        )
        
        # DCT 행렬 초기화
        self.dct_matrix = self._build_dct_matrix()

    def _build_dct_matrix(self):
        """DCT 변환 행렬 생성"""
        n = np.arange(self.block_size)
        k = n.reshape((self.block_size, 1))
        
        # DCT 행렬 계산
        dct_matrix = np.cos(np.pi * (2 * n + 1) * k / (2 * self.block_size))
        dct_matrix *= np.sqrt(2 / self.block_size)
        dct_matrix[0] = dct_matrix[0] / np.sqrt(2)
        
        return tf.constant(dct_matrix, dtype=tf.float32)

    @tf.function
    def dct2_tf(self, block):
        """2D DCT using pre-computed matrix"""
        temp = tf.matmul(self.dct_matrix, block)
        return tf.matmul(temp, self.dct_matrix, transpose_b=True)

    @tf.function
    def idct2_tf(self, block):
        """2D IDCT using pre-computed matrix"""
        temp = tf.matmul(self.dct_matrix, block, transpose_a=True)
        return tf.matmul(temp, self.dct_matrix)

    @tf.function
    def process_block(self, block):
        """단일 8x8 블록 처리"""
        # DCT 변환 및 양자화
        dct_block = self.dct2_tf(block)
        quantized = tf.round(dct_block / self.q_table)
        
        # 역양자화 및 IDCT 변환
        unquantized = quantized * self.q_table
        return self.idct2_tf(unquantized)

    @tf.function
    def process_channel(self, channel, height, width):
        """단일 채널 처리"""
        # 8x8 블록으로 reshape
        blocks = tf.reshape(
            channel, 
            [height // self.block_size, width // self.block_size, 
             self.block_size, self.block_size]
        )
        
        # 블록 처리
        processed = tf.map_fn(
            lambda x: tf.map_fn(self.process_block, x),
            blocks
        )
        
        # 원래 크기로 복원
        return tf.reshape(processed, [height, width])

    @tf.function
    def process_single_image(self, image):
        """단일 이미지 처리"""
        # [-1, 1] -> [0, 255] 변환
        image = (image + 1) / 2.0 * 255
        
        # 304x304로 리사이즈
        image = tf.image.resize(image, [304, 304])
        
        # 필요한 패딩 계산
        target_height = tf.cast(
            tf.math.ceil(304 / self.block_size) * self.block_size,
            tf.int32
        )
        target_width = target_height  # 정사각형 유지
        
        # 패딩 추가
        padded = tf.image.resize_with_crop_or_pad(
            image, target_height, target_width
        )
        
        # 채널별 처리
        channels = tf.unstack(padded, axis=-1)
        processed_channels = [
            self.process_channel(channel, target_height, target_width)
            for channel in channels
        ]
        
        # 채널 결합 및 크기 조정
        processed = tf.stack(processed_channels, axis=-1)
        processed = tf.image.resize_with_crop_or_pad(processed, 304, 304)
        processed = tf.image.resize(processed, [299, 299])
        
        # [0, 255] -> [-1, 1] 변환
        processed = tf.clip_by_value(processed / 255.0, 0.0, 1.0)
        return processed * 2.0 - 1.0

    def call(self, inputs):
        return tf.map_fn(
            self.process_single_image,
            inputs,
            fn_output_signature=tf.float32
        )

    def get_config(self):
        return super(FeatureDistillation, self).get_config()

def create_defended_model(ckpt_path='upgraded_ens3_adv_inception_v3_rename.ckpt-1'):
    """방어 모델을 생성합니다."""
    suppress_tensorflow_warnings()
    base_model = load_ensemble_model(ckpt_path)
    
    input_layer = tf.keras.Input(shape=(299, 299, 3))
    fd_layer = FeatureDistillation()
    defended_input = fd_layer(input_layer)
    output = base_model(defended_input)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    return model

def verify_feature_distillation(model, images):
    """모델의 Feature Distillation 적용을 검증합니다."""
    fd_layer = model.layers[1]
    manual_processed = fd_layer(images)
    
    model_output = model(images, training=False)
    manual_output = model.layers[-1](manual_processed, training=False)
    
    output_diff = tf.reduce_max(tf.abs(model_output - manual_output))
    print(f"\nMaximum difference between model and manual processing: {output_diff:.10f}")
    
    processed_images = manual_processed.numpy()
    unique_values = np.unique(processed_images)
    print(f"Number of unique values in processed output: {len(unique_values)}")
    print(f"Min value: {np.min(unique_values):.6f}")
    print(f"Max value: {np.max(unique_values):.6f}")

# 모델 저장 옵션
save_options = tf.saved_model.SaveOptions(
    experimental_custom_gradients=True,
    save_debug_info=False
)

if __name__ == "__main__":
    suppress_tensorflow_warnings()
    
    # 모델 생성
    defended_model = create_defended_model()
    
    # 모델 저장
    defended_model.save(
        'defended_fd_model', 
        options=save_options,
        include_optimizer=False
    )
    
    # 모델 로드
    loaded_model = tf.keras.models.load_model(
        'defended_fd_model', 
        custom_objects={'FeatureDistillation': FeatureDistillation},
        compile=False
    )
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # 테스트
    test_images = tf.random.uniform((10, 299, 299, 3), -1, 1)
    verify_feature_distillation(loaded_model, test_images)
