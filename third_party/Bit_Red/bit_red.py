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

class BitDepthReduce(tf.keras.layers.Layer):
    """Bit Depth Reduction 레이어"""
    def __init__(self, x_min=0.0, x_max=1.0, **kwargs):
        super(BitDepthReduce, self).__init__(**kwargs)
        self.x_min = x_min
        self.x_max = x_max
        self.step_num = 4
        self.alpha = 200.0
        
        # 양자화 단계 미리 계산
        steps = self.x_min + np.arange(1, self.step_num, dtype=np.float32) / (self.step_num / (self.x_max - self.x_min))
        self.steps = tf.constant(steps.reshape([1, 1, 1, 1, self.step_num-1]), dtype=tf.float32)

    @tf.custom_gradient
    def call(self, inputs):
        # 입력을 float32로 캐스팅
        input_dtype = inputs.dtype
        inputs_32 = tf.cast(inputs, tf.float32)
        
        expanded_inputs = tf.expand_dims(inputs_32, 4)
        quantized = self.x_min + tf.reduce_sum(
            tf.sigmoid(self.alpha * (expanded_inputs - self.steps)), 
            axis=4
        )
        quantized = quantized / ((self.step_num-1) / (self.x_max - self.x_min))

        def grad(upstream):
            # gradient를 원래 입력 dtype으로 변환
            return tf.cast(upstream, input_dtype)

        # 출력도 원래 입력 dtype으로 변환
        return tf.cast(quantized, input_dtype), grad

    def get_config(self):
        config = super(BitDepthReduce, self).get_config()
        config.update({
            'x_min': self.x_min,
            'x_max': self.x_max
        })
        return config

def create_defended_model(ckpt_path='ens3_adv_inception_v3.ckpt-1'):
    """방어 모델을 생성합니다."""
    suppress_tensorflow_warnings()
    
    # 기본 모델 로드
    base_model = load_ensemble_model(ckpt_path)
    
    # 새로운 모델 구성
    input_layer = tf.keras.Input(shape=(299, 299, 3))
    bdr_layer = BitDepthReduce()
    defended_input = bdr_layer(input_layer)
    output = base_model(defended_input)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # 컴파일하여 워닝 제거
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    return model

def verify_bit_depth_reduction(model, images):
    """모델의 Bit Depth Reduction 적용을 검증합니다."""
    # 수동으로 Bit Depth Reduction 적용
    step_num = 4
    alpha = 200.0
    x_min, x_max = 0.0, 1.0
    
    steps = x_min + np.arange(1, step_num, dtype=np.float32) / (step_num / (x_max - x_min))
    steps = steps.reshape([1, 1, 1, 1, step_num-1])
    tf_steps = tf.constant(steps, dtype=tf.float32)
    
    expanded_inputs = tf.expand_dims(images, 4)
    manual_quantized = x_min + tf.reduce_sum(
        tf.sigmoid(alpha * (expanded_inputs - tf_steps)), 
        axis=4
    )
    manual_quantized = manual_quantized / ((step_num-1) / (x_max - x_min))
    
    # 모델 예측과 수동 양자화 후 예측 비교
    model_output = model(images, training=False)
    manual_output = model.layers[-1](manual_quantized, training=False)
    
    output_diff = tf.reduce_max(tf.abs(model_output - manual_output))
    print(f"\nMaximum difference between model and manual quantization: {output_diff:.10f}")
    
    unique_values = np.unique(images.numpy())
    print(f"Number of unique values in quantized output: {len(unique_values)}")
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
    defended_model.save('defended_ensemble_model', 
                       options=save_options,
                       include_optimizer=False)
    
    # 모델 로드 (워닝 없이)
    loaded_model = tf.keras.models.load_model(
        'defended_ensemble_model', 
        custom_objects={'BitDepthReduce': BitDepthReduce},
        compile=False  # 컴파일 워닝 제거
    )
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # 테스트
    test_images = tf.random.uniform((10, 299, 299, 3), 0, 1)
    verify_bit_depth_reduction(loaded_model, test_images)
