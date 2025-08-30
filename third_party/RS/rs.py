import tensorflow as tf
import numpy as np
import h5py

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.mean = tf.constant([0.485, 0.456, 0.406])
        self.std = tf.constant([0.229, 0.224, 0.225])

    def call(self, x):
        x = tf.cast(x, tf.float32) / 255.0
        return (x - self.mean) / self.std

class RandomSmoothingModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model, sigma: float = 0.25):
        super().__init__()
        self.normalize = NormalizeLayer()
        self.base_model = base_model
        self.sigma = sigma

    def call(self, x, training=False):
        x = self.normalize(x)
        return self.base_model(x, training=training)

    def random_smoothing_predict(self, x: tf.Tensor, n_samples: int = 100):
        predictions = []
        for _ in range(n_samples):
            noise = tf.random.normal(shape=tf.shape(x), 
                                   mean=0.0, 
                                   stddev=self.sigma)
            noisy_x = tf.clip_by_value(x + noise, 0.0, 1.0)
            pred = self(noisy_x, training=False)
            predictions.append(pred)

        predictions = tf.stack(predictions)
        avg_pred = tf.reduce_mean(predictions, axis=0)
        
        pred_class = tf.argmax(avg_pred, axis=-1)
        confidence = tf.reduce_max(avg_pred, axis=-1)
        
        return int(pred_class), float(confidence)

def create_test_image(size=(224, 224)):
    img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return tf.cast(img, tf.float32)

def inspect_h5_weights(filepath):
    """H5 가중치 파일 구조 확인"""
    with h5py.File(filepath, 'r') as f:
        print("\nWeight file structure:")
        def print_structure(name, obj):
            print(name)
        f.visititems(print_structure)

def build_resnet50_model():
    """ResNet50 모델 구축"""
    base_model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=1000
    )
    
    # 레이어 이름 출력
    print("\nModel layers:")
    for layer in base_model.layers:
        print(f"Layer name: {layer.name}, type: {type(layer).__name__}")
    
    return base_model

def test_defense_model():
    print("방어 모델 테스트 시작...")
    
    try:
        # 1. 가중치 파일 구조 확인
        print("가중치 파일 구조 확인...")
        inspect_h5_weights('tf_model_weights.h5')
        
        # 2. ResNet50 모델 생성
        base_model = build_resnet50_model()
        print("기본 모델 생성 완료")
        
        # 3. 더미 입력으로 모델 초기화
        dummy_input = tf.random.normal((1, 224, 224, 3))
        _ = base_model(dummy_input)
        
        try:
            # 4. 가중치 로드 시도
            base_model.load_weights('tf_model_weights.h5', by_name=True)
            print("모델 가중치 로드 완료")
        except Exception as weight_error:
            print(f"가중치 로드 실패: {str(weight_error)}")
            return
            
        # 5. 랜덤 스무딩 모델 생성
        smooth_model = RandomSmoothingModel(base_model, sigma=0.25)
        print("랜덤 스무딩 모델 생성 완료")
        
        # 6. 테스트
        print("\n임의 이미지로 테스트 수행...")
        test_img = create_test_image()
        test_img = tf.expand_dims(test_img, 0)
        
        pred_class, confidence = smooth_model.random_smoothing_predict(test_img)
        
        print(f"\n예측 결과:")
        print(f"예측 클래스: {pred_class}")
        print(f"신뢰도: {confidence:.4f}")
        print("\n테스트 완료!")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

def predict(self, x, batch_size=32):
    predictions = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size]
        pred_class, confidence = self.random_smoothing_predict(batch)
        predictions.extend(zip(pred_class, confidence))
    return np.array(predictions)

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    test_defense_model()
