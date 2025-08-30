import tensorflow as tf
import numpy as np
import cv2
import os
import time

def create_directory(path):
    """디렉토리 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory: {path}')

class ComDefend(tf.keras.Model):
    """ComDefend 모델"""
    def __init__(self):
        super(ComDefend, self).__init__()
        self.code_noise = tf.Variable(1.0)
        self.quantization_threshold = tf.Variable(0.5)
        self.model = tf.keras.models.load_model('third_party/ComDefend/tf2_weights/comdefend_model')
    
    @tf.function
    def call(self, inputs, training=False):
        x = tf.clip_by_value(inputs, 0., 1.)
        
        if training:
            linear_code = self.model.encoder(x)
            noisy_code = linear_code - tf.random.normal(
                stddev=self.code_noise,
                shape=tf.shape(linear_code)
            )
            binary_code = tf.sigmoid(noisy_code)
            return self.model.decoder(binary_code)
        else:
            linear_code = self.model.encoder(x)
            binary_code = tf.cast(
                tf.sigmoid(linear_code) > self.quantization_threshold,
                tf.float32
            )
            return self.model.decoder(binary_code)
        
def process_image(path, output_path, model, threshold=0.5):
    """단일 이미지 처리"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    processed = model(image, training=False)
    
    output = np.clip(processed[0].numpy() * 255, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, output)
    return output

def divide_image(image_path, output_dir, patch_size=(299, 299)):
    """이미지를 패치로 분할"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    height, width = img.shape[:2]
    patches = []
    
    for y in range(0, height, patch_size[0]):
        for x in range(0, width, patch_size[1]):
            patch = img[y:min(y+patch_size[0], height), 
                       x:min(x+patch_size[1], width)]
            
            if patch.shape[:2] != patch_size:
                patch = cv2.resize(patch, patch_size)
            
            patch_path = os.path.join(output_dir, f'patch_{len(patches)}.png')
            cv2.imwrite(patch_path, patch)
            patches.append(patch_path)
    
    return patches

def merge_images(image_dir, original_image_path):
    """패치들을 원본 이미지 크기로 병합"""
    original = cv2.imread(original_image_path)
    if original is None:
        raise ValueError(f"Failed to load original image from {original_image_path}")
    
    height, width = original.shape[:2]
    
    patches = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.png'):
            patch = cv2.imread(os.path.join(image_dir, filename))
            patches.append(patch)
    
    patch_height, patch_width = patches[0].shape[:2]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    patch_idx = 0
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            if patch_idx < len(patches):
                patch = patches[patch_idx]
                h = min(patch_height, height - y)
                w = min(patch_width, width - x)
                result[y:y+h, x:x+w] = cv2.resize(patch, (w, h))
                patch_idx += 1
    
    return result

def defend_image(input_path, output_path):
    """ComDefend를 사용하여 이미지 방어"""
    # ComDefend 모델 로드
    print("Loading ComDefend model...")
    comdefend_model = ComDefend()
    
    # 디렉토리 생성
    create_directory("temp_imagenet")
    create_directory("com_imagenet_temp")
    
    start_time = time.time()
    
    try:
        # 이미지 패치 분할
        patches = divide_image(input_path, "temp_imagenet")
        
        # 각 패치 처리
        for i, patch_path in enumerate(patches):
            output_patch_path = os.path.join("com_imagenet_temp", f'processed_{i}.png')
            process_image(patch_path, output_patch_path, comdefend_model)
            print(f"Processed patch {i+1}/{len(patches)}")
        
        # 처리된 패치 병합
        result = merge_images("com_imagenet_temp", input_path)
        
        # 결과 저장
        cv2.imwrite(output_path, result)
        
        print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return False

if __name__ == '__main__':
    # 사용 예시
    input_image = 'adv.png'  # 적대적 이미지 경로
    output_image = 'defended.png'  # 방어된 이미지 저장 경로
    
    success = defend_image(input_image, output_image)
    if success:
        print(f"Image successfully defended and saved to {output_image}")
