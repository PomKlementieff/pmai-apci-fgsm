'''
Purify adversarial images within l_inf <= 16/255 (TensorFlow version)
'''
import tensorflow as tf
import numpy as np
import os
import argparse
import cv2
from glob import glob
import matplotlib.pyplot as plt

def create_test_adversarial_images():
    """테스트용 적대적 이미지 생성 (256x256 크기)"""
    if not os.path.exists('adv_images'):
        os.makedirs('adv_images')
    
    # 샘플 이미지 생성 (3개)
    for i in range(3):
        # 깨끗한 이미지 생성 (256x256 크기)
        img = np.ones((256, 256, 3)) * 0.5  # 회색 배경
        cv2.circle(img, (128, 128), 60, (0.8, 0.2, 0.2), -1)  # 빨간 원
        
        # 적대적 노이즈 추가
        noise = np.random.uniform(-16/255, 16/255, img.shape)
        adv_img = np.clip(img + noise, 0, 1)
        
        # 저장
        save_path = os.path.join('adv_images', f'adv_image_{i}.png')
        save_image(adv_img, save_path)

def purify_image(img, model, dynamic=False):
    """이미지 정제 함수"""
    # 입력 이미지를 float32로 변환 및 정규화
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    
    # 256x256으로 리사이즈
    if img.shape[:2] != (256, 256):
        img = tf.image.resize(img, (256, 256))
    
    # Dynamic inference를 위한 노이즈 추가
    if dynamic:
        eps = 16/255
        noise = tf.random.normal(img.shape, mean=0, stddev=0.05)
        img_m = img + noise
        # Projection
        img_m = tf.clip_by_value(img_m, img - eps, img + eps)
        img_m = tf.clip_by_value(img_m, 0.0, 1.0)
    else:
        img_m = img
    
    # 배치 차원 추가
    if len(img_m.shape) == 3:
        img_m = tf.expand_dims(img_m, 0)
    
    # 이미지 정제
    purified = model(img_m, training=False)
    purified = tf.clip_by_value(purified, 0.0, 1.0)
    
    return purified[0].numpy()  # 배치 차원 제거 및 NumPy array로 변환

def load_image(path):
    """이미지 로드 함수"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(img, path):
    """이미지 저장 함수"""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def main():
    parser = argparse.ArgumentParser(description='Purify Images (TensorFlow version)')
    parser.add_argument('--dir', default='adv_images/')
    parser.add_argument('--purifier', type=str, default='NRP', help='NRP, NRP_resG')
    parser.add_argument('--dynamic', action='store_true', 
                       help='Dynamic inference (in case of whitebox attack)')
    args = parser.parse_args()
    print(args)

    # 테스트용 적대적 이미지 생성
    if not os.path.exists(args.dir) or len(os.listdir(args.dir)) == 0:
        print("Creating test adversarial images...")
        create_test_adversarial_images()

    # 모델 로드
    if args.purifier == 'NRP':
        model_path = 'tf_models/nrp_model'
    else:  # NRP_resG
        model_path = 'tf_models/nrp_resG_model'
    
    print(f"Loading {args.purifier} model...")
    model = tf.keras.models.load_model(model_path)
    
    # 출력 디렉토리 생성
    os.makedirs('purified_imgs', exist_ok=True)
    
    # 이미지 처리
    image_paths = glob(os.path.join(args.dir, '*'))
    print(f"Found {len(image_paths)} images to process")
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        
        try:
            # 이미지 로드
            img = load_image(img_path)
            img = img.astype(np.float32) / 255.0
            
            # 이미지 크기 조정 (필요한 경우)
            if img.shape[:2] != (256, 256):
                img = cv2.resize(img, (256, 256))
            
            # 이미지 정제
            purified = purify_image(img, model, args.dynamic)
            
            # 결과 저장
            output_path = os.path.join('purified_imgs', os.path.basename(img_path))
            save_image(purified, output_path)

            # 시각화
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(img)
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(purified)
            plt.title('Purified')
            plt.axis('off')
            
            plt.subplot(133)
            diff = np.abs(img - purified) * 5
            plt.imshow(diff)
            plt.title('Difference (×5)')
            plt.axis('off')
            
            plt.savefig(os.path.join('purified_imgs', f'comparison_{i}.png'))
            plt.close()

            print(f"Saved purified image and comparison to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    print("Processing complete!")

if __name__ == '__main__':
    main()
