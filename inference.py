# inference.py
import torch
from model import YOLOModel
from params import Config
from PIL import Image
import numpy as np

def inference(rgb_img_path, depth_img_path):
    # 모델 로드
    model = YOLOModel(Config.model_save_path)
    
    # RGB 이미지 불러오기
    rgb_image = Image.open(rgb_img_path).convert('RGB')
    
    # Depth 이미지 불러오기
    depth_image = np.load(depth_img_path)
    
    # 예측
    results = model.predict(rgb_image)

    # 결과 출력 (Bounding box를 기반으로 3D 좌표 계산)
    print("Detection results:")
    for result in results:
        bbox = result['boxes'][0].xyxy.tolist()[0]  # 첫 번째 객체의 바운딩 박스 정보
        x, y, z = model.get_xyz(depth_image, bbox, Config.focal_length, Config.principal_point)
        print(f"Object: {result['class']}, XYZ: ({x}, {y}, {z})")

# 예시 실행
if __name__ == "__main__":
    rgb_img_path = "./dataset/rgb/test_image.jpg"
    depth_img_path = "./dataset/depth/test_image.npy"
    inference(rgb_img_path, depth_img_path)
