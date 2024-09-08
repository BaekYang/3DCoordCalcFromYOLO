# model.py
import torch
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, img):
        results = self.model(img)  # YOLOv8을 이용한 예측
        return results

    def get_xyz(self, depth_image, bounding_box, focal_length, principal_point):
        """
        bounding_box: YOLO 모델의 탐지 결과에서 얻은 bounding box 좌표 (x1, y1, x2, y2)
        depth_image: depth 카메라로 얻은 깊이 정보
        focal_length: 카메라 초점 거리
        principal_point: 주점 좌표 (cx, cy)
        
        bounding_box의 중심점을 기준으로 해당 지점의 XYZ 좌표를 계산
        """
        # 바운딩 박스의 중심점 좌표 계산
        x_center = int((bounding_box[0] + bounding_box[2]) / 2)
        y_center = int((bounding_box[1] + bounding_box[3]) / 2)

        # 중심점에서의 깊이 값(Z 값)
        z_value = depth_image[y_center, x_center]

        # 카메라 주점(cx, cy)
        cx, cy = principal_point

        # 3D 좌표 계산
        x = (x_center - cx) * z_value / focal_length
        y = (y_center - cy) * z_value / focal_length
        z = z_value

        return x, y, z
