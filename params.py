# params.py

class Config:
    # 모델 설정
    num_classes = 3  # 접시(dish), 주전자(pot), 컵(cup)
    labels = {0: 'dish', 1: 'pot', 2: 'cup'}

    # 경로 설정 (상대 경로)
    dataset_path = './dataset/'  # 데이터셋 경로
    model_save_path = './models/yolo_model.pt'  # 모델 저장 경로

    # 학습 설정
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    # YOLOv8 모델 설정
    yolo_model = 'yolov8n.pt'  # YOLOv8 모델 (Segmentation 대신 Detection만 사용)
    input_size = 640  # 이미지 크기

    # Depth 카메라 및 캘리브레이션 정보
    focal_length = 615.0  # 예시로 615 픽셀로 설정 (실제 값으로 교체 필요)
    principal_point = (320.0, 240.0)  # 주점 (cx, cy)
