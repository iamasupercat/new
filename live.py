import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time
from datetime import datetime
import yaml


class DINOv2Classifier(nn.Module):
    """DINOv2 분류 모델"""
    def __init__(self, backbone, embed_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class RealtimeInspectionSystem:
    def __init__(self, mode='frontdoor', yolo_model_path=None, dino_models=None,
                 device='cuda', conf_threshold=0.25, voting_method='soft'):
        """
        실시간 카메라 검사 시스템
        
        Args:
            mode (str): 'frontdoor' 또는 'bolt'
            yolo_model_path (str): YOLO 모델 경로
            dino_models (dict): DINOv2 모델 경로들
            device (str): 디바이스
            conf_threshold (float): YOLO 신뢰도 임계값
            voting_method (str): 'hard' 또는 'soft' (frontdoor용)
        """
        self.mode = mode.lower()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.voting_method = voting_method
        
        # YOLO 모델 로드
        print(f"🔄 YOLO 모델 로드 중: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # DINOv2 모델 로드
        self.dino_models = {}
        if self.mode == 'frontdoor':
            for part in ['high', 'mid', 'low']:
                print(f"🔄 DINOv2 모델 로드 중 ({part}): {dino_models[part]}")
                self.dino_models[part] = self._load_dino_model(dino_models[part])
        else:  # bolt
            print(f"🔄 DINOv2 모델 로드 중 (bolt): {dino_models['bolt']}")
            self.dino_models['bolt'] = self._load_dino_model(dino_models['bolt'])
        
        # DINOv2 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 조건 체크 변수
        self.condition_start_time = None
        self.condition_met = False
        self.last_valid_frame = None
        self.last_valid_detections = None
        
        # 타임아웃 설정
        if self.mode == 'frontdoor':
            self.required_duration = 3.0  # 3초
        else:  # bolt
            self.required_duration = 5.0  # 5초
        
        print(f"✓ 실시간 검사 시스템 초기화 완료")
        print(f"  - 모드: {self.mode}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - YOLO 신뢰도: {self.conf_threshold}")
        print(f"  - 조건 유지 시간: {self.required_duration}초")
        if self.mode == 'frontdoor':
            print(f"  - Voting 방법: {self.voting_method}")
    
    def _load_dino_model(self, model_path):
        """DINOv2 모델 체크포인트 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        model_size = config.get('model_size', 'small')
        num_classes = config.get('num_classes', 2)
        
        # 백본 로드
        model_map = {
            'small': ('dinov2_vits14', 384),
            'base': ('dinov2_vitb14', 768),
            'large': ('dinov2_vitl14', 1024),
            'giant': ('dinov2_vitg14', 1536)
        }
        model_name, embed_dim = model_map.get(model_size, ('dinov2_vits14', 384))
        
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        model = DINOv2Classifier(backbone, embed_dim, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def run(self, source=0):
        """
        실시간 검사 실행
        
        Args:
            source: 카메라 소스 (0: 웹캠, 또는 RTSP URL 등)
        """
        print(f"\n{'='*60}")
        print(f"🎥 카메라 시작: {source}")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ 카메라를 열 수 없습니다: {source}")
            return
        
        print(f"✓ 카메라 연결 성공")
        print(f"📋 대기 중... (조건이 만족되면 자동으로 캡처됩니다)")
        print(f"   종료하려면 'q' 키를 누르세요\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  프레임을 읽을 수 없습니다")
                    break
                
                # YOLO 검출
                results = self.yolo_model.predict(
                    frame, 
                    conf=self.conf_threshold,
                    verbose=False
                )[0]
                
                # 조건 확인
                condition_satisfied, detections = self._check_condition(results.boxes)
                
                # 화면에 표시
                display_frame = self._draw_detections(frame.copy(), results.boxes)
                
                # 조건 만족 여부에 따른 처리
                if condition_satisfied:
                    if not self.condition_met:
                        # 조건이 처음 만족됨
                        self.condition_met = True
                        self.condition_start_time = time.time()
                        print(f"✓ 조건 만족! 타이머 시작...")
                    
                    # 경과 시간 계산
                    elapsed = time.time() - self.condition_start_time
                    
                    # 유효한 프레임 저장
                    self.last_valid_frame = frame.copy()
                    self.last_valid_detections = detections
                    
                    # 타이머 표시
                    timer_text = f"Timer: {elapsed:.1f}s / {self.required_duration}s"
                    cv2.putText(display_frame, timer_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 조건 유지 시간 충족 확인
                    if elapsed >= self.required_duration:
                        print(f"\n{'='*60}")
                        print(f"📸 조건이 {self.required_duration}초 이상 유지됨! 검사 시작...")
                        print(f"{'='*60}\n")
                        
                        # 카메라 종료
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        # 검사 수행
                        self._perform_inspection(self.last_valid_frame, self.last_valid_detections)
                        return
                else:
                    if self.condition_met:
                        # 조건이 해제됨
                        print(f"⚠️  조건 해제됨. 타이머 리셋.")
                        self.condition_met = False
                        self.condition_start_time = None
                        self.last_valid_frame = None
                        self.last_valid_detections = None
                    
                    # 상태 표시
                    status_text = "Waiting for condition..."
                    cv2.putText(display_frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 화면 표시
                cv2.imshow('Real-time Inspection', display_frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n사용자가 종료함")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _check_condition(self, boxes):
        """조건 확인"""
        if self.mode == 'frontdoor':
            return self._check_frontdoor_condition(boxes)
        else:  # bolt
            return self._check_bolt_condition(boxes)
    
    def _check_frontdoor_condition(self, boxes):
        """프론트도어 조건 확인: high, mid, low 각 1개씩"""
        detections = {'high': [], 'mid': [], 'low': []}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            class_name = self.yolo_model.names[cls_id].lower()
            if class_name in detections:
                detections[class_name].append({
                    'bbox': xyxy,
                    'conf': conf
                })
        
        # 조건: high, mid, low 각 1개씩
        condition_met = (len(detections['high']) == 1 and 
                        len(detections['mid']) == 1 and 
                        len(detections['low']) == 1)
        
        return condition_met, detections
    
    def _check_bolt_condition(self, boxes):
        """볼트 조건 확인: 3~8번 프레임 객체 정확히 1개"""
        bolt_detections = []  # 0, 1, 2번
        frame_detections = []  # 3~8번
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            detection = {
                'class_id': cls_id,
                'bbox': xyxy,
                'conf': conf,
                'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            }
            
            if cls_id in [0, 1, 2]:  # 볼트
                bolt_detections.append(detection)
            elif cls_id in [3, 4, 5, 6, 7, 8]:  # 프레임
                frame_detections.append(detection)
        
        # 조건: 프레임 객체 정확히 1개
        condition_met = len(frame_detections) == 1
        
        detections = {
            'bolts': bolt_detections,
            'frames': frame_detections
        }
        
        return condition_met, detections
    
    def _draw_detections(self, frame, boxes):
        """검출 결과를 프레임에 그리기"""
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            x1, y1, x2, y2 = map(int, xyxy)
            
            # 클래스명
            class_name = self.yolo_model.names[cls_id]
            
            # 색상 결정
            if self.mode == 'frontdoor':
                color = (0, 255, 0) if class_name.lower() in ['high', 'mid', 'low'] else (128, 128, 128)
            else:  # bolt
                if cls_id in [0, 1, 2]:
                    color = (255, 0, 0)  # 파란색 (볼트)
                elif cls_id in [3, 4, 5, 6, 7, 8]:
                    color = (0, 255, 0)  # 초록색 (프레임)
                else:
                    color = (128, 128, 128)
            
            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _perform_inspection(self, frame, detections):
        """검사 수행"""
        if self.mode == 'frontdoor':
            self._inspect_frontdoor(frame, detections)
        else:  # bolt
            self._inspect_bolt(frame, detections)
    
    def _inspect_frontdoor(self, frame, detections):
        """프론트도어 검사"""
        print(f"🔍 프론트도어 검사 중...\n")
        
        part_results = {}
        
        for part in ['high', 'mid', 'low']:
            if len(detections[part]) > 0:
                bbox = detections[part][0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cropped = frame[y1:y2, x1:x2]
                
                # DINOv2 분류
                pred_class, confidence = self._classify_with_dino(cropped, part)
                
                part_results[part] = {
                    'pred_class': pred_class,
                    'confidence': confidence
                }
                
                result_text = "양품" if pred_class == 0 else "불량"
                print(f"  [{part.upper()}] {result_text} (신뢰도: {confidence[pred_class]:.2%})")
        
        # Voting
        print(f"\n📊 최종 판정 ({self.voting_method.upper()} Voting):")
        if self.voting_method == 'hard':
            final_result = self._hard_voting(part_results)
        else:  # soft
            final_result = self._soft_voting(part_results)
        
        print(f"  결과: {'✅ 양품' if final_result == 'good' else '❌ 불량'}")
        print(f"\n{'='*60}\n")
    
    def _inspect_bolt(self, frame, detections):
        """볼트 검사"""
        print(f"🔍 볼트 검사 중...\n")
        
        frame_obj = detections['frames'][0]
        frame_bbox = frame_obj['bbox']
        frame_cls = frame_obj['class_id']
        
        # 프레임 클래스명
        frame_class_names = {
            3: 'sedan (trunklid)',
            4: 'suv (trunklid)',
            5: 'hood',
            6: 'long (frontfender)',
            7: 'mid (frontfender)',
            8: 'short (frontfender)'
        }
        frame_name = frame_class_names.get(frame_cls, 'unknown')
        
        print(f"  프레임 타입: {frame_name}")
        
        # 프레임 내 볼트 찾기
        bolts_in_frame = []
        for bolt in detections['bolts']:
            cx, cy = bolt['center']
            if (frame_bbox[0] <= cx <= frame_bbox[2] and 
                frame_bbox[1] <= cy <= frame_bbox[3]):
                bolts_in_frame.append(bolt)
        
        print(f"  프레임 내 볼트 개수: {len(bolts_in_frame)}")
        
        # 3~5번 프레임: 볼트 2개 체크
        if frame_cls in [3, 4, 5]:
            if len(bolts_in_frame) != 2:
                print(f"\n📊 최종 판정:")
                print(f"  결과: ❌ 불량 (볼트 개수 불일치: {len(bolts_in_frame)}/2)")
                print(f"\n{'='*60}\n")
                return
        
        # 볼트가 없으면 불량
        if len(bolts_in_frame) == 0:
            print(f"\n📊 최종 판정:")
            print(f"  결과: ❌ 불량 (프레임 내 볼트 없음)")
            print(f"\n{'='*60}\n")
            return
        
        # 각 볼트 검사
        print(f"\n  볼트별 검사:")
        bolt_results = []
        for i, bolt in enumerate(bolts_in_frame):
            bbox = bolt['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cropped = frame[y1:y2, x1:x2]
            
            pred_class, confidence = self._classify_with_dino(cropped, 'bolt')
            bolt_results.append(pred_class)
            
            result_text = "양품" if pred_class == 0 else "불량"
            print(f"    볼트 #{i+1}: {result_text} (신뢰도: {confidence[pred_class]:.2%})")
        
        # 최종 판정
        has_defect = any(r == 1 for r in bolt_results)
        
        print(f"\n📊 최종 판정:")
        if has_defect:
            print(f"  결과: ❌ 불량 (불량 볼트 존재)")
        else:
            print(f"  결과: ✅ 양품")
        print(f"\n{'='*60}\n")
    
    def _classify_with_dino(self, cropped_img, part):
        """DINOv2로 분류"""
        if cropped_img.size == 0:
            return 1, [0.0, 1.0]
        
        # BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        
        # 전처리
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.dino_models[part](img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0].cpu().numpy().tolist()
        
        return pred_class, confidence
    
    def _hard_voting(self, part_results):
        """Hard Voting"""
        votes = [result['pred_class'] for result in part_results.values()]
        defect_count = sum(votes)
        
        if defect_count > len(votes) / 2:
            return 'defect'
        else:
            return 'good'
    
    def _soft_voting(self, part_results):
        """Soft Voting"""
        defect_confidences = [result['confidence'][1] for result in part_results.values()]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences)
        
        if avg_defect_conf > 0.5:
            return 'defect'
        else:
            return 'good'


def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['mode', 'yolo_model']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"설정 파일에 '{key}' 필드가 없습니다")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='실시간 카메라 양불량 검사 시스템')
    
    parser.add_argument('--config', type=str, required=True,
                        help='설정 YAML 파일 경로')
    parser.add_argument('--source', type=str, default='0',
                        help='카메라 소스 (0: 웹캠, RTSP URL 등, 기본값: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='디바이스 (기본값: cuda)')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config = load_config(args.config)
    
    mode = config['mode'].lower()
    yolo_model = config['yolo_model']
    conf_threshold = config.get('conf_threshold', 0.25)
    
    # DINOv2 모델 설정
    dino_models = {}
    if mode == 'frontdoor':
        dino_models = {
            'high': config['dino_high'],
            'mid': config['dino_mid'],
            'low': config['dino_low']
        }
        voting_method = config.get('voting_method', 'soft')
    else:  # bolt
        dino_models = {
            'bolt': config['dino_bolt']
        }
        voting_method = 'soft'
    
    # 카메라 소스 처리
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # 시스템 초기화
    system = RealtimeInspectionSystem(
        mode=mode,
        yolo_model_path=yolo_model,
        dino_models=dino_models,
        device=args.device,
        conf_threshold=conf_threshold,
        voting_method=voting_method
    )
    
    # 실행
    system.run(source=source)


if __name__ == "__main__":
    # 예시 1: 프론트도어 검사 (웹캠)
    # python realtime_inspection.py --config configs/frontdoor_realtime.yaml --source 0
    
    # 예시 2: 볼트 검사 (외부 카메라)
    # python realtime_inspection.py --config configs/bolt_realtime.yaml --source 1
    
    # 예시 3: RTSP 카메라
    # python realtime_inspection.py --config configs/frontdoor_realtime.yaml --source "rtsp://192.168.1.100:554/stream"
    
    main()