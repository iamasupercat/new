"""YOLOv11 + DINOv2 테스트 파이프라인"""

import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import yaml
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class DINOv2Classifier(nn.Module):
    """DINOv2 분류 모델 (체크포인트 로딩용)"""
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


class YOLODINOPipeline:
    def __init__(self, mode='frontdoor', yolo_model_path=None, 
                 dino_models=None, device='cuda', conf_threshold=0.25,
                 voting_method='soft', project_name='pipeline_test'):
        """
        YOLO + DINOv2 테스트 파이프라인
        
        Args:
            mode (str): 'frontdoor' 또는 'bolt'
            yolo_model_path (str): YOLO 모델 경로
            dino_models (dict): DINOv2 모델 경로들
                - frontdoor: {'high': path, 'mid': path, 'low': path}
                - bolt: {'bolt': path}
            device (str): 디바이스
            conf_threshold (float): YOLO 신뢰도 임계값
            voting_method (str): 'hard' 또는 'soft' (frontdoor용)
            project_name (str): 프로젝트 이름 (결과 폴더명에 사용)
        """
        self.mode = mode.lower()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.voting_method = voting_method
        self.project_name = project_name
        
        # YOLO 모델 로드
        if yolo_model_path is None:
            raise ValueError("YOLO 모델 경로를 제공해야 합니다.")
        print(f"🔄 YOLO 모델 로드 중: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # DINOv2 모델 로드
        self.dino_models = {}
        if dino_models is None:
            raise ValueError("DINOv2 모델 경로를 제공해야 합니다.")
        
        if self.mode == 'frontdoor':
            required_keys = ['high', 'mid', 'low']
            for key in required_keys:
                if key not in dino_models:
                    raise ValueError(f"frontdoor 모드는 {required_keys} 모델이 필요합니다.")
            
            for part, model_path in dino_models.items():
                print(f"🔄 DINOv2 모델 로드 중 ({part}): {model_path}")
                self.dino_models[part] = self._load_dino_model(model_path)
        
        elif self.mode == 'bolt':
            if 'bolt' not in dino_models:
                raise ValueError("bolt 모드는 'bolt' 모델이 필요합니다.")
            print(f"🔄 DINOv2 모델 로드 중 (bolt): {dino_models['bolt']}")
            self.dino_models['bolt'] = self._load_dino_model(dino_models['bolt'])
        
        else:
            raise ValueError(f"지원하지 않는 모드: {self.mode}")
        
        # YOLO 클래스 매핑 (bolt 모드용)
        self.bolt_class_names = {
            0: 'bolt_frontside',
            1: 'bolt_side',
            2: 'sedan (trunklid)',
            3: 'suv (trunklid)',
            4: 'hood',
            5: 'long (frontfender)',
            6: 'mid (frontfender)',
            7: 'short (frontfender)'
        }
        
        # DINOv2 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ 파이프라인 초기화 완료")
        print(f"  - 모드: {self.mode}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - YOLO 신뢰도 임계값: {self.conf_threshold}")
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
    
    def _extract_gt_label(self, img_path):
        """
        이미지 경로에서 GT 라벨 추출
        경로에 'bad' 또는 'defect'가 있으면 불량(1), 'good'이 있으면 양품(0)
        """
        path_lower = img_path.lower()
        
        # 경로를 '/'로 분할하여 폴더명 확인
        parts = path_lower.split('/')
        
        if 'bad' in parts or 'defect' in parts:
            return 1  # 불량
        elif 'good' in parts:
            return 0  # 양품
        else:
            # 파일명에서도 확인
            filename = os.path.basename(path_lower)
            if 'bad' in filename or 'defect' in filename:
                return 1
            elif 'good' in filename:
                return 0
            else:
                return None  # GT를 알 수 없음
    
    def process_image_list(self, txt_file):
        """
        이미지 리스트 처리
        
        Args:
            txt_file (str): 이미지 경로가 담긴 txt 파일
        """
        # 결과 폴더 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = Path('runs') / f"{self.project_name}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 폴더 생성
        crops_dir = result_dir / 'crops'
        vis_dir = result_dir / 'visualizations'
        crops_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        
        # 이미지 경로 읽기
        with open(txt_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*60}")
        print(f"🚀 이미지 처리 시작")
        print(f"{'='*60}")
        print(f"  - 총 이미지 수: {len(image_paths)}")
        print(f"  - 결과 저장 위치: {result_dir}\n")
        
        results = []
        y_true = []
        y_pred = []
        
        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
            if not os.path.exists(img_path):
                print(f"⚠️  이미지를 찾을 수 없습니다: {img_path}")
                continue
            
            # GT 라벨 추출
            gt_label = self._extract_gt_label(img_path)
            
            # 이미지 처리
            result = self.process_single_image(
                img_path, 
                result_dir, 
                crops_dir, 
                vis_dir, 
                idx,
                gt_label
            )
            results.append(result)
            
            # confusion matrix용 데이터 수집
            if gt_label is not None and result['status'] in ['processed', 'defect']:
                y_true.append(gt_label)
                pred_label = 1 if result['final_prediction'] == 'defect' else 0
                y_pred.append(pred_label)
        
        # 결과 저장
        self._save_results(results, result_dir)
        
        # Confusion Matrix 생성
        if len(y_true) > 0:
            self._plot_confusion_matrix(y_true, y_pred, result_dir)
        
        # 통계 출력
        self._print_statistics(results, y_true, y_pred)
        
        return results
    
    def process_single_image(self, img_path, result_dir, crops_dir, vis_dir, idx, gt_label):
        """단일 이미지 처리"""
        try:
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                return {
                    'image_path': img_path,
                    'status': 'error',
                    'message': 'Failed to load image',
                    'gt_label': gt_label
                }
            
            # YOLO 검출
            yolo_results = self.yolo_model.predict(
                img_path, 
                conf=self.conf_threshold,
                verbose=False
            )[0]
            
            boxes = yolo_results.boxes
            
            if self.mode == 'frontdoor':
                result = self._process_frontdoor(
                    img, img_path, boxes, crops_dir, vis_dir, idx, gt_label
                )
            elif self.mode == 'bolt':
                result = self._process_bolt(
                    img, img_path, boxes, crops_dir, vis_dir, idx, gt_label
                )
            
            result['gt_label'] = gt_label
            return result
        
        except Exception as e:
            import traceback
            return {
                'image_path': img_path,
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc(),
                'gt_label': gt_label
            }
    
    def _process_frontdoor(self, img, img_path, boxes, crops_dir, vis_dir, idx, gt_label):
        """프론트도어 처리"""
        # 클래스별 검출 결과 정리
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
        
        # 조건 확인: high/mid/low 각 1개씩 OR high/low 각 1개씩
        has_all_three = (len(detections['high']) == 1 and 
                        len(detections['mid']) == 1 and 
                        len(detections['low']) == 1)
        has_high_low = (len(detections['high']) == 1 and 
                       len(detections['low']) == 1 and 
                       len(detections['mid']) == 0)
        
        if not (has_all_three or has_high_low):
            # 시각화 (검출 실패)
            self._save_visualization(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'Detection condition not met',
                'detections': {k: len(v) for k, v in detections.items()}
            }
        
        # 각 부위별 크롭 및 분류
        part_results = {}
        parts_to_process = ['high', 'mid', 'low'] if has_all_three else ['high', 'low']
        crop_info = []
        
        for part in parts_to_process:
            if len(detections[part]) > 0:
                bbox = detections[part][0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cropped = img[y1:y2, x1:x2]
                
                # 크롭 이미지 저장
                crop_filename = f"{idx:04d}_{part}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped)
                
                # DINOv2 분류
                pred_class, confidence = self._classify_with_dino(cropped, part)
                
                part_results[part] = {
                    'bbox': bbox.tolist(),
                    'yolo_conf': detections[part][0]['conf'],
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'crop_path': str(crop_path)
                }
                
                crop_info.append({
                    'bbox': bbox,
                    'label': f"{part}: {'Bad' if pred_class == 1 else 'Good'} ({confidence[pred_class]:.2f})",
                    'color': (0, 0, 255) if pred_class == 1 else (0, 255, 0)
                })
        
        # Voting
        if self.voting_method == 'hard':
            final_pred = self._hard_voting(part_results)
        else:  # soft
            final_pred = self._soft_voting(part_results)
        
        # 시각화 저장
        self._save_visualization(
            img, img_path, crop_info, vis_dir, idx, 
            final_pred, gt_label, part_results
        )
        
        return {
            'image_path': img_path,
            'status': 'processed',
            'mode': 'frontdoor',
            'parts': part_results,
            'final_prediction': final_pred,
            'voting_method': self.voting_method
        }
    
    def _process_bolt(self, img, img_path, boxes, crops_dir, vis_dir, idx, gt_label):
        """볼트 처리"""
        # 클래스별 검출 결과 정리
        bolt_detections = []  # 0, 1번 클래스 (볼트)
        frame_detections = []  # 2~7번 클래스 (프레임)
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            detection = {
                'class_id': cls_id,
                'class_name': self.bolt_class_names.get(cls_id, 'unknown'),
                'bbox': xyxy,
                'conf': conf,
                'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            }
            
            if cls_id in [0, 1]:  # 볼트
                bolt_detections.append(detection)
            elif cls_id in [2, 3, 4, 5, 6, 7]:  # 프레임
                frame_detections.append(detection)
        
        # 2~7번 프레임이 없으면 스킵
        if len(frame_detections) == 0:
            self._save_visualization(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'No frame detection (class 2-7)',
                'bolt_count': len(bolt_detections),
                'frame_count': len(frame_detections)
            }
        
        # 각 프레임 영역 내의 볼트 찾기
        valid_bolts = []
        for frame in frame_detections:
            frame_bbox = frame['bbox']
            frame_cls = frame['class_id']
            
            # 이 프레임 내의 볼트들
            bolts_in_frame = []
            for bolt in bolt_detections:
                cx, cy = bolt['center']
                if (frame_bbox[0] <= cx <= frame_bbox[2] and 
                    frame_bbox[1] <= cy <= frame_bbox[3]):
                    bolts_in_frame.append(bolt)
            
            # 프레임 종류가 2, 3, 4번일 때 볼트 개수 체크 (sedan, suv, hood)
            if frame_cls in [2, 3, 4]:
                if len(bolts_in_frame) != 2:
                    self._save_visualization(
                        img, img_path, [], vis_dir, idx, 
                        'defect', gt_label, None
                    )
                    return {
                        'image_path': img_path,
                        'status': 'defect',
                        'reason': 'bolt_count_mismatch',
                        'frame_class': frame['class_name'],
                        'expected_bolts': 2,
                        'actual_bolts': len(bolts_in_frame),
                        'final_prediction': 'defect'
                    }
            
            valid_bolts.extend(bolts_in_frame)
        
        # 볼트가 없으면 불량
        if len(valid_bolts) == 0:
            self._save_visualization(
                img, img_path, [], vis_dir, idx, 
                'defect', gt_label, None
            )
            return {
                'image_path': img_path,
                'status': 'defect',
                'reason': 'no_bolts_in_frame',
                'final_prediction': 'defect'
            }
        
        # 각 볼트를 DINOv2로 분류
        bolt_results = []
        crop_info = []
        
        for bolt_idx, bolt in enumerate(valid_bolts):
            bbox = bolt['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cropped = img[y1:y2, x1:x2]
            
            # 크롭 이미지 저장
            crop_filename = f"{idx:04d}_bolt_{bolt_idx}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            
            pred_class, confidence = self._classify_with_dino(cropped, 'bolt')
            
            bolt_results.append({
                'bbox': bbox.tolist(),
                'yolo_class': bolt['class_name'],
                'yolo_conf': bolt['conf'],
                'pred_class': pred_class,
                'confidence': confidence,
                'crop_path': str(crop_path)
            })
            
            crop_info.append({
                'bbox': bbox,
                'label': f"Bolt: {'Bad' if pred_class == 1 else 'Good'} ({confidence[pred_class]:.2f})",
                'color': (0, 0, 255) if pred_class == 1 else (0, 255, 0)
            })
        
        # 하나라도 불량이면 전체 불량
        has_defect = any(b['pred_class'] == 1 for b in bolt_results)
        final_pred = 'defect' if has_defect else 'good'
        
        # 시각화 저장
        self._save_visualization(
            img, img_path, crop_info, vis_dir, idx, 
            final_pred, gt_label, bolt_results
        )
        
        return {
            'image_path': img_path,
            'status': 'processed',
            'mode': 'bolt',
            'bolt_count': len(valid_bolts),
            'bolt_results': bolt_results,
            'final_prediction': final_pred
        }
    
    def _save_visualization(self, img, img_path, crop_info, vis_dir, idx, 
                           prediction, gt_label, detail_results):
        """시각화 이미지 저장"""
        vis_img = img.copy()
        
        # 바운딩 박스 그리기
        for crop in crop_info:
            bbox = crop['bbox']
            label = crop['label']
            color = crop['color']
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 배경
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 오른쪽 상단에 결과 표시
        h, w = vis_img.shape[:2]
        
        # GT vs Prediction 비교
        if gt_label is not None:
            gt_text = "GT: Good" if gt_label == 0 else "GT: Bad"
            pred_text = f"Pred: {prediction.capitalize()}"
            
            # 정답 여부 판단
            pred_label = 1 if prediction == 'defect' else 0
            is_correct = (gt_label == pred_label)
            result_symbol = "✓" if is_correct else "✗"
            result_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            # 배경 사각형
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 110), (0, 0, 0), -1)
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 110), (255, 255, 255), 2)
            
            # 텍스트
            cv2.putText(vis_img, gt_text, (w - 240, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, pred_text, (w - 240, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, result_symbol, (w - 240, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 3)
        else:
            # GT 없는 경우
            pred_text = f"Pred: {prediction.capitalize()}"
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 60), (0, 0, 0), -1)
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 60), (255, 255, 255), 2)
            cv2.putText(vis_img, pred_text, (w - 240, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 저장
        vis_filename = f"{idx:04d}_vis.jpg"
        vis_path = vis_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_img)
    
    def _classify_with_dino(self, cropped_img, part):
        """DINOv2로 크롭된 이미지 분류"""
        if cropped_img.size == 0:
            return 1, [0.0, 1.0]  # 빈 이미지는 불량으로
        
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
        """Hard Voting: 다수결"""
        votes = [result['pred_class'] for result in part_results.values()]
        defect_count = sum(votes)
        
        # 과반수 이상이 불량이면 불량
        if defect_count > len(votes) / 2:
            return 'defect'
        else:
            return 'good'
    
    def _soft_voting(self, part_results):
        """Soft Voting: 평균 confidence"""
        # 각 부위의 불량(class 1) confidence 평균
        defect_confidences = [result['confidence'][1] for result in part_results.values()]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences)
        
        # 평균이 0.5 이상이면 불량
        if avg_defect_conf > 0.5:
            return 'defect'
        else:
            return 'good'
    
    def _save_results(self, results, result_dir):
        """결과 저장"""
        output_file = result_dir / 'results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 결과 저장: {output_file}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, result_dir):
        """Confusion Matrix 생성 및 저장"""
        # Confusion Matrix 계산
        cm = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
        
        for true, pred in zip(y_true, y_pred):
            cm[true][pred] += 1
        
        # 시각화
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Good', 'Defect'],
                   yticklabels=['Good', 'Defect'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        cm_path = result_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion Matrix 저장: {cm_path}")
        
        # 메트릭 계산
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 메트릭 저장
        metrics = {
            'confusion_matrix': {
                'TN': int(tn), 'FP': int(fp),
                'FN': int(fn), 'TP': int(tp)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        }
        
        metrics_path = result_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ 메트릭 저장: {metrics_path}")
        
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + DINOv2 테스트 파이프라인")
    parser.add_argument("--config", required=True, type=str, help="모델 경로들이 들어있는 YAML 파일 경로")
    parser.add_argument("--txt", required=True, type=str, help="처리할 이미지 경로 목록이 담긴 txt 파일 경로")
    parser.add_argument("--mode", required=True, choices=["frontdoor", "bolt"], help="실행 모드")
    parser.add_argument("--voting", default="soft", choices=["soft", "hard"], help="frontdoor 모드에서의 보팅 방식")
    parser.add_argument("--project", default="pipeline_test", type=str, help="runs 하위 결과 폴더명 prefix")
    parser.add_argument("--conf", default=0.25, type=float, help="YOLO 신뢰도 임계값")
    parser.add_argument("--device", default="cuda", type=str, help="디바이스 (cuda|cpu)")
    return parser.parse_args()


def load_models_from_yaml(config_path, mode):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # YAML은 모델 경로만 포함한다고 가정
    yolo_model_path = cfg.get("yolo_model") or cfg.get("yolo") or cfg.get("yolo_model_path")
    if yolo_model_path is None:
        raise ValueError("YAML에 'yolo_model' 경로가 필요합니다.")

    dino_models = {}
    if mode == "frontdoor":
        # 예상 키: high, mid, low
        for key in ["high", "mid", "low"]:
            if key not in cfg:
                raise ValueError("frontdoor 모드는 YAML에 'high', 'mid', 'low' 키가 필요합니다.")
            dino_models[key] = cfg[key]
    else:
        # bolt 모드: bolt 단일 키
        bolt_path = cfg.get("bolt")
        if bolt_path is None:
            raise ValueError("bolt 모드는 YAML에 'bolt' 키가 필요합니다.")
        dino_models["bolt"] = bolt_path

    return yolo_model_path, dino_models


def main():
    args = parse_args()

    # YAML에서 경로들 로드
    yolo_model_path, dino_models = load_models_from_yaml(args.config, args.mode)

    # 파이프라인 생성
    pipeline = YOLODINOPipeline(
        mode=args.mode,
        yolo_model_path=yolo_model_path,
        dino_models=dino_models,
        device=args.device,
        conf_threshold=args.conf,
        voting_method=args.voting,
        project_name=args.project,
    )

    # 이미지 리스트 처리
    pipeline.process_image_list(args.txt)


if __name__ == "__main__":
    main()