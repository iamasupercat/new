"""
    python yolov11_obb.py \
    --project Bolt \
    --data-yaml yaml/Bolt.yaml \
    --obb \
    --convert-format \
    --tune

    python yolov11_obb.py \
    --project Door \
    --data-yaml yaml/Door.yaml \
    --obb \
    --convert-format \
    --tune 


"""

import argparse
from ultralytics import YOLO
import yaml
import os
import shutil
import math
import signal
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # (pip install tqdm)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class YOLOv11Trainer:
    def __init__(self, project_name='yolo_training', use_obb=False, 
                 model='yolo11s.pt', epochs=100, imgsz=640, batch=16):
        """
        YOLOv11 학습 클래스
        """
        self.project_name = project_name
        self.use_obb = use_obb
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{self.project_name}_{self.timestamp}"
        
        # 복원을 위해 변환된 라벨 목록 추적
        self.modified_labels = [] 
        
        self.config = {
            'model': model,
            'data': f'{project_name}_data_{self.timestamp}.yaml',
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'patience': 10,
            'project': 'runs',
            'name': self.run_name,
            'task': 'obb' if use_obb else 'detect',
            'classes': [] 
        }

    def _xywhr_to_corners(self, cx, cy, w, h, angle_rad):
        """
        (cx, cy, w, h, angle_rad) -> (x1, y1, x2, y2, x3, y3, x4, y4) 변환
        angle_rad: 라디안(radian) 단위의 회전 각도
        """
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        wx, wy = (w / 2) * cos_a, (w / 2) * sin_a
        hx, hy = -(h / 2) * sin_a, (h / 2) * cos_a
        p1x, p1y = cx - wx - hx, cy - wy - hy
        p2x, p2y = cx + wx - hx, cy + wy - hy
        p3x, p3y = cx + wx + hx, cy + wy + hy
        p4x, p4y = cx - wx + hx, cy - wy + hy
        return [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]

    def _find_label_path(self, img_path):
        """이미지 경로를 기반으로 라벨 파일 경로 탐색"""
        img_path = Path(img_path)
        # 1. .../images/train/img.jpg -> .../labels/train/img.txt
        label_path = Path(str(img_path.parent).replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)) / f"{img_path.stem}.txt"
        if label_path.exists():
            return label_path
        
        # 2. .../dataset/train/img.jpg -> .../dataset/train/img.txt (같은 폴더)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            return label_path

        # 3. .../images/img.jpg -> .../labels/img.txt (단순 구조)
        label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
        if label_path.exists():
            return label_path
            
        return None

    def preprocess_obb_dataset(self, image_txt_path, subset_name):
        """
        [효율화] 원본 라벨(xywha)을 백업(.bak)하고, 
        YOLO OBB 포맷(xyxyxyxy)으로 변환하여 원본 위치에 덮어쓰기
        """
        if not image_txt_path or not os.path.exists(image_txt_path):
            print(f"⚠️  [{subset_name}] 이미지 목록 파일을 찾을 수 없어 변환을 건너뜁니다: {image_txt_path}")
            return

        print(f"\n🔄 [{subset_name}] OBB 라벨 포맷 변환 시작 (In-place 방식)...")
        
        with open(image_txt_path, 'r') as f:
            img_paths = [line.strip() for line in f if line.strip()]

        iterator = tqdm(img_paths, desc=f"Converting {subset_name}") if 'tqdm' in globals() else img_paths
        
        converted_count = 0
        for img_path in iterator:
            src_label_path = self._find_label_path(img_path)
            
            if src_label_path is None or not src_label_path.exists():
                continue # 라벨 파일이 없는 경우 (배경 이미지 등)
            
            backup_label_path = src_label_path.with_suffix(f"{src_label_path.suffix}.bak")
            
            if backup_label_path.exists():
                continue

            try:
                shutil.move(src_label_path, backup_label_path)
                self.modified_labels.append((src_label_path, backup_label_path))
            except Exception as e:
                print(f"❌ 라벨 백업 실패: {e}")
                continue

            new_lines = []
            try:
                with open(backup_label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 6:
                        continue 
                        
                    cls_id = int(parts[0])
                    cx, cy, w, h, angle = parts[1:6] # x,y,w,h,angle (라디안 가정)
                    
                    # [주의] 데이터가 '도(degree)' 단위인 경우 아래 주석 해제
                    # angle = math.radians(angle) 
                    
                    corners = self._xywhr_to_corners(cx, cy, w, h, angle)
                    corner_str = " ".join([f"{c:.6f}" for c in corners])
                    new_lines.append(f"{cls_id} {corner_str}\n")
                
                with open(src_label_path, 'w') as f:
                    f.writelines(new_lines)
                
                converted_count += 1
                
            except Exception as e:
                print(f"❌ 라벨 변환 실패: {e}. 백업 파일 복구 중...")
                shutil.move(backup_label_path, src_label_path)
                self.modified_labels.remove((src_label_path, backup_label_path))

        print(f"✓ [{subset_name}] 변환 완료: {converted_count}개 라벨 처리됨")

    def restore_original_labels(self):
        """백업했던 .bak 파일들을 원본 .txt로 복원"""
        if not self.modified_labels:
            return
            
        print(f"\n🔄 학습 완료. 원본 라벨 파일 복원 중...")
        
        iterator = tqdm(self.modified_labels, desc="Restoring labels") if 'tqdm' in globals() else self.modified_labels
        
        restored_count = 0
        for original_path, backup_path in iterator:
            try:
                if backup_path.exists():
                    shutil.move(backup_path, original_path)
                    restored_count += 1
            except Exception as e:
                print(f"❌ 복원 실패: {backup_path} -> {original_path}. 오류: {e}")
                
        print(f"✓ 복원 완료: {restored_count}개 파일 복원됨.")
        self.modified_labels = []

    def load_data_yaml(self, yaml_path, convert_format=False):
        """기존 데이터 YAML 파일 로드"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        self.config['data'] = yaml_path
        if isinstance(data_config.get('names'), dict):
            self.config['classes'] = list(data_config['names'].values())
        elif data_config.get('names'):
            self.config['classes'] = data_config['names']
        
        # YAML에서 train, val, test 경로 추출
        train_txt = data_config.get('train')
        val_txt = data_config.get('val')
        test_txt = data_config.get('test')
        
        # 라벨 포맷 변환이 필요한 경우 수행
        if convert_format and self.use_obb:
            print(f"\n🚀 데이터셋 포맷 변환 모드 활성화 (xywha -> xyxyxyxy)")
            if train_txt:
                self.preprocess_obb_dataset(train_txt, 'train')
            if val_txt:
                self.preprocess_obb_dataset(val_txt, 'val')
            if test_txt:
                self.preprocess_obb_dataset(test_txt, 'test')
        
        print(f"✓ 데이터 YAML 파일 로드: {yaml_path}")
        return yaml_path
    
    def create_data_yaml(self, train_txt, val_txt, class_names, test_txt=None, convert_format=False):
        """
        데이터셋 YAML 파일 생성
        """
        if not os.path.exists(train_txt):
            raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {train_txt}")
        if not os.path.exists(val_txt):
            raise FileNotFoundError(f"검증 데이터 파일을 찾을 수 없습니다: {val_txt}")
        
        if convert_format and self.use_obb:
            print(f"\n🚀 데이터셋 포맷 변환 모드 활성화 (xywha -> xyxyxyxy)")
            self.preprocess_obb_dataset(train_txt, 'train')
            self.preprocess_obb_dataset(val_txt, 'val')
            if test_txt:
                self.preprocess_obb_dataset(test_txt, 'test')
        
        self.config['classes'] = class_names
        
        data_yaml = {
            'train': os.path.abspath(train_txt),
            'val': os.path.abspath(val_txt),
            'nc': len(class_names),
            'names': class_names
        }
        
        if test_txt and os.path.exists(test_txt):
            data_yaml['test'] = os.path.abspath(test_txt)
        
        yaml_path = self.config['data']
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✓ 데이터 YAML 파일 생성: {yaml_path}")
        print(f"  - 학습 데이터: {train_txt}")
        print(f"  - 검증 데이터: {val_txt}")
        
        return yaml_path
    
    def tune_hyperparameters(self, iterations=30):
        """
        하이퍼파라미터 자동 튜닝
        
        각 iteration마다:
        - 모델을 학습하고 validation을 수행
        - best.pt (validation 성능이 가장 좋은 모델)와 last.pt (마지막 epoch 모델) 저장
        - confusion matrix는 best.pt로 validation set을 평가한 결과입니다.
        """
        print(f"\n{'='*60}")
        print(f"🔧 하이퍼파라미터 자동 튜닝 시작")
        print(f"{'='*60}\n")
        
        # 모델 로드
        model_type = self.config['model']
        if self.use_obb and self.config['task'] == 'obb':
            if not model_type.endswith('-obb.pt') and 'yolo' in model_type:
                model_type = model_type.replace('.pt', '-obb.pt')
        
        model = YOLO(model_type)
        
        print(f"📊 튜닝 설정:")
        print(f"  - 모델: {model_type}")
        print(f"  - 태스크: {self.config['task']} ({'회전 박스' if self.use_obb else '일반 박스'})")
        print(f"  - 데이터: {self.config['data']}")
        print(f"  - 반복 횟수: {iterations}")
        print(f"  - 저장 위치: {self.config['project']}/{self.run_name}_tune")
        print(f"  - 각 iteration의 confusion matrix는 best.pt로 validation set을 평가한 결과입니다.\n")
        
        # 하이퍼파라미터 튜닝 실행
        tune_results = model.tune(
            data=self.config['data'],
            epochs=self.config['epochs'],
            iterations=iterations,
            optimizer='AdamW',
            plots=True,
            save=True,
            val=True,  # validation 수행 및 best.pt로 confusion matrix 생성
            project=os.path.join(self.config['project'], f"{self.run_name}_tune"),
            name='iter',
            # --- [버그 수정] ---
            task=self.config['task']  # OBB 모드('obb') 또는 'detect' 모드 전달
            # ---------------------
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 하이퍼파라미터 튜닝 완료!")
        print(f"{'='*60}")
        print(f"📁 결과 저장 위치: {self.config['project']}/{self.run_name}_tune")
        print(f"📄 최적 하이퍼파라미터는 best_hyperparameters.yaml 파일에 저장되었습니다.")
        
        return tune_results
    
    def train(self, use_tuned_hyperparameters=False, tuned_yaml_path=None, **kwargs):
        """
        모델 학습 시작
        """
        print(f"\n{'='*60}")
        print(f"🚀 YOLOv11 학습 시작")
        print(f"{'='*60}\n")
        
        train_config = {**self.config, **kwargs}
        
        if use_tuned_hyperparameters and tuned_yaml_path:
            if os.path.exists(tuned_yaml_path):
                print(f"🔧 튜닝된 하이퍼파라미터 로드: {tuned_yaml_path}")
                with open(tuned_yaml_path, 'r') as f:
                    tuned_params = yaml.safe_load(f)
                train_config.update(tuned_params)
        
        model_type = train_config['model']
        if self.use_obb and train_config['task'] == 'obb':
            if not model_type.endswith('-obb.pt') and 'yolo' in model_type:
                model_type = model_type.replace('.pt', '-obb.pt')
        
        model = YOLO(model_type)
        
        train_params = {
            'data': train_config['data'],
            'epochs': train_config['epochs'],
            'imgsz': train_config['imgsz'],
            'batch': train_config['batch'],
            'patience': train_config['patience'],
            'name': train_config['name'],
            'project': train_config['project'],
            'device': 0, 
            'workers': 8,
            'save': True,
            'save_period': 10,
            'plots': True,
            'verbose': True,
            'exist_ok': False,
        }
        
        # OBB 모드인 경우 task 파라미터 추가
        if train_config.get('task') == 'obb':
            train_params['task'] = 'obb'
        
        # 튜닝된 파라미터가 있으면 추가
        if use_tuned_hyperparameters and tuned_yaml_path:
            tuned_keys = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                          'warmup_momentum', 'box', 'cls', 'dfl', 'hsv_h', 'hsv_s', 
                          'hsv_v', 'degrees', 'translate', 'scale', 'shear', 
                          'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup']
            for key in tuned_keys:
                if key in train_config:
                    train_params[key] = train_config[key]
        
        results = model.train(**train_params)
        
        save_path = f"{train_config['project']}/{train_config['name']}"
        print(f"\n{'='*60}")
        print(f"✅ 학습 완료!")
        print(f"{'='*60}")
        print(f"📁 결과 저장 위치: {save_path}")
        
        return results
    
    def test_with_best_model(self, best_pt_path, test_data_yaml=None, test_txt=None, save_dir=None, convert_format=False):
        """
        best.pt 모델로 test set에 대해 confusion matrix 생성
        
        Args:
            best_pt_path (str): best.pt 모델 파일 경로
            test_data_yaml (str): test set이 포함된 YAML 파일 경로 (선택)
            test_txt (str): test 이미지 경로가 담긴 txt 파일 (선택)
            save_dir (str): 결과 저장 디렉토리 (기본값: best.pt가 있는 디렉토리)
            convert_format (bool): 라벨 포맷 변환 활성화 (xywha -> xyxyxyxy)
        """
        print(f"\n{'='*60}")
        print(f"🔍 Test Set 평가 시작 (best.pt)")
        print(f"{'='*60}\n")
        
        if not os.path.exists(best_pt_path):
            raise FileNotFoundError(f"best.pt 파일을 찾을 수 없습니다: {best_pt_path}")
        
        # 라벨 포맷 변환이 필요한 경우 수행
        if convert_format and self.use_obb:
            print(f"\n🚀 데이터셋 포맷 변환 모드 활성화 (xywha -> xyxyxyxy)")
            if test_txt:
                self.preprocess_obb_dataset(test_txt, 'test')
            elif test_data_yaml:
                with open(test_data_yaml, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f)
                test_path = data_config.get('test')
                if test_path:
                    self.preprocess_obb_dataset(test_path, 'test')
        
        # 저장 디렉토리 설정
        if save_dir is None:
            save_dir = Path(best_pt_path).parent.parent
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        print(f"📦 모델 로드 중: {best_pt_path}")
        model = YOLO(best_pt_path)
        
        # 학습 시 사용된 원본 클래스 정보 가져오기 (우선순위: args.yaml > test_data_yaml)
        orig_nc = None
        orig_names = None
        
        # 1단계: args.yaml에서 원본 data.yaml 경로 찾기
        args_yaml = Path(best_pt_path).parent.parent / 'args.yaml'
        if args_yaml.exists():
            with open(args_yaml, 'r', encoding='utf-8') as f:
                args_config = yaml.safe_load(f)
                orig_data_yaml_path = args_config.get('data')
                
                if orig_data_yaml_path and os.path.exists(orig_data_yaml_path):
                    print(f"📋 학습 시 사용된 원본 YAML 발견: {orig_data_yaml_path}")
                    with open(orig_data_yaml_path, 'r', encoding='utf-8') as df:
                        orig_data_config = yaml.safe_load(df)
                        # nc 또는 num_classes 처리
                        orig_nc = orig_data_config.get('nc') or orig_data_config.get('num_classes')
                        orig_names = orig_data_config.get('names')
                        if orig_names and isinstance(orig_names, dict):
                            orig_names = list(orig_names.values())
        
        # test 데이터 설정
        if test_data_yaml:
            # YAML 파일 사용
            if not os.path.exists(test_data_yaml):
                raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {test_data_yaml}")
            with open(test_data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            test_path = data_config.get('test')
            if not test_path:
                raise ValueError(f"YAML 파일에 'test' 필드가 없습니다: {test_data_yaml}")
            
            # 클래스 정보 결정 (원본 정보 우선, 없으면 test_data_yaml 사용)
            final_nc = orig_nc if orig_nc is not None else (data_config.get('nc') or data_config.get('num_classes'))
            final_names = orig_names if orig_names is not None else data_config.get('names')
            
            # names가 dict인 경우 list로 변환
            if final_names and isinstance(final_names, dict):
                final_names = list(final_names.values())
            
            # 여전히 없으면 추정
            if final_nc is None:
                if final_names:
                    final_nc = len(final_names)
                else:
                    final_nc = 3  # 기본값
            
            if not final_names:
                final_names = [f'class{i}' for i in range(final_nc)]
            
            print(f"📊 클래스 정보:")
            print(f"  - 클래스 수 (nc): {final_nc}")
            print(f"  - 클래스 이름: {final_names}")
            
            # ultralytics는 'train'과 'val' 필드가 모두 필요하므로, test를 train과 val 모두에 매핑
            # 원본 YAML에서 train 경로가 있으면 사용, 없으면 test 경로 사용
            train_path = data_config.get('train', test_path)
            temp_yaml_path = save_dir / 'temp_test_data.yaml'
            temp_yaml = {
                'train': os.path.abspath(train_path),
                'val': os.path.abspath(test_path),  # test를 val로 매핑
                'nc': final_nc,
                'names': final_names
            }
            with open(temp_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(temp_yaml, f, default_flow_style=False, allow_unicode=True)
            data_yaml_path = str(temp_yaml_path)
        elif test_txt:
            # test_txt만 제공된 경우, 임시 YAML 파일 생성
            if not os.path.exists(test_txt):
                raise FileNotFoundError(f"Test 데이터 파일을 찾을 수 없습니다: {test_txt}")
            
            # 클래스 정보 결정 (원본 정보 우선, 없으면 기본값)
            final_nc = orig_nc if orig_nc is not None else 1
            final_names = orig_names if orig_names is not None else ['class0']
            
            # names가 dict인 경우 list로 변환
            if final_names and isinstance(final_names, dict):
                final_names = list(final_names.values())
            
            if not final_names:
                final_names = [f'class{i}' for i in range(final_nc)]
            
            print(f"📊 클래스 정보:")
            print(f"  - 클래스 수 (nc): {final_nc}")
            print(f"  - 클래스 이름: {final_names}")
            
            # ultralytics는 'train'과 'val' 필드가 모두 필요하므로, test를 train과 val 모두에 매핑
            temp_yaml_path = save_dir / 'temp_test_data.yaml'
            temp_yaml = {
                'train': os.path.abspath(test_txt),  # train 필드도 필요 (test 경로 사용)
                'val': os.path.abspath(test_txt),   # test를 val로 매핑
                'nc': final_nc,
                'names': final_names
            }
            with open(temp_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(temp_yaml, f, default_flow_style=False, allow_unicode=True)
            data_yaml_path = str(temp_yaml_path)
            test_path = test_txt
        else:
            raise ValueError("test_data_yaml 또는 test_txt 중 하나를 제공해야 합니다.")
        
        print(f"📊 Test 데이터: {test_path}")
        print(f"📁 결과 저장 위치: {save_dir}")
        print(f"📄 사용할 YAML 파일: {data_yaml_path}\n")
        
        # Test 평가 수행 (confusion matrix 포함)
        # split 파라미터 제거 - ultralytics가 YAML의 test 필드를 자동으로 사용
        print("🔄 Test set 평가 중...")
        
        # 모델이 OBB 모델인지 확인 (best.pt 파일명이나 모델 구조로 판단)
        # best.pt가 있는 디렉토리에서 args.yaml 찾기 시도
        args_yaml = Path(best_pt_path).parent.parent / 'args.yaml'
        task_type = None
        if args_yaml.exists():
            with open(args_yaml, 'r', encoding='utf-8') as f:
                args_config = yaml.safe_load(f)
                task_type = args_config.get('task', None)
        
        # task가 없으면 모델 파일명이나 use_obb 설정으로 판단
        if task_type is None:
            if self.use_obb or 'obb' in str(best_pt_path).lower():
                task_type = 'obb'
            else:
                task_type = 'detect'
        
        val_params = {
            'data': data_yaml_path,
            'plots': True,
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.001,  # 낮은 confidence threshold로 모든 예측 포함
            'iou': 0.6,
            'device': 0,
            'project': str(save_dir),
            'name': 'test_results',
            'exist_ok': True
        }
        
        # OBB 모드인 경우 task 파라미터 추가
        if task_type == 'obb':
            val_params['task'] = 'obb'
            print(f"  - Task: OBB (회전 박스)")
        else:
            print(f"  - Task: Detect (일반 박스)")
        
        results = model.val(**val_params)
        
        print(f"\n{'='*60}")
        print(f"✅ Test Set 평가 완료!")
        print(f"{'='*60}")
        print(f"📁 결과 저장 위치: {save_dir / 'test_results'}")
        print(f"  - confusion_matrix.png: Test set 혼동행렬")
        print(f"  - results.json: 평가 메트릭")
        
        # 메트릭 출력
        if hasattr(results, 'box'):
            print(f"\n📊 Test Set 메트릭:")
            print(f"  - mAP50: {results.box.map50:.4f}")
            print(f"  - mAP50-95: {results.box.map:.4f}")
            if hasattr(results.box, 'maps'):
                print(f"  - 클래스별 mAP50-95:")
                class_names = self.config.get('classes', [])
                for i, map_val in enumerate(results.box.maps):
                    class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                    print(f"    * {class_name}: {map_val:.4f}")
        
        return results


# 전역 변수: 시그널 핸들러에서 trainer에 접근하기 위해
_global_trainer = None


def signal_handler(signum, frame):
    """시그널 핸들러: 중단 시 라벨 복구"""
    print(f"\n\n⚠️  중단 신호 수신 (Signal {signum}). 라벨 파일 복구 중...")
    if _global_trainer is not None:
        try:
            _global_trainer.restore_original_labels()
        except Exception as e:
            print(f"❌ 라벨 복구 중 오류 발생: {e}")
    print("프로그램을 종료합니다.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 학습 스크립트')
    
    parser.add_argument('--project', type=str, required=False, help='프로젝트 이름 (--test-best 모드에서는 선택)')
    parser.add_argument('--data-yaml', type=str, default=None, help='데이터셋 YAML 파일 경로')
    parser.add_argument('--train', type=str, default=None, help='학습 이미지 경로가 담긴 txt 파일')
    parser.add_argument('--val', type=str, default=None, help='검증 이미지 경로가 담긴 txt 파일')
    parser.add_argument('--test', type=str, default=None, help='테스트 이미지 경로가 담긴 txt 파일')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='클래스 이름 리스트')
    parser.add_argument('--obb', action='store_true', help='회전된 바운딩 박스(OBB) 사용')
    
    parser.add_argument('--convert-format', action='store_true',
                        help='라벨 포맷 변환 활성화: (class x y w h a) -> (class x1 y1 ... x4 y4)')
    
    parser.add_argument('--no-cleanup', action='store_true',
                        help='학습 후 .bak 파일 자동 복원(정리) 비활성화')
    
    parser.add_argument('--model', type=str, default='yolo11s.pt', help='사용할 모델')
    parser.add_argument('--epochs', type=int, default=70, help='학습 에포크 수')
    parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    
    parser.add_argument('--tune', action='store_true', help='학습 전 하이퍼파라미터 튜닝')
    parser.add_argument('--tune-iterations', type=int, default=30, help='튜닝 반복 횟수')
    parser.add_argument('--use-tuned', type=str, default=None, help='튜닝된 하이퍼파라미터 YAML 파일 경로')
    
    parser.add_argument('--test-best', type=str, default=None, help='best.pt 모델 경로 (test set 평가용)')
    parser.add_argument('--test-data-yaml', type=str, default=None, help='test set이 포함된 YAML 파일 경로')
    parser.add_argument('--test-txt', type=str, default=None, help='test 이미지 경로가 담긴 txt 파일')
    parser.add_argument('--test-save-dir', type=str, default=None, help='test 결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 전역 변수 선언 (함수 시작 부분에서 한 번만)
    global _global_trainer
    
    # Test set 평가 모드인 경우 --project가 필요 없음
    if args.test_best:
        # test 모드에서는 간단한 trainer만 생성 (클래스 정보 추출용)
        trainer = YOLOv11Trainer(
            project_name=args.project or 'test_eval',
            use_obb=args.obb,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
        # 시그널 핸들러 등록 (test 모드에서도 라벨 복원 필요)
        if args.convert_format and not args.no_cleanup:
            _global_trainer = trainer
            signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # kill 명령
        
        try:
            trainer.test_with_best_model(
                best_pt_path=args.test_best,
                test_data_yaml=args.test_data_yaml,
                test_txt=args.test_txt,
                save_dir=args.test_save_dir,
                convert_format=args.convert_format
            )
        finally:
            # 스크립트가 성공/실패 여부와 관계없이 항상 라벨 복원 시도
            if args.convert_format and not args.no_cleanup:
                trainer.restore_original_labels()
        
        return
    
    # 일반 학습/튜닝 모드에서는 --project 필수
    if not args.project:
        parser.error("--project는 학습 또는 튜닝 모드에서 필수입니다. (--test-best 모드에서는 선택)")
    
    # 전역 변수에 trainer 저장 (시그널 핸들러에서 접근하기 위해)
    trainer = YOLOv11Trainer(
        project_name=args.project,
        use_obb=args.obb,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    _global_trainer = trainer
    
    # 시그널 핸들러 등록 (Ctrl+C, kill 등)
    if args.convert_format and not args.no_cleanup:
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill 명령
    
    try:
        # 데이터 설정 및 변환 로직
        if args.data_yaml:
            trainer.load_data_yaml(yaml_path=args.data_yaml, convert_format=args.convert_format)
        elif args.train and args.val and args.classes:
            trainer.create_data_yaml(
                train_txt=args.train,
                val_txt=args.val,
                class_names=args.classes,
                test_txt=args.test,
                convert_format=args.convert_format
            )
        else:
            raise ValueError("데이터를 제공해야 합니다 (yaml 또는 txt 파일)")
        
        if args.tune:
            trainer.tune_hyperparameters(iterations=args.tune_iterations)
            return # 튜닝 후 학습은 별도 명령으로 실행
        
        # 학습 시작
        use_tuned = args.use_tuned is not None
        trainer.train(
            use_tuned_hyperparameters=use_tuned,
            tuned_yaml_path=args.use_tuned
        )
    
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류 발생: {e}")
    
    finally:
        # 스크립트가 성공/실패 여부와 관계없이 항상 라벨 복원 시도
        if args.convert_format and not args.no_cleanup:
            trainer.restore_original_labels()
        else:
            if args.convert_format and args.no_cleanup:
                print("\nℹ️  --no-cleanup 플래그가 설정되어 원본 라벨을 복원하지 않습니다.")


if __name__ == "__main__":
    # 예시 1: OBB + 포맷 변환 + 튜닝
    # python yolov11_obb.py --project obb_tune_test --train data/train.txt --val data/val.txt --classes car --obb --convert-format --tune

    # 예시 2: OBB + 포맷 변환 + 학습
    # python yolov11_obb.py --project obb_train_test --train data/train.txt --val data/val.txt --classes car --obb --convert-format
    
    # 예시 3: best.pt로 test set 평가
    # python yolov11_obb.py --test-best runs/project_name/weights/best.pt --test-data-yaml yaml/data.yaml
    # 또는
    # python yolov11_obb.py --test-best runs/project_name/weights/best.pt --test-txt data/test.txt
    
    main()