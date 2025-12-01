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
            'patience': 50,
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
        print(f"  - 저장 위치: {self.config['project']}/{self.run_name}_tune\n")
        
        # 하이퍼파라미터 튜닝 실행
        tune_results = model.tune(
            data=self.config['data'],
            epochs=self.config['epochs'],
            iterations=iterations,
            optimizer='AdamW',
            plots=True,
            save=True,
            val=True,
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
    
    parser.add_argument('--project', type=str, required=True, help='프로젝트 이름')
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
    
    args = parser.parse_args()
    
    # 전역 변수에 trainer 저장 (시그널 핸들러에서 접근하기 위해)
    global _global_trainer
    
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
    # python train.py --project obb_tune_test --train data/train.txt --val data/val.txt --classes car --obb --convert-format --tune
    
    """
    python yolov11_obb.py \
    --project Bolt \
    --data-yaml yaml/Bolt.yaml \
    --obb \
    --convert-format
    """



    # 예시 2: OBB + 포맷 변환 + 학습
    # python train.py --project obb_train_test --train data/train.txt --val data/val.txt --classes car --obb --convert-format
    
    main()