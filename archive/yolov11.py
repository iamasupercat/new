import argparse
from ultralytics import YOLO
import yaml
import os
from pathlib import Path
from datetime import datetime


class YOLOv11Trainer:
    def __init__(self, project_name='yolo_training', use_obb=False, 
                 model='yolo11s.pt', epochs=100, imgsz=640, batch=16):
        """
        YOLOv11 학습 클래스
        
        Args:
            project_name (str): 프로젝트 이름
            use_obb (bool): 회전된 바운딩 박스(OBB) 사용 여부 (기본값: False)
            model (str): 사용할 모델 (기본값: yolo11n.pt)
            epochs (int): 학습 에포크 수 (기본값: 100)
            imgsz (int): 이미지 크기 (기본값: 640)
            batch (int): 배치 크기 (기본값: 16)
        """
        self.project_name = project_name
        self.use_obb = use_obb
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{self.project_name}_{self.timestamp}"
        
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
            'classes': []  # load_data_yaml 또는 create_data_yaml에서 설정
        }
    
    def load_data_yaml(self, yaml_path):
        """
        기존 데이터 YAML 파일 로드
        
        Args:
            yaml_path (str): 데이터 YAML 파일 경로
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # YAML 파일 검증
        required_keys = ['train', 'val', 'names']
        for key in required_keys:
            if key not in data_config:
                raise ValueError(f"YAML 파일에 '{key}' 필드가 없습니다: {yaml_path}")
        
        # config 업데이트
        self.config['data'] = yaml_path
        if isinstance(data_config['names'], dict):
            self.config['classes'] = list(data_config['names'].values())
        else:
            self.config['classes'] = data_config['names']
        
        print(f"✓ 데이터 YAML 파일 로드: {yaml_path}")
        print(f"  - 학습 데이터: {data_config['train']}")
        print(f"  - 검증 데이터: {data_config['val']}")
        if 'test' in data_config:
            print(f"  - 테스트 데이터: {data_config['test']}")
        
        # txt 파일 존재 여부 및 이미지 개수 확인
        train_path = data_config['train']
        val_path = data_config['val']
        
        if os.path.exists(train_path):
            with open(train_path, 'r') as f:
                train_count = len([line for line in f if line.strip()])
        else:
            train_count = "Unknown (파일 없음)"
        
        if os.path.exists(val_path):
            with open(val_path, 'r') as f:
                val_count = len([line for line in f if line.strip()])
        else:
            val_count = "Unknown (파일 없음)"
        
        print(f"\n📊 데이터셋 통계:")
        print(f"  - 클래스 개수: {len(self.config['classes'])}개")
        print(f"  - 클래스: {', '.join(self.config['classes'])}")
        print(f"  - 학습 이미지: {train_count}개" if isinstance(train_count, int) else f"  - 학습 이미지: {train_count}")
        print(f"  - 검증 이미지: {val_count}개" if isinstance(val_count, int) else f"  - 검증 이미지: {val_count}")
        
        return yaml_path
    
    def create_data_yaml(self, train_txt, val_txt, class_names, test_txt=None):
        """
        데이터셋 YAML 파일 생성 (txt 파일 경로 포함)
        
        Args:
            train_txt (str): 학습 이미지 경로가 담긴 txt 파일
            val_txt (str): 검증 이미지 경로가 담긴 txt 파일
            class_names (list): 클래스 이름 리스트 (예: ['high', 'mid', 'low'])
            test_txt (str): 테스트 이미지 경로가 담긴 txt 파일 (선택)
        """
        # txt 파일 존재 여부 확인
        if not os.path.exists(train_txt):
            raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {train_txt}")
        if not os.path.exists(val_txt):
            raise FileNotFoundError(f"검증 데이터 파일을 찾을 수 없습니다: {val_txt}")
        
        # 클래스 정보 업데이트
        self.config['classes'] = class_names
        
        # 데이터 YAML 구성 - txt 파일 경로를 직접 지정
        data_yaml = {
            'train': os.path.abspath(train_txt),  # txt 파일 경로
            'val': os.path.abspath(val_txt),      # txt 파일 경로
            'nc': len(class_names),               # 클래스 개수
            'names': class_names                  # 클래스 이름 리스트
        }
        
        if test_txt and os.path.exists(test_txt):
            data_yaml['test'] = os.path.abspath(test_txt)
        
        yaml_path = self.config['data']
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 데이터 YAML 파일 생성: {yaml_path}")
        print(f"  - 학습 데이터: {train_txt}")
        print(f"  - 검증 데이터: {val_txt}")
        if test_txt:
            print(f"  - 테스트 데이터: {test_txt}")
        
        # txt 파일의 이미지 개수 확인
        with open(train_txt, 'r') as f:
            train_count = len([line for line in f if line.strip()])
        with open(val_txt, 'r') as f:
            val_count = len([line for line in f if line.strip()])
        
        print(f"\n📊 데이터셋 통계:")
        print(f"  - 클래스 개수: {len(class_names)}개")
        print(f"  - 클래스: {', '.join(class_names)}")
        print(f"  - 학습 이미지: {train_count}개")
        print(f"  - 검증 이미지: {val_count}개")
        
        return yaml_path
    
    def tune_hyperparameters(self, iterations=30):
        """
        하이퍼파라미터 자동 튜닝
        
        Args:
            iterations (int): 튜닝 반복 횟수 (기본값: 30)
        """
        print(f"\n{'='*60}")
        print(f"🔧 하이퍼파라미터 자동 튜닝 시작")
        print(f"{'='*60}\n")
        
        # 모델 로드
        model_type = self.config['model']
        if self.use_obb and self.config['task'] == 'obb':
            if not model_type.endswith('-obb.pt'):
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
            name='iter'  # 각 iteration을 iter1, iter2, ... 로 저장
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
        
        Args:
            use_tuned_hyperparameters (bool): 튜닝된 하이퍼파라미터 사용 여부
            tuned_yaml_path (str): 튜닝된 하이퍼파라미터 YAML 파일 경로
            **kwargs: 추가 학습 파라미터 (config 설정 덮어쓰기 가능)
        """
        print(f"\n{'='*60}")
        print(f"🚀 YOLOv11 학습 시작")
        print(f"{'='*60}\n")
        
        # 설정 병합
        train_config = {**self.config, **kwargs}
        
        # 튜닝된 하이퍼파라미터 로드
        if use_tuned_hyperparameters and tuned_yaml_path:
            if os.path.exists(tuned_yaml_path):
                print(f"🔧 튜닝된 하이퍼파라미터 로드: {tuned_yaml_path}")
                with open(tuned_yaml_path, 'r') as f:
                    tuned_params = yaml.safe_load(f)
                train_config.update(tuned_params)
            else:
                print(f"⚠️  튜닝 파일을 찾을 수 없습니다: {tuned_yaml_path}")
                print(f"⚠️  기본 하이퍼파라미터로 학습을 진행합니다.")
        
        # 모델 로드
        model_type = train_config['model']
        if self.use_obb and train_config['task'] == 'obb':
            if not model_type.endswith('-obb.pt'):
                model_type = model_type.replace('.pt', '-obb.pt')
        
        model = YOLO(model_type)
        
        print(f"📊 학습 설정:")
        print(f"  - 프로젝트: {self.project_name}")
        print(f"  - 모델: {model_type}")
        print(f"  - 태스크: {train_config.get('task', 'detect')} ({'회전 박스' if self.use_obb else '일반 박스'})")
        print(f"  - 데이터: {train_config['data']}")
        print(f"  - 에포크: {train_config['epochs']}")
        print(f"  - 이미지 크기: {train_config['imgsz']}")
        print(f"  - 배치 크기: {train_config['batch']}")
        print(f"  - 클래스: {', '.join(train_config['classes'])}")
        print(f"  - 저장 위치: {train_config['project']}/{train_config['name']}\n")
        
        # 학습 파라미터 준비
        train_params = {
            'data': train_config['data'],
            'epochs': train_config['epochs'],
            'imgsz': train_config['imgsz'],
            'batch': train_config['batch'],
            'patience': train_config['patience'],
            'name': train_config['name'],
            'project': train_config['project'],
            'device': 0,  # GPU 사용 (CPU는 'cpu')
            'workers': 8,
            'save': True,
            'save_period': 10,  # 10 에포크마다 체크포인트 저장
            'plots': True,
            'verbose': True,
            'exist_ok': False,  # 기존 폴더 덮어쓰기 방지
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
        
        # 학습 시작
        results = model.train(**train_params)
        
        save_path = f"{train_config['project']}/{train_config['name']}"
        print(f"\n{'='*60}")
        print(f"✅ 학습 완료!")
        print(f"{'='*60}")
        print(f"📁 결과 저장 위치: {save_path}")
        print(f"\n📂 저장된 파일:")
        print(f"  - weights/best.pt: 최적 모델")
        print(f"  - weights/last.pt: 마지막 모델")
        print(f"  - results.png: 학습 결과 그래프")
        print(f"  - confusion_matrix.png: 혼동 행렬")
        print(f"  - val_batch*_pred.jpg: 검증 결과 시각화")
        
        return results
    
    def validate(self, model_path=None):
        """모델 검증"""
        if model_path is None:
            model_path = f"{self.config['project']}/{self.config['name']}/weights/best.pt"
        
        print(f"\n🔍 모델 검증 중: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        model = YOLO(model_path)
        results = model.val(data=self.config['data'])
        
        print(f"\n✅ 검증 완료!")
        return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 학습 스크립트')
    
    # 필수 인자: 프로젝트 이름
    parser.add_argument('--project', type=str, required=True,
                        help='프로젝트 이름')
    
    # 데이터 입력 방식 1: YAML 파일
    parser.add_argument('--data-yaml', type=str, default=None,
                        help='데이터셋 YAML 파일 경로 (train, val, names 포함)')
    
    # 데이터 입력 방식 2: txt 파일들
    parser.add_argument('--train', type=str, default=None,
                        help='학습 이미지 경로가 담긴 txt 파일')
    parser.add_argument('--val', type=str, default=None,
                        help='검증 이미지 경로가 담긴 txt 파일')
    parser.add_argument('--test', type=str, default=None,
                        help='테스트 이미지 경로가 담긴 txt 파일 (선택)')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='클래스 이름 리스트 (예: --classes high mid low)')
    
    # OBB (회전된 바운딩 박스) 옵션
    parser.add_argument('--obb', action='store_true',
                        help='회전된 바운딩 박스(OBB) 사용 (기본값: 일반 박스)')
    
    # 학습 설정
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='사용할 모델 (기본값: yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수 (기본값: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='배치 크기 (기본값: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='이미지 크기 (기본값: 640)')
    
    # 하이퍼파라미터 튜닝
    parser.add_argument('--tune', action='store_true',
                        help='학습 전 하이퍼파라미터 튜닝 실행')
    parser.add_argument('--tune-iterations', type=int, default=30,
                        help='튜닝 반복 횟수 (기본값: 30)')
    parser.add_argument('--use-tuned', type=str, default=None,
                        help='튜닝된 하이퍼파라미터 YAML 파일 경로')
    
    args = parser.parse_args()
    
    # 트레이너 초기화
    trainer = YOLOv11Trainer(
        project_name=args.project,
        use_obb=args.obb,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    
    # 데이터 설정: YAML 파일 또는 txt 파일들
    if args.data_yaml:
        # 방식 1: 기존 YAML 파일 사용
        trainer.load_data_yaml(yaml_path=args.data_yaml)
    elif args.train and args.val and args.classes:
        # 방식 2: txt 파일들로 YAML 생성
        trainer.create_data_yaml(
            train_txt=args.train,
            val_txt=args.val,
            class_names=args.classes,
            test_txt=args.test
        )
    else:
        raise ValueError(
            "데이터를 제공해야 합니다:\n"
            "  방식 1: --data-yaml <yaml_file>\n"
            "  방식 2: --train <train.txt> --val <val.txt> --classes <class1> <class2> ..."
        )
    
    # 하이퍼파라미터 튜닝 (옵션)
    if args.tune:
        trainer.tune_hyperparameters(iterations=args.tune_iterations)
        print(f"\n💡 튜닝이 완료되었습니다!")
        print(f"💡 튜닝된 하이퍼파라미터로 학습하려면 다음 명령어를 사용하세요:")
        print(f"   --use-tuned runs/{trainer.run_name}_tune/best_hyperparameters.yaml\n")
        return
    
    # 학습 시작
    use_tuned = args.use_tuned is not None
    trainer.train(
        use_tuned_hyperparameters=use_tuned,
        tuned_yaml_path=args.use_tuned
    )


if __name__ == "__main__":
    # 예시 1: YAML 파일로 학습
    # python train_yolo11.py --project door_detection --data-yaml data/door_dataset.yaml
    
    # 예시 2: txt 파일 + 클래스 지정
    # python train_yolo11.py --project door_detection --train train.txt --val val.txt --classes high mid low
    
    # 예시 3: 회전된 바운딩 박스 사용
    # python train_yolo11.py --project door_detection --data-yaml data/door_dataset.yaml --obb
    
    # 예시 4: 커스텀 모델 + 에포크 + 배치
    # python train_yolo11.py --project bolt_detection --data-yaml data/bolt.yaml --model yolo11m.pt --epochs 200 --batch 32
    
    # 예시 5: 하이퍼파라미터 튜닝
    # python train_yolo11.py --project door_detection --data-yaml data/door.yaml --obb --tune --tune-iterations 50
    
    # 예시 6: 튜닝된 파라미터로 학습
    # python train_yolo11.py --project door_detection --data-yaml data/door.yaml --obb --use-tuned runs/door_detection_xxx_tune/best_hyperparameters.yaml
    
    # 예시 7: 코드에서 직접 실행
    # trainer = YOLOv11Trainer(project_name='my_project', use_obb=True, model='yolo11s.pt', epochs=150)
    # trainer.load_data_yaml('data/dataset.yaml')
    # trainer.tune_hyperparameters(iterations=50)
    # trainer.train(use_tuned_hyperparameters=True, tuned_yaml_path='runs/my_project_xxx_tune/best_hyperparameters.yaml')
    
    main()