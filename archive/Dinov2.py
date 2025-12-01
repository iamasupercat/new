import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import shutil


class DefectDataset(Dataset):
    """양품/불량 분류 데이터셋"""
    def __init__(self, txt_file, transform=None, label_map=None):
        """
        Args:
            txt_file (str): 이미지 경로와 라벨이 담긴 txt 파일
                           형식: /path/to/image.jpg 0 (또는 1)
            transform: 이미지 변환
        """
        self.data = []
        self.transform = transform
        self.label_map = label_map
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_path = ' '.join(parts[:-1])  # 경로에 공백이 있을 수 있음
                    label = int(parts[-1])
                    self.data.append((img_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 더미 이미지 반환
            image = Image.new('RGB', (224, 224), color='black')
        
        # label mapping (예: door+simple: 1,2,3 -> 1)
        if self.label_map is not None:
            label = self.label_map.get(label, label)

        if self.transform:
            image = self.transform(image)
        
        return image, label


class DINOv2Trainer:
    def __init__(self, project_name='dinov2_training', 
                 model_size='small', imgsz=224, batch_size=32, 
                 epochs=100, lr=1e-4, device='cuda', project_dir='runs',
                 lr_backbone=None, lr_head=None, freeze_epochs=0):
        """
        DINOv2 양품/불량 분류 학습 클래스
        
        Args:
            project_name (str): 프로젝트 이름
            model_size (str): 모델 크기 ('small', 'base', 'large', 'giant')
            imgsz (int): 이미지 크기 (기본값: 224)
            batch_size (int): 배치 크기
            epochs (int): 학습 에포크 수
            lr (float): 학습률
            device (str): 디바이스 ('cuda' 또는 'cpu')
            project_dir (str): 프로젝트 폴더 (기본값: 'runs')
            lr_backbone (float): 백본 학습률
            lr_head (float): 헤드 학습률
            freeze_epochs (int): 백본 고정 에포크 수
        """
        self.project_name = project_name
        self.model_size = model_size
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # 분리 학습률 및 초기 백본 고정 설정
        self.lr_head = lr_head if lr_head is not None else lr
        self.lr_backbone = lr_backbone if lr_backbone is not None else max(lr * 0.1, 1e-6)
        self.freeze_epochs = max(int(freeze_epochs), 0)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{self.project_name}_{self.timestamp}"
        self.save_dir = Path(project_dir) / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 클래스 정보 (YAML의 parts/mode에 따라 설정)
        self.class_names = None
        self.num_classes = None
        self.parts = None
        self.mode = None
        self.label_map = None  # ex) door+simple: {0:0,1:1,2:1,3:1}
        self.preprocess_enabled = True  # YAML의 preprocess on/off
        
        # 모델 초기화
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self._opt_group_head_idx = None
        self._opt_group_backbone_idx = None
        
        # AMP (autocast + GradScaler) 기본 활성화
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"✓ DINOv2 Trainer 초기화 완료")
        print(f"  - 프로젝트: {self.project_name}")
        print(f"  - 모델 크기: {self.model_size}")
        print(f"  - 이미지 크기: {self.imgsz}")
        print(f"  - 배치 크기: {self.batch_size}")
        print(f"  - 에포크: {self.epochs}")
        print(f"  - 학습률: {self.lr}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - 저장 위치: {self.save_dir}")
    
    def _load_dinov2_model(self):
        """DINOv2 백본 로드 및 분류 헤드 추가"""
        print(f"\n🔄 DINOv2 모델 로드 중...")
        
        # DINOv2 모델 매핑
        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        
        model_name = model_map.get(self.model_size, 'dinov2_vits14')
        
        try:
            # torch hub에서 DINOv2 로드
            backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            
            # 특징 차원 가져오기
            if self.model_size == 'small':
                embed_dim = 384
            elif self.model_size == 'base':
                embed_dim = 768
            elif self.model_size == 'large':
                embed_dim = 1024
            elif self.model_size == 'giant':
                embed_dim = 1536
            else:
                embed_dim = 384
            
            # 분류 헤드 추가
            class DINOv2Classifier(nn.Module):
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
                    # DINOv2는 CLS 토큰을 반환
                    features = self.backbone(x)
                    return self.classifier(features)
            
            # num_classes는 YAML 로드 후 설정됨
            num_classes = self.num_classes if self.num_classes is not None else 2
            self.model = DINOv2Classifier(backbone, embed_dim, num_classes=num_classes)
            self.model = self.model.to(self.device)

            # Optimizer 설정: 백본/헤드 파라미터 그룹 분리
            head_params = list(self.model.classifier.parameters())
            backbone_params = list(self.model.backbone.parameters())

            self.optimizer = optim.AdamW([
                {'params': head_params, 'lr': self.lr_head},
                {'params': backbone_params, 'lr': 0.0 if self.freeze_epochs > 0 else self.lr_backbone},
            ], weight_decay=0.01)

            # 파라미터 그룹 인덱스 기록
            self._opt_group_head_idx = 0
            self._opt_group_backbone_idx = 1
            
            print(f"✓ {model_name} 모델 로드 완료 (특징 차원: {embed_dim})")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print(f"💡 torch와 torchvision이 설치되어 있는지 확인하세요.")
            raise
    
    def load_data_yaml(self, yaml_path):
        """
        데이터 YAML 파일 로드
        
        Args:
            yaml_path (str): YAML 파일 경로
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # YAML 파일 검증
        required_keys = ['train', 'val', 'parts']
        for key in required_keys:
            if key not in data_config:
                raise ValueError(f"YAML 파일에 '{key}' 필드가 없습니다: {yaml_path}")
        
        train_txt = data_config['train']
        val_txt = data_config['val']
        test_txt = data_config.get('test', None)
        parts = data_config['parts']
        mode = data_config.get('mode', None)
        preprocess_cfg = data_config.get('preprocess', 'on')

        # parts에 따른 클래스 구성
        if isinstance(parts, str):
            parts_lower = parts.strip().lower()
        else:
            raise ValueError("YAML의 'parts' 값은 문자열이어야 합니다. 예: parts: frontdoor 또는 parts: bolt")

        self.parts = parts_lower
        self.mode = mode.strip().lower() if isinstance(mode, str) else None

        # preprocess on/off 해석
        if isinstance(preprocess_cfg, str):
            preprocess_flag = preprocess_cfg.strip().lower() in ['on', 'true', '1', 'yes']
        elif isinstance(preprocess_cfg, bool):
            preprocess_flag = preprocess_cfg
        elif isinstance(preprocess_cfg, int):
            preprocess_flag = preprocess_cfg != 0
        else:
            preprocess_flag = True
        self.preprocess_enabled = preprocess_flag

        if parts_lower in ['frontdoor', 'door']:
            if self.mode == 'simple':
                # 1,2,3 -> 1 로 매핑하여 2-class 학습
                self.class_names = ['good', 'defect']
                self.label_map = {0: 0, 1: 1, 2: 1, 3: 1}
            else:
                self.class_names = ['good', 'no sealing', 'sealing differs', 'tape sealing']
                self.label_map = None
        elif parts_lower == 'bolt':
            self.class_names = ['good', 'bad']
            self.label_map = None
        else:
            raise ValueError("'parts' 값이 유효하지 않습니다. 'frontdoor' 또는 'bolt' 중 하나여야 합니다.")

        self.num_classes = len(self.class_names)
        
        print(f"\n✓ 데이터 YAML 파일 로드: {yaml_path}")
        print(f"  - 학습 데이터: {train_txt}")
        print(f"  - 검증 데이터: {val_txt}")
        if test_txt:
            print(f"  - 테스트 데이터: {test_txt}")
        print(f"  - parts: {parts} (클래스 수: {self.num_classes})")
        if self.mode:
            print(f"  - mode: {self.mode}")
        print(f"  - preprocess: {'on' if self.preprocess_enabled else 'off'}")
        
        # 데이터셋 통계
        with open(train_txt, 'r') as f:
            train_data = [line.strip() for line in f if line.strip()]
        with open(val_txt, 'r') as f:
            val_data = [line.strip() for line in f if line.strip()]
        
        # 라벨 유효성 검사 및 클래스별 개수 계산
        def parse_label(line):
            try:
                return int(line.split()[-1])
            except Exception:
                raise ValueError(f"잘못된 라벨 형식: '{line}'. 각 줄은 '이미지경로 라벨' 이어야 합니다.")

        train_labels = [parse_label(line) for line in train_data]
        val_labels = [parse_label(line) for line in val_data]

        # door/frontdoor 원천 라벨은 0~3, bolt는 0~1 범위 검사
        if self.parts in ['frontdoor', 'door']:
            max_allowed = 3
        elif self.parts == 'bolt':
            max_allowed = 1
        else:
            max_allowed = self.num_classes - 1

        invalid_train = [l for l in train_labels if l < 0 or l > max_allowed]
        invalid_val = [l for l in val_labels if l < 0 or l > max_allowed]
        if invalid_train or invalid_val:
            raise ValueError(
                f"라벨 값이 'parts={parts}'의 원천 라벨 범위를 벗어났습니다. "
                f"허용 범위: 0~{max_allowed}, "
                f"잘못된 학습 라벨 예: {invalid_train[:5]}, 잘못된 검증 라벨 예: {invalid_val[:5]}"
            )

        # mode=simple이면 매핑 후 통계를 최종 클래스 기준으로 산출
        if self.label_map is not None:
            mapped_train = [self.label_map.get(l, l) for l in train_labels]
            mapped_val = [self.label_map.get(l, l) for l in val_labels]
            train_counts = [sum(1 for l in mapped_train if l == cid) for cid in range(self.num_classes)]
            val_counts = [sum(1 for l in mapped_val if l == cid) for cid in range(self.num_classes)]
        else:
            train_counts = [sum(1 for l in train_labels if l == cid) for cid in range(self.num_classes)]
            val_counts = [sum(1 for l in val_labels if l == cid) for cid in range(self.num_classes)]
        
        print(f"\n📊 데이터셋 통계:")
        print(f"  - 학습 이미지: {len(train_data)}개")
        print(f"  - 검증 이미지: {len(val_data)}개")
        for cid, name in enumerate(self.class_names):
            print(f"    * {cid} - {name}: train {train_counts[cid]} / val {val_counts[cid]}")
        
        return train_txt, val_txt, test_txt
    
    def _get_transforms(self, is_train=True):
        """데이터 augmentation 및 전처리"""
        if not self.preprocess_enabled:
            return transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.ToTensor(),
            ])
        if is_train:
            return transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def train(self, train_txt, val_txt, test_txt=None):
        """
        모델 학습
        
        Args:
            train_txt (str): 학습 데이터 txt 파일 경로
            val_txt (str): 검증 데이터 txt 파일 경로
            test_txt (str): 테스트 데이터 txt 파일 경로 (선택)
        """
        print(f"\n{'='*60}")
        print(f"🚀 DINOv2 학습 시작")
        print(f"{'='*60}\n")
        
        # 모델 로드
        self._load_dinov2_model()
        
        # 데이터셋 준비
        train_dataset = DefectDataset(
            train_txt,
            transform=self._get_transforms(is_train=True),
            label_map=self.label_map,
        )
        val_dataset = DefectDataset(
            val_txt,
            transform=self._get_transforms(is_train=False),
            label_map=self.label_map,
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"📊 데이터로더 준비 완료:")
        print(f"  - 학습 배치 수: {len(train_loader)}")
        print(f"  - 검증 배치 수: {len(val_loader)}\n")
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        # 학습 루프
        last_val_true = []
        last_val_pred = []

        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{self.epochs}]")
            print(f"{'='*60}")

            # Epoch 시작 시 freeze/unfreeze 적용
            if epoch < self.freeze_epochs:
                # 백본 고정, 백본 LR=0
                for p in self.model.backbone.parameters():
                    p.requires_grad = False
                if self._opt_group_backbone_idx is not None:
                    self.optimizer.param_groups[self._opt_group_backbone_idx]['lr'] = 0.0
            else:
                # 백본 해제, 백본 LR 복원
                for p in self.model.backbone.parameters():
                    p.requires_grad = True
                if self._opt_group_backbone_idx is not None:
                    self.optimizer.param_groups[self._opt_group_backbone_idx]['lr'] = self.lr_backbone
            
            # 학습 단계
            train_loss, train_acc = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 검증 단계
            val_loss, val_acc, val_true, val_pred = self._validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            last_val_true = val_true
            last_val_pred = val_pred
            
            print(f"\n📊 Epoch {epoch+1} 결과:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Best 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint('best.pt', epoch, val_acc)
                print(f"  ✓ Best 모델 저장! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Last 모델 저장 (10 에포크마다)
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint('last.pt', epoch, val_acc)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (patience: {patience})")
                break
        
        # 최종 모델 저장
        self._save_checkpoint('last.pt', epoch, val_acc)
        
        # 학습 결과 저장
        self._save_results()
        self._plot_training_curves()

        # 혼동행렬 저장 (마지막 검증 기준)
        try:
            if last_val_true and last_val_pred and self.class_names:
                self._plot_confusion_matrix(last_val_true, last_val_pred)
        except Exception as e:
            print(f"혼동행렬 생성 중 오류: {e}")
        
        print(f"\n{'='*60}")
        print(f"✅ 학습 완료!")
        print(f"{'='*60}")
        print(f"📁 결과 저장 위치: {self.save_dir}")
        print(f"  - weights/best.pt: 최적 모델 (Val Acc: {best_val_acc:.2f}%)")
        print(f"  - weights/last.pt: 마지막 모델")
        print(f"  - results.png: 학습 곡선")
        print(f"  - metrics.json: 학습 메트릭")
        
        # 테스트 데이터가 있으면 예측 및 이미지 분류
        if test_txt and os.path.exists(test_txt):
            print(f"\n{'='*60}")
            print(f"🔍 테스트 데이터 예측 및 이미지 분류 시작")
            print(f"{'='*60}\n")
            self._test_and_classify_images(test_txt)
    
    def _train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward with AMP
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward with GradScaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 통계
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def _validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_true.extend(labels.detach().cpu().tolist())
                all_pred.extend(predicted.detach().cpu().tolist())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, all_true, all_pred
    
    def _save_checkpoint(self, filename, epoch, val_acc):
        """체크포인트 저장"""
        weights_dir = self.save_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': {
                'project_name': self.project_name,
                'model_size': self.model_size,
                'imgsz': self.imgsz,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'lr_head': self.lr_head,
                'lr_backbone': self.lr_backbone,
                'freeze_epochs': self.freeze_epochs,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'parts': self.parts,
                'mode': self.mode,
                'preprocess': self.preprocess_enabled,
            }
        }
        
        torch.save(checkpoint, weights_dir / filename)
    
    def _save_results(self):
        """학습 결과 저장"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': max(self.val_accs) if self.val_accs else 0,
            'config': {
                'project_name': self.project_name,
                'model_size': self.model_size,
                'imgsz': self.imgsz,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'lr': self.lr,
                'lr_head': self.lr_head,
                'lr_backbone': self.lr_backbone,
                'freeze_epochs': self.freeze_epochs,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'parts': self.parts,
                'mode': self.mode,
                'preprocess': self.preprocess_enabled,
            }
        }
        
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def _plot_training_curves(self):
        """학습 곡선 그리기"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss 곡선
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy 곡선
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'results.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred):
        """혼동행렬 히트맵 저장"""
        num_classes = self.num_classes if self.num_classes is not None else (max(max(y_true), max(y_pred)) + 1)

        # 혼동행렬 계산
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t][p] += 1

        fig, ax = plt.subplots(figsize=(6 + num_classes * 0.4, 5 + num_classes * 0.3))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 축/라벨
        class_names = self.class_names if self.class_names else [str(i) for i in range(num_classes)]
        ax.set(xticks=range(num_classes), yticks=range(num_classes),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label', xlabel='Predicted label',
               title='Confusion Matrix (Validation)')

        # 라벨 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # 셀 내 숫자 주석
        thresh = max(max(row) for row in cm) * 0.5 if cm and cm[0] else 0
        for i in range(num_classes):
            for j in range(num_classes):
                value = cm[i][j]
                ax.text(j, i, str(value), ha='center', va='center',
                        color='white' if value > thresh else 'black')

        plt.tight_layout()
        out_path = self.save_dir / 'confusion_matrix.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 간단한 정답 개수 출력
        total_correct = sum(cm[i][i] for i in range(num_classes))
        total_samples = len(y_true)
        print(f"혼동행렬 저장: {out_path}")
        print(f"검증 정답 개수: {total_correct} / {total_samples} ({(100.0*total_correct/total_samples if total_samples else 0):.2f}%)")

    def _test_and_classify_images(self, test_txt):
        """
        테스트 데이터 예측 및 correct/incorrect 폴더에 이미지 분류
        
        Args:
            test_txt (str): 테스트 이미지 경로와 라벨이 담긴 txt 파일
        """
        # 폴더 생성
        correct_dir = self.save_dir / 'test_results' / 'correct'
        incorrect_dir = self.save_dir / 'test_results' / 'incorrect'
        correct_dir.mkdir(parents=True, exist_ok=True)
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 준비
        test_dataset = DefectDataset(
            test_txt,
            transform=self._get_transforms(is_train=False),
            label_map=self.label_map,
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"📊 테스트 데이터로더 준비 완료:")
        print(f"  - 테스트 배치 수: {len(test_loader)}")
        print(f"  - 테스트 이미지 수: {len(test_dataset)}\n")
        
        self.model.eval()
        
        correct_count = 0
        incorrect_count = 0
        
        # 원본 이미지 경로와 라벨 가져오기
        with open(test_txt, 'r') as f:
            test_lines = [line.strip() for line in f if line.strip()]
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            batch_idx = 0
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                    outputs = self.model(images)
                
                _, predicted = outputs.max(1)
                
                # 배치 내 각 이미지 처리
                for i in range(images.size(0)):
                    img_idx = batch_idx * self.batch_size + i
                    if img_idx >= len(test_lines):
                        break
                    
                    # 원본 이미지 경로 추출
                    line_parts = test_lines[img_idx].split()
                    img_path = ' '.join(line_parts[:-1])
                    
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    confidence = torch.softmax(outputs[i], dim=0)
                    
                    all_predictions.append(pred_label)
                    all_labels.append(true_label)
                    
                    # 이미지 복사
                    if os.path.exists(img_path):
                        img_filename = os.path.basename(img_path)
                        
                        # 파일명에 예측 정보 추가
                        name_parts = os.path.splitext(img_filename)
                        class_name_true = self.class_names[true_label] if true_label < len(self.class_names) else str(true_label)
                        class_name_pred = self.class_names[pred_label] if pred_label < len(self.class_names) else str(pred_label)
                        new_filename = f"{name_parts[0]}_gt{true_label}({class_name_true})_pred{pred_label}({class_name_pred})_conf{confidence[pred_label]:.2f}{name_parts[1]}"
                        
                        if true_label == pred_label:
                            dest_path = correct_dir / new_filename
                            shutil.copy2(img_path, dest_path)
                            correct_count += 1
                        else:
                            dest_path = incorrect_dir / new_filename
                            shutil.copy2(img_path, dest_path)
                            incorrect_count += 1
                
                batch_idx += 1
                pbar.set_postfix({
                    'correct': correct_count,
                    'incorrect': incorrect_count,
                    'acc': f'{100.*correct_count/(correct_count+incorrect_count):.2f}%' if (correct_count+incorrect_count) > 0 else '0%'
                })
        
        # 테스트 결과 통계
        total = correct_count + incorrect_count
        accuracy = 100. * correct_count / total if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 테스트 결과 통계")
        print(f"{'='*60}")
        print(f"  - 총 이미지: {total}")
        print(f"  - 정답: {correct_count}")
        print(f"  - 오답: {incorrect_count}")
        print(f"  - 정확도: {accuracy:.2f}%")
        print(f"\n📁 분류된 이미지 저장 위치:")
        print(f"  - Correct: {correct_dir}")
        print(f"  - Incorrect: {incorrect_dir}")
        
        # 테스트 혼동행렬 생성
        if len(all_labels) > 0:
            self._plot_test_confusion_matrix(all_labels, all_predictions)
        
        # 테스트 결과 JSON 저장
        test_results = {
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'accuracy': accuracy,
            'class_names': self.class_names
        }
        
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)
        
        print(f"✓ 테스트 결과 저장: {self.save_dir / 'test_results.json'}")

    def _plot_test_confusion_matrix(self, y_true, y_pred):
        """테스트 데이터 혼동행렬 저장"""
        num_classes = self.num_classes if self.num_classes is not None else (max(max(y_true), max(y_pred)) + 1)

        # 혼동행렬 계산
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t][p] += 1

        fig, ax = plt.subplots(figsize=(6 + num_classes * 0.4, 5 + num_classes * 0.3))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 축/라벨
        class_names = self.class_names if self.class_names else [str(i) for i in range(num_classes)]
        ax.set(xticks=range(num_classes), yticks=range(num_classes),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label', xlabel='Predicted label',
               title='Confusion Matrix (Test)')

        # 라벨 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # 셀 내 숫자 주석
        thresh = max(max(row) for row in cm) * 0.5 if cm and cm[0] else 0
        for i in range(num_classes):
            for j in range(num_classes):
                value = cm[i][j]
                ax.text(j, i, str(value), ha='center', va='center',
                        color='white' if value > thresh else 'black')

        plt.tight_layout()
        out_path = self.save_dir / 'test_confusion_matrix.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 테스트 혼동행렬 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='DINOv2 양품/불량 분류 학습 스크립트')
    
    parser.add_argument('--project', type=str, required=True,
                        help='프로젝트 이름')
    parser.add_argument('--data-yaml', type=str, required=True,
                        help='데이터셋 YAML 파일 경로')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large', 'giant'],
                        help='DINOv2 모델 크기 (기본값: small)')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='이미지 크기 (기본값: 224)')
    parser.add_argument('--batch', type=int, default=32,
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수 (기본값: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률 (기본값: 1e-4)')
    parser.add_argument('--lr-backbone', type=float, default=None,
                        help='백본 학습률 (기본은 lr의 0.1배)')
    parser.add_argument('--lr-head', type=float, default=None,
                        help='분류기 헤드 학습률 (기본은 lr)')
    parser.add_argument('--freeze-epochs', type=int, default=0,
                        help='초기 백본 고정 에포크 수 (기본 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='디바이스 (기본값: cuda)')
    
    args = parser.parse_args()
    
    # 트레이너 초기화
    trainer = DINOv2Trainer(
        project_name=args.project,
        model_size=args.model_size,
        imgsz=args.imgsz,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        freeze_epochs=args.freeze_epochs
    )
    
    # 데이터 로드
    train_txt, val_txt, test_txt = trainer.load_data_yaml(args.data_yaml)
    
    # 학습 시작
    trainer.train(train_txt, val_txt, test_txt)


if __name__ == "__main__":
    # 예시 1: 기본 학습 (test 데이터 있는 경우)
    # python train_dinov2.py --project defect_detection --data-yaml data/defect.yaml
    
    # 예시 2: Large 모델 사용
    # python train_dinov2.py --project defect_detection --data-yaml data/defect.yaml --model-size large
    
    # 예시 3: 커스텀 설정
    # python train_dinov2.py --project defect_detection --data-yaml data/defect.yaml --model-size base --imgsz 384 --batch 16 --epochs 200 --lr 5e-5
    
    # 예시 4: 코드에서 직접 실행
    # trainer = DINOv2Trainer(project_name='defect_detection', model_size='base', epochs=150)
    # train_txt, val_txt, test_txt = trainer.load_data_yaml('data/defect.yaml')
    # trainer.train(train_txt, val_txt, test_txt)
    
    main()