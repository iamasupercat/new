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
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import torch.nn.functional as F
import plotly.express as px
import pandas as pd


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
                    img_path = ' '.join(parts[:-1])
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
            image = Image.new('RGB', (224, 224), color='black')
        
        # Label Mapping (예: 1,2,3 -> 1)
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
        
        self.project_name = project_name
        self.model_size = model_size
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_head = lr_head if lr_head is not None else lr
        self.lr_backbone = lr_backbone if lr_backbone is not None else max(lr * 0.1, 1e-6)
        self.freeze_epochs = max(int(freeze_epochs), 0)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{self.project_name}_{self.timestamp}"
        self.save_dir = Path(project_dir) / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = None
        self.num_classes = None
        self.parts = None
        self.mode = None
        self.label_map = None
        self.preprocess_enabled = True
        
        self.model = None
        self.criterion = nn.CrossEntropyLoss() # 초기엔 기본값, 나중에 가중치 적용
        self.optimizer = None
        self._opt_group_head_idx = None
        self._opt_group_backbone_idx = None
        
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"✓ DINOv2 Trainer 초기화 완료")
        print(f"  - 저장 위치: {self.save_dir}")
    
    def _load_dinov2_model(self):
        """DINOv2 백본 로드 및 분류 헤드 추가"""
        print(f"\n🔄 DINOv2 모델 로드 중 ({self.model_size})...")
        
        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        model_name = model_map.get(self.model_size, 'dinov2_vits14')
        
        try:
            backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            
            if self.model_size == 'small': embed_dim = 384
            elif self.model_size == 'base': embed_dim = 768
            elif self.model_size == 'large': embed_dim = 1024
            elif self.model_size == 'giant': embed_dim = 1536
            else: embed_dim = 384
            
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
                    features = self.backbone(x)
                    return self.classifier(features)
            
            num_classes = self.num_classes if self.num_classes is not None else 2
            self.model = DINOv2Classifier(backbone, embed_dim, num_classes=num_classes)
            self.model = self.model.to(self.device)

            head_params = list(self.model.classifier.parameters())
            backbone_params = list(self.model.backbone.parameters())

            self.optimizer = optim.AdamW([
                {'params': head_params, 'lr': self.lr_head},
                {'params': backbone_params, 'lr': 0.0 if self.freeze_epochs > 0 else self.lr_backbone},
            ], weight_decay=0.01)

            self._opt_group_head_idx = 0
            self._opt_group_backbone_idx = 1
            
            print(f"✓ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def load_data_yaml(self, yaml_path):
        """데이터 YAML 파일 로드 및 Class Weight 계산"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        train_txt = data_config['train']
        val_txt = data_config['val']
        test_txt = data_config.get('test', None)
        parts = data_config['parts']
        mode = data_config.get('mode', None)
        preprocess_cfg = data_config.get('preprocess', 'on')

        self.parts = parts.strip().lower() if isinstance(parts, str) else 'frontdoor'
        self.mode = mode.strip().lower() if isinstance(mode, str) else None
        self.preprocess_enabled = str(preprocess_cfg).lower() in ['on', 'true', '1', 'yes']

        if self.parts in ['frontdoor', 'door']:
            if self.mode == 'simple':
                self.class_names = ['good', 'defect']
                self.label_map = {0: 0, 1: 1, 2: 1, 3: 1}
            else:
                self.class_names = ['good', 'no sealing', 'sealing differs', 'tape sealing']
                self.label_map = None
        elif self.parts == 'bolt':
            self.class_names = ['good', 'bad']
            self.label_map = None
        else: # Default fallback
             self.class_names = ['good', 'bad']
             self.label_map = None

        self.num_classes = len(self.class_names)
        
        # 통계 계산을 위한 데이터 로드
        with open(train_txt, 'r') as f:
            train_lines = [line.strip() for line in f if line.strip()]
        
        def get_label(line):
            return int(line.split()[-1])

        train_labels_raw = [get_label(line) for line in train_lines]
        
        # Label Mapping 적용 후 카운트
        if self.label_map:
            final_train_labels = [self.label_map.get(l, l) for l in train_labels_raw]
        else:
            final_train_labels = train_labels_raw

        train_counts = [final_train_labels.count(i) for i in range(self.num_classes)]

        print(f"\n📊 데이터셋 통계 (Total: {len(train_lines)}):")
        
        # --- [핵심] 클래스 가중치 계산 (Inverse Frequency) ---
        total_samples = sum(train_counts)
        class_weights = []
        
        print("⚖️  클래스 가중치 계산:")
        for cid, count in enumerate(train_counts):
            name = self.class_names[cid]
            if count > 0:
                weight = total_samples / (self.num_classes * count)
            else:
                weight = 1.0
            class_weights.append(weight)
            print(f"    * {cid} - {name}: {count}개 (Weight: {weight:.4f})")

        # 가중치를 텐서로 변환하고 Loss Function에 적용
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"  ✓ CrossEntropyLoss에 가중치가 적용되었습니다.\n")
        # -----------------------------------------------------
        
        return train_txt, val_txt, test_txt
    
    def _get_transforms(self, is_train=True):
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def train(self, train_txt, val_txt, test_txt=None):
        self._load_dinov2_model()
        
        train_dataset = DefectDataset(train_txt, transform=self._get_transforms(True), label_map=self.label_map)
        val_dataset = DefectDataset(val_txt, transform=self._get_transforms(False), label_map=self.label_map)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        # 마지막 검증 결과 저장용
        last_val_true = []
        last_val_pred = []

        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            
            # Freeze/Unfreeze Logic
            if epoch < self.freeze_epochs:
                for p in self.model.backbone.parameters(): p.requires_grad = False
                if self._opt_group_backbone_idx is not None:
                    self.optimizer.param_groups[self._opt_group_backbone_idx]['lr'] = 0.0
            else:
                for p in self.model.backbone.parameters(): p.requires_grad = True
                if self._opt_group_backbone_idx is not None:
                    self.optimizer.param_groups[self._opt_group_backbone_idx]['lr'] = self.lr_backbone
            
            train_loss, train_acc = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            val_loss, val_acc, val_true, val_pred = self._validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            last_val_true = val_true
            last_val_pred = val_pred
            
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint('best.pt', epoch, val_acc)
                print(f"  ✓ Best Saved! ({val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint('last.pt', epoch, val_acc)
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered")
                break
        
        self._save_checkpoint('last.pt', epoch, val_acc)
        self._save_results()
        self._plot_training_curves()

        # 결과 시각화
        try:
            if last_val_true and last_val_pred:
                self._plot_confusion_matrix(last_val_true, last_val_pred)
            self._visualize_feature_space(val_loader, 'val')
        except Exception as e:
            print(f"시각화 중 오류 발생: {e}")
        
        print(f"\n✅ 학습 완료. 결과 경로: {self.save_dir}")
        
        if test_txt and os.path.exists(test_txt):
            self._test_and_classify_images(test_txt)
    
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_true.extend(labels.cpu().tolist())
                all_pred.extend(predicted.cpu().tolist())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(val_loader), 100. * correct / total, all_true, all_pred
    
    def _save_checkpoint(self, filename, epoch, val_acc):
        weights_dir = self.save_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc
        }, weights_dir / filename)
    
    def _save_results(self):
        metrics = {
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
            'train_accs': self.train_accs, 'val_accs': self.val_accs,
            'best_val_acc': max(self.val_accs) if self.val_accs else 0
        }
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def _plot_training_curves(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Val')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train')
        plt.plot(self.val_accs, label='Val')
        plt.title('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / 'results.png')
        plt.close()

    def _visualize_feature_space(self, data_loader, split_name='val'):
            print(f"\n🔄 3D t-SNE 피처 공간 시각화 ({split_name})...")
            self.model.eval()
            all_features = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(data_loader, desc="Extracting features"):
                    images = images.to(self.device)
                    with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                        features = self.model.backbone(images)
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # 샘플링 (너무 많으면 느림)
            if len(all_features) > 2000:
                idx = np.random.permutation(len(all_features))[:2000]
                all_features, all_labels = all_features[idx], all_labels[idx]

            if len(all_features) <= 1: return

            perplexity = min(30.0, float(len(all_features) - 1))
            tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
            tsne_results = tsne.fit_transform(all_features)
            
            mapped_labels = [self.class_names[l] for l in all_labels]
            df = pd.DataFrame({
                'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'z': tsne_results[:, 2],
                'label': mapped_labels
            })
            
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                                title=f'DINOv2 3D Feature Space - {split_name}',
                                color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_traces(marker=dict(size=3, opacity=0.8))
            fig.write_html(str(self.save_dir / f'{split_name}_tsne_3d.html'))
            print(f"✓ 3D 시각화 저장 완료")

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = [[0]*self.num_classes for _ in range(self.num_classes)]
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                cm[t][p] += 1

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()

    def _test_and_classify_images(self, test_txt):
        correct_dir = self.save_dir / 'test_results' / 'correct'
        incorrect_dir = self.save_dir / 'test_results' / 'incorrect'
        correct_dir.mkdir(parents=True, exist_ok=True)
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        
        test_dataset = DefectDataset(test_txt, transform=self._get_transforms(False), label_map=self.label_map)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        self.model.eval()
        
        # 역정규화 transform (시각화용)
        inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                      std=[1/0.229, 1/0.224, 1/0.225]) if self.preprocess_enabled else None
        to_pil = transforms.ToPILImage()
        
        with open(test_txt, 'r') as f:
            test_lines = [line.strip() for line in f if line.strip()]
            
        correct_cnt, incorrect_cnt = 0, 0
        batch_idx = 0
        
        # Attention Map 저장 제한
        attn_saved_counts = {c: 0 for c in range(self.num_classes)}
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda' and self.use_amp)):
                    outputs = self.model(images)
                    # Attention Map 추출 시도 (모델이 지원할 경우)
                    attn_maps = None
                    if hasattr(self.model.backbone, 'get_last_selfattention'):
                         attn_maps = self.model.backbone.get_last_selfattention(images)

                _, predicted = outputs.max(1)
                probs = torch.softmax(outputs, dim=1)
                
                for i in range(len(images)):
                    global_idx = batch_idx * self.batch_size + i
                    if global_idx >= len(test_lines): break
                    
                    img_path_str = ' '.join(test_lines[global_idx].split()[:-1])
                    true_lbl = labels[i].item()
                    pred_lbl = predicted[i].item()
                    conf = probs[i][pred_lbl].item()
                    
                    is_correct = (true_lbl == pred_lbl)
                    dest_root = correct_dir if is_correct else incorrect_dir
                    
                    # 파일명 생성
                    fname = os.path.basename(img_path_str)
                    name, ext = os.path.splitext(fname)
                    true_name = self.class_names[true_lbl]
                    pred_name = self.class_names[pred_lbl]
                    new_name = f"{name}_GT-{true_name}_Pred-{pred_name}_Conf-{conf:.2f}{ext}"
                    
                    if os.path.exists(img_path_str):
                        shutil.copy2(img_path_str, dest_root / new_name)
                    
                    if is_correct: correct_cnt += 1
                    else: incorrect_cnt += 1
                    
                    # 어텐션 맵 저장 (클래스별 5개까지만)
                    if attn_maps is not None and attn_saved_counts[true_lbl] < 5:
                        try:
                            # [CLS] 토큰 어텐션 평균
                            cls_attn = attn_maps[i, :, 0, 1:].mean(dim=0) # shape: [patches]
                            w = h = int(np.sqrt(cls_attn.shape[0]))
                            cls_attn = cls_attn.reshape(w, h)
                            
                            # Upsample
                            cls_attn = F.interpolate(cls_attn.unsqueeze(0).unsqueeze(0), 
                                                   size=(self.imgsz, self.imgsz), mode='bilinear').squeeze()
                            
                            # Normalize & Heatmap
                            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
                            heatmap = (plt.get_cmap('jet')(cls_attn.cpu().numpy())[:,:,:3] * 255).astype(np.uint8)
                            
                            # Overlay
                            orig = images[i]
                            if inv_norm: orig = inv_norm(orig)
                            orig = to_pil(orig.cpu()).convert('RGB')
                            overlay = Image.blend(orig, Image.fromarray(heatmap), alpha=0.5)
                            
                            overlay.save(dest_root / f"ATTN_{new_name}")
                            attn_saved_counts[true_lbl] += 1
                        except:
                            pass
                
                batch_idx += 1
        
        print(f"\n📊 테스트 완료: 정답 {correct_cnt}, 오답 {incorrect_cnt}, 정확도 {100*correct_cnt/(correct_cnt+incorrect_cnt):.2f}%")

def main():
    parser = argparse.ArgumentParser(description='DINOv2 양품/불량 분류 학습 스크립트')
    
    parser.add_argument('--project', type=str, required=True, help='프로젝트 이름')
    parser.add_argument('--data-yaml', type=str, required=True, help='데이터셋 YAML 파일 경로')
    parser.add_argument('--model-size', type=str, default='small', 
                        choices=['small', 'base', 'large', 'giant'], help='모델 크기')
    parser.add_argument('--imgsz', type=int, default=224, help='이미지 크기')
    parser.add_argument('--batch', type=int, default=32, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='기본 학습률')
    
    # ▼▼▼ [수정됨] 누락된 인자 추가 ▼▼▼
    parser.add_argument('--lr-backbone', type=float, default=None, 
                        help='백본 학습률 (설정 안 하면 기본 lr의 0.1배)')
    parser.add_argument('--lr-head', type=float, default=None, 
                        help='헤드 학습률 (설정 안 하면 기본 lr과 동일)')
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    parser.add_argument('--freeze-epochs', type=int, default=0, help='초기 백본 고정 에포크')
    
    args = parser.parse_args()
    
    # Trainer 초기화 시 인자 전달
    trainer = DINOv2Trainer(
        project_name=args.project,
        model_size=args.model_size,
        imgsz=args.imgsz,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        # ▼▼▼ 전달 부분 추가 ▼▼▼
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        freeze_epochs=args.freeze_epochs
    )
    
    train_txt, val_txt, test_txt = trainer.load_data_yaml(args.data_yaml)
    trainer.train(train_txt, val_txt, test_txt)

if __name__ == "__main__":
    main()