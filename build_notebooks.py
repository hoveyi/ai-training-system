"""
生成四个模型的Jupyter Notebook训练文件，包含完整的评价指标图表
"""
import nbformat as nbf
import os

PROJECT_ROOT = r'C:\Users\DOVE\python学习\大二下实训'
OUT_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 通用：创建 notebook 结构
# ============================================================
def mk_cell(nb, source, ctype='code'):
    """添加一个 cell"""
    cell = nbf.v4.new_code_cell(source) if ctype == 'code' else nbf.v4.new_markdown_cell(source)
    nb.cells.append(cell)

def save_nb(nb, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'  Saved: {path}')

# ============================================================
# 1. 花卉分类 Notebook
# ============================================================
def build_flower_nb():
    nb = nbf.v4.new_notebook()
    mk_cell(nb, """# 花卉图像分类模型训练

## 任务：5种花卉分类（雏菊、蒲公英、玫瑰、向日葵、郁金香）
## 架构：ResNet50（迁移学习） + SimpleCNN（自定义CNN）
## 目标精度：≥85%
""", 'markdown')

    mk_cell(nb, """import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np, pandas as pd, os, sys, time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Auto-detect project root (walk up from cwd until backend/ + dataset/ found)
def get_project_root():
    current = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.exists(os.path.join(current, 'backend')) and os.path.exists(os.path.join(current, 'dataset')):
            return current
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return os.getcwd()  # fallback
ROOT = get_project_root()
print(f'Project root: {ROOT}')

sys.path.insert(0, os.path.join(ROOT, 'backend', 'flower_model'))
from model import get_model, SimpleCNN

import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
""")

    mk_cell(nb, """# ==================== 1. 数据探索 ====================
DATA_DIR = os.path.join(ROOT, 'dataset', 'flowers')
full_ds = datasets.ImageFolder(DATA_DIR)
print(f'总样本数: {len(full_ds)}')
print(f'类别: {full_ds.classes}')
print(f'类别→索引: {full_ds.class_to_idx}')

# 类别分布图
from collections import Counter
class_counts = Counter([label for _, label in full_ds])
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
colors = ['#FF6B6B','#4ECDC4','#A8E6CF','#FFD93D','#EE5A24']
ax[0].bar(full_ds.classes, [class_counts[i] for i in range(5)], color=colors, edgecolor='white')
ax[0].set_title('Samples per Class', fontsize=14)
ax[0].set_ylabel('Count')
for i, v in enumerate([class_counts[i] for i in range(5)]):
    ax[0].text(i, v+10, str(v), ha='center', fontweight='bold')

# 展示样本图片
indices = [np.where(np.array(full_ds.targets)==i)[0][0] for i in range(5)]
for i, (idx, cls) in enumerate(zip(indices, full_ds.classes)):
    img, _ = full_ds[idx]
    ax[1].imshow(img)
    ax[1].set_title(cls, fontsize=12)
    ax[1].axis('off')
ax[1].set_title('Sample per Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'flower_model', 'checkpoints', '01_data_overview.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 2. 数据预处理与加载 ====================
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform: x = self.transform(x)
        return x, y

train_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20), transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ColorJitter(0.2,0.2,0.2), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# 分层划分
from sklearn.model_selection import train_test_split
indices = list(range(len(full_ds)))
tr_idx, tmp = train_test_split(indices, test_size=0.3, random_state=42, stratify=full_ds.targets)
v_idx, te_idx = train_test_split(tmp, test_size=0.5, random_state=42, stratify=[full_ds.targets[i] for i in tmp])
print(f'Train: {len(tr_idx)} | Val: {len(v_idx)} | Test: {len(te_idx)}')

BATCH = 32
tr_loader = DataLoader(TransformedDataset(full_ds, tr_idx, train_tf), batch_size=BATCH, shuffle=True)
v_loader  = DataLoader(TransformedDataset(full_ds, v_idx, val_tf), batch_size=BATCH, shuffle=False)
te_loader  = DataLoader(TransformedDataset(full_ds, te_idx, val_tf), batch_size=BATCH, shuffle=False)
""")

    mk_cell(nb, """# ==================== 3. 训练工具函数 ====================
def train_epoch(model, loader, crit, opt, dev):
    model.train(); total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(model(x), y); loss.backward(); opt.step()
        total_loss += loss.item()
        correct += model(x).argmax(1).eq(y).sum().item()  # re-eval
        total += y.size(0)
    return total_loss/len(loader), 100.*correct/total

@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval(); total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        total_loss += crit(out, y).item()
        preds = out.argmax(1)
        correct += preds.eq(y).sum().item(); total += y.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    return total_loss/len(loader), 100.*correct/total, all_preds, all_labels

def train_full(model, name, tr_loader, v_loader, epochs, lr, patience, dev):
    model = model.to(dev); crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    history = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    best_acc, best_st, wait = 0, None, 0
    for ep in range(epochs):
        tl, ta = train_epoch(model, tr_loader, crit, opt, dev)
        vl, va, _, _ = evaluate(model, v_loader, crit, dev)
        sched.step(vl)
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)
        if (ep+1)%5==0: print(f'[{name}] E{ep+1:3d} | TL={tl:.3f} TA={ta:.1f}% | VL={vl:.3f} VA={va:.1f}%')
        if va>best_acc: best_acc=va; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=patience: print(f'Early stop at {ep+1}'); break
    model.load_state_dict(best_st)
    return model, history, best_acc
""")

    mk_cell(nb, """# ==================== 4. 训练 ResNet50 ====================
print('=== Training ResNet50 (Transfer Learning) ===')
resnet = get_model('resnet50', 5)
resnet, hist_rn, acc_rn = train_full(resnet, 'ResNet50', tr_loader, v_loader, epochs=50, lr=0.001, patience=10, dev=DEVICE)

# 测试
_, te_acc_rn, rn_preds, rn_labels = evaluate(resnet, te_loader, nn.CrossEntropyLoss(), DEVICE)
print(f'ResNet50 Test Accuracy: {te_acc_rn:.2f}%')
""")

    mk_cell(nb, """# ==================== 5. 训练 SimpleCNN ====================
print('=== Training SimpleCNN ===')
cnn = SimpleCNN(5)
cnn, hist_cnn, acc_cnn = train_full(cnn, 'SimpleCNN', tr_loader, v_loader, epochs=80, lr=0.001, patience=15, dev=DEVICE)

_, te_acc_cnn, cnn_preds, cnn_labels = evaluate(cnn, te_loader, nn.CrossEntropyLoss(), DEVICE)
print(f'SimpleCNN Test Accuracy: {te_acc_cnn:.2f}%')
""")

    mk_cell(nb, """# ==================== 6. 训练曲线对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, key, title, ylab in [
    (axes[0,0], 'train_loss', 'Training Loss', 'Loss'),
    (axes[0,1], 'val_loss', 'Validation Loss', 'Loss'),
    (axes[1,0], 'train_acc', 'Training Accuracy', 'Accuracy (%)'),
    (axes[1,1], 'val_acc', 'Validation Accuracy', 'Accuracy (%)')]:
    ax.plot(hist_rn[key], label='ResNet50', color='#FF6B6B', linewidth=2)
    ax.plot(hist_cnn[key], label='SimpleCNN', color='#4ECDC4', linewidth=2)
    ax.set_title(title, fontsize=13); ax.set_xlabel('Epoch'); ax.set_ylabel(ylab)
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'flower_model', 'checkpoints', '02_training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 7. 测试精度对比柱状图 ====================
fig, ax = plt.subplots(figsize=(8, 5))
models = ['ResNet50', 'SimpleCNN']
accs = [te_acc_rn, te_acc_cnn]
bars = ax.bar(models, accs, color=['#FF6B6B','#4ECDC4'], edgecolor='white', width=0.4)
for b, v in zip(bars, accs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=14)
ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Test Accuracy Comparison (Target >= 85%)', fontsize=14)
ax.axhline(y=85, color='green', linestyle='--', label='Target 85%'); ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'flower_model', 'checkpoints', '03_accuracy_compare.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 8. 混淆矩阵 (最佳模型) ====================
best_model = resnet if te_acc_rn >= te_acc_cnn else cnn
best_preds = rn_preds if te_acc_rn >= te_acc_cnn else cnn_preds
best_labels = rn_labels if te_acc_rn >= te_acc_cnn else cnn_labels
best_name = 'ResNet50' if te_acc_rn >= te_acc_cnn else 'SimpleCNN'

cm = confusion_matrix(best_labels, best_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=full_ds.classes, yticklabels=full_ds.classes)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix - {best_name} (Acc: {max(te_acc_rn, te_acc_cnn):.1f}%)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'flower_model', 'checkpoints', '04_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# 分类报告
print('\\nClassification Report:')
print(classification_report(best_labels, best_preds, target_names=full_ds.classes))
""")

    mk_cell(nb, """# ==================== 9. 保存模型与结果 ====================
CHECKPOINT_DIR = os.path.join(ROOT, 'backend', 'flower_model', 'checkpoints')
torch.save({
    'resnet50_state_dict': {k:v.cpu().clone() for k,v in resnet.state_dict().items()},
    'simple_cnn_state_dict': {k:v.cpu().clone() for k,v in cnn.state_dict().items()},
    'class_names': full_ds.classes,
    'test_accuracy_resnet50': te_acc_rn,
    'test_accuracy_simple_cnn': te_acc_cnn,
}, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
print(f'Model saved to {CHECKPOINT_DIR}')
print(f'ResNet50: {te_acc_rn:.2f}% | SimpleCNN: {te_acc_cnn:.2f}%')
""")

    save_nb(nb, '01_flower_classification.ipynb')


# ============================================================
# 2. Titanic 生存预测 Notebook
# ============================================================
def build_titanic_nb():
    nb = nbf.v4.new_notebook()
    mk_cell(nb, """# Titanic 旅客生存概率预测

## 任务：二分类——预测乘客是否生存
## 架构：MLP深度型（4层+BN+Dropout） + MLP宽型（3层大隐藏层）
## 目标精度：≥70%
""", 'markdown')

    mk_cell(nb, """import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, pandas as pd, os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Auto-detect project root
def get_project_root():
    current = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.exists(os.path.join(current, 'backend')) and os.path.exists(os.path.join(current, 'dataset')):
            return current
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return os.getcwd()
ROOT = get_project_root()
print(f'Project root: {ROOT}')

sys.path.insert(0, os.path.join(ROOT, 'backend', 'titanic_model'))
from model import get_model

import warnings as _w; _w.filterwarnings('ignore', category=UserWarning)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
""")

    mk_cell(nb, """# ==================== 1. 数据加载与探索 ====================
import seaborn as sns_lib
df = sns_lib.load_dataset('titanic')
print(f'数据形状: {df.shape}')
print(f'列名: {list(df.columns)}')
print(f'\\n缺失值:\\n{df.isnull().sum()}')
print(f'\\n生存率: {df.survived.mean()*100:.1f}%')

# 可视化：各类别分布
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features = ['pclass','sex','age','embarked','sibsp','parch']
for ax, feat in zip(axes.flat, features):
    if feat in ['age','fare']:
        for survived in [0,1]:
            subset = df[df.survived==survived][feat].dropna()
            ax.hist(subset, alpha=0.6, label=f'Survived={survived}', bins=25)
        ax.legend()
    else:
        ct = pd.crosstab(df[feat].fillna('Unknown'), df.survived)
        ct.plot(kind='bar', ax=ax, color=['#FF6B6B','#4ECDC4'])
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'titanic_model', 'checkpoints', '01_eda.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 2. 特征工程 ====================
df2 = df[['pclass','sex','age','sibsp','parch','fare','embarked','survived']].copy()
df2['sex'] = df2['sex'].map({'male':0,'female':1})
df2['embarked'] = df2['embarked'].map({'C':0,'Q':1,'S':2})
df2['age'] = df2['age'].fillna(df2['age'].median())
df2['fare'] = df2['fare'].fillna(df2['fare'].median())
df2['embarked'] = df2['embarked'].fillna(2)
df2['family_size'] = df2['sibsp'] + df2['parch'] + 1
df2['is_alone'] = (df2['family_size']==1).astype(int)

FEATURES = ['pclass','sex','age','sibsp','parch','fare','embarked','family_size','is_alone']
X = df2[FEATURES].values.astype(np.float32); y = df2['survived'].values.astype(np.float32)

scaler = StandardScaler(); X = scaler.fit_transform(X)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_v, X_te, y_v, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print(f'Train: {len(X_tr)} | Val: {len(X_v)} | Test: {len(X_te)}')

BATCH = 32
tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1)), BATCH, True)
v_loader  = DataLoader(TensorDataset(torch.tensor(X_v), torch.tensor(y_v).unsqueeze(1)), BATCH, False)
te_loader  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te).unsqueeze(1)), BATCH, False)
""")

    mk_cell(nb, """# ==================== 3. 训练函数 ====================
def train_epoch_t(model, loader, crit, opt, dev):
    model.train(); loss_sum, correct, total = 0, 0, 0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); opt.zero_grad()
        loss=crit(model(x),y); loss.backward(); opt.step()
        loss_sum+=loss.item(); preds=(model(x)>0.5).float()
        correct+=preds.eq(y).sum().item(); total+=y.size(0)
    return loss_sum/len(loader), correct/total

@torch.no_grad()
def evaluate_t(model, loader, crit, dev):
    model.eval(); loss_sum, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); out=model(x)
        loss_sum+=crit(out,y).item(); preds=(out>0.5).float()
        correct+=preds.eq(y).sum().item(); total+=y.size(0)
        all_probs.extend(out.cpu().numpy().flatten()); all_labels.extend(y.cpu().numpy().flatten())
    return loss_sum/len(loader), correct/total, all_probs, all_labels

def train_titanic(model, name, tr_loader, v_loader, epochs, patience, dev):
    model=model.to(dev); crit=nn.BCELoss(); opt=optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
    hist={'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    best_acc, best_st, wait=0, None, 0
    for ep in range(epochs):
        tl,ta=train_epoch_t(model,tr_loader,crit,opt,dev)
        vl,va,_,_=evaluate_t(model,v_loader,crit,dev); sched.step(vl)
        for k,v in zip(['train_loss','train_acc','val_loss','val_acc'],[tl,ta,vl,va]): hist[k].append(v)
        if (ep+1)%20==0: print(f'[{name}] E{ep+1:3d} | TA={ta:.4f} | VA={va:.4f}')
        if va>best_acc: best_acc=va; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=patience: print(f'Early stop at {ep+1}'); break
    model.load_state_dict(best_st)
    return model, hist, best_acc
""")

    mk_cell(nb, """# ==================== 4. 训练 MLP 深度型 ====================
print('=== MLP Deep (4层+BN) ===')
mlp_deep = get_model('mlp', X.shape[1])
mlp_deep, hist_deep, acc_deep = train_titanic(mlp_deep, 'MLP-Deep', tr_loader, v_loader, 200, 30, DEVICE)
_, te_acc_deep, probs_deep, labels_deep = evaluate_t(mlp_deep, te_loader, nn.BCELoss(), DEVICE)
print(f'MLP-Deep Test Acc: {te_acc_deep:.4f}')
""")

    mk_cell(nb, """# ==================== 5. 训练 MLP 宽型 ====================
print('=== MLP Wide (3层宽网络) ===')
mlp_wide = get_model('mlp_wide', X.shape[1])
mlp_wide, hist_wide, acc_wide = train_titanic(mlp_wide, 'MLP-Wide', tr_loader, v_loader, 200, 30, DEVICE)
_, te_acc_wide, probs_wide, labels_wide = evaluate_t(mlp_wide, te_loader, nn.BCELoss(), DEVICE)
print(f'MLP-Wide Test Acc: {te_acc_wide:.4f}')
""")

    mk_cell(nb, """# ==================== 6. 训练曲线对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, key, title in [(axes[0,0],'train_loss','Training Loss'), (axes[0,1],'val_loss','Validation Loss'),
                        (axes[1,0],'train_acc','Training Accuracy'), (axes[1,1],'val_acc','Validation Accuracy')]:
    ax.plot(hist_deep[key], label='MLP-Deep', color='#FF6B6B', linewidth=2)
    ax.plot(hist_wide[key], label='MLP-Wide', color='#4ECDC4', linewidth=2)
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'titanic_model', 'checkpoints', '02_training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 7. ROC曲线对比 ====================
fig, ax = plt.subplots(figsize=(8, 6))
for probs, labels, name, color in [(probs_deep, labels_deep, 'MLP-Deep', '#FF6B6B'),
                                     (probs_wide, labels_wide, 'MLP-Wide', '#4ECDC4')]:
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr,tpr):.3f})', color=color, linewidth=2)
ax.plot([0,1],[0,1],'k--',alpha=0.3); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curve Comparison', fontsize=14); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'titanic_model', 'checkpoints', '03_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 8. 模型对比与保存 ====================
fig, ax = plt.subplots(figsize=(8, 5))
models = ['MLP-Deep', 'MLP-Wide']
accs = [te_acc_deep*100, te_acc_wide*100]
bars = ax.bar(models, accs, color=['#FF6B6B','#4ECDC4'], edgecolor='white', width=0.4)
for b, v in zip(bars, accs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=14)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Test Accuracy Comparison (Target >= 70%)', fontsize=14)
ax.axhline(y=70, color='green', linestyle='--', label='Target 70%'); ax.legend(); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'titanic_model', 'checkpoints', '04_accuracy_compare.png'), dpi=150, bbox_inches='tight')
plt.show()

# 保存
CKPT = os.path.join(ROOT, 'backend', 'titanic_model', 'checkpoints')
torch.save({
    'mlp_state_dict': {k:v.cpu().clone() for k,v in mlp_deep.state_dict().items()},
    'mlp_wide_state_dict': {k:v.cpu().clone() for k,v in mlp_wide.state_dict().items()},
    'input_dim': X.shape[1], 'feature_cols': FEATURES,
    'test_accuracy_mlp': te_acc_deep, 'test_accuracy_mlp_wide': te_acc_wide,
    'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
}, os.path.join(CKPT, 'best_model.pth'))
print(f'Saved. MLP-Deep: {te_acc_deep*100:.1f}% | MLP-Wide: {te_acc_wide*100:.1f}%')
""")

    save_nb(nb, '02_titanic_survival.ipynb')


# ============================================================
# 3. 时尚服饰分类 Notebook
# ============================================================
def build_fashion_nb():
    nb = nbf.v4.new_notebook()
    mk_cell(nb, """# 时尚服饰图像分类模型训练

## 任务：10类服饰识别（T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、踝靴）
## 架构：FashionCNN（7层自定义CNN） + FashionResNet（ResNet18迁移学习）
## 目标精度：≥80%
""", 'markdown')

    mk_cell(nb, """import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, os, sys, struct
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Auto-detect project root
def get_project_root():
    current = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.exists(os.path.join(current, 'backend')) and os.path.exists(os.path.join(current, 'dataset')):
            return current
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return os.getcwd()
ROOT = get_project_root()
print(f'Project root: {ROOT}')

sys.path.insert(0, os.path.join(ROOT, 'backend', 'fashion_model'))
from model import get_model

import warnings as _w; _w.filterwarnings('ignore', category=UserWarning)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
""")

    mk_cell(nb, """# ==================== 1. 加载本地IDX数据 ====================
def read_idx(fn):
    with open(fn,'rb') as f:
        _, dtype, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

DATA_DIR = os.path.join(ROOT, 'dataset', 'cloths')
tr_img = read_idx(os.path.join(DATA_DIR, 'train-images-idx3-ubyte','train-images-idx3-ubyte'))
tr_lbl = read_idx(os.path.join(DATA_DIR, 'train-labels-idx1-ubyte','train-labels-idx1-ubyte'))
te_img = read_idx(os.path.join(DATA_DIR, 't10k-images-idx3-ubyte','t10k-images-idx3-ubyte'))
te_lbl = read_idx(os.path.join(DATA_DIR, 't10k-labels-idx1-ubyte','t10k-labels-idx1-ubyte'))
print(f'Train: {tr_img.shape}, Test: {te_img.shape}')

# 类别分布
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, lbls, title in [(axes[0], tr_lbl, 'Training Set'), (axes[1], te_lbl, 'Test Set')]:
    counts = np.bincount(lbls)
    ax.bar(CLASS_NAMES, counts, color=plt.cm.tab10(np.arange(10)), edgecolor='white')
    ax.set_title(f'{title} Class Distribution', fontsize=13); ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'fashion_model', 'checkpoints', '01_data_dist.png'), dpi=150, bbox_inches='tight')
plt.show()

# 样本展示
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    idx = np.where(tr_lbl==i)[0][0]
    axes[i//5][i%5].imshow(tr_img[idx], cmap='gray')
    axes[i//5][i%5].set_title(CLASS_NAMES[i], fontsize=10); axes[i//5][i%5].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'fashion_model', 'checkpoints', '01_samples.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 2. 数据预处理 ====================
MEAN, STD = 0.2860, 0.3530
tr_img = tr_img.astype(np.float32)/255.0; te_img = te_img.astype(np.float32)/255.0
tr_t = torch.tensor(tr_img).unsqueeze(1); tr_l = torch.tensor(tr_lbl, dtype=torch.long)
te_t = torch.tensor(te_img).unsqueeze(1); te_l = torch.tensor(te_lbl, dtype=torch.long)

# 验证集划分
val_size = int(0.1*len(tr_t))
indices = torch.randperm(len(tr_t), generator=torch.Generator().manual_seed(42))
v_idx, tr_idx = indices[:val_size], indices[val_size:]

class AugDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, mean, std, augment=False):
        self.imgs, self.labels, self.mean, self.std, self.augment = imgs, labels, mean, std, augment
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        x, y = self.imgs[idx], self.labels[idx]
        if self.augment and torch.rand(1)>0.5: x = torch.flip(x, [2])
        return (x-self.mean)/self.std, y

BATCH = 128
tr_loader = DataLoader(AugDataset(tr_t[tr_idx], tr_l[tr_idx], MEAN, STD, True), BATCH, True)
v_loader  = DataLoader(AugDataset(tr_t[v_idx], tr_l[v_idx], MEAN, STD, False), BATCH, False)
te_loader  = DataLoader(AugDataset(te_t, te_l, MEAN, STD, False), BATCH, False)
print(f'Train: {len(tr_idx)} | Val: {len(v_idx)} | Test: {len(te_t)}')
""")

    mk_cell(nb, """# ==================== 3. 训练工具函数 ====================
def f_train_ep(model, loader, crit, opt, dev):
    model.train(); loss_sum, correct, total = 0, 0, 0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); opt.zero_grad()
        loss=crit(model(x),y); loss.backward(); opt.step()
        loss_sum+=loss.item(); correct+=model(x).argmax(1).eq(y).sum().item(); total+=y.size(0)
    return loss_sum/len(loader), 100.*correct/total

@torch.no_grad()
def f_eval(model, loader, crit, dev):
    model.eval(); loss_sum, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); out=model(x)
        loss_sum+=crit(out,y).item(); preds=out.argmax(1)
        correct+=preds.eq(y).sum().item(); total+=y.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    return loss_sum/len(loader), 100.*correct/total, all_preds, all_labels

def f_train(model, name, tr_loader, v_loader, epochs, dev):
    model=model.to(dev); crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(), lr=0.001)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist={'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    best_acc, best_st = 0, None
    for ep in range(epochs):
        tl,ta=f_train_ep(model,tr_loader,crit,opt,dev)
        vl,va,_,_=f_eval(model,v_loader,crit,dev); sched.step()
        for k,v in zip(['train_loss','train_acc','val_loss','val_acc'],[tl,ta,vl,va]): hist[k].append(v)
        print(f'[{name}] E{ep+1:2d}/{epochs} | TA={ta:.2f}% VA={va:.2f}%')
        if va>best_acc: best_acc=va; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model, hist, best_acc
""")

    mk_cell(nb, """# ==================== 4. 训练 CNN ====================
print('=== FashionCNN ===')
cnn_f = get_model('cnn', 10)
cnn_f, hist_cnn_f, acc_cnn_f = f_train(cnn_f, 'CNN', tr_loader, v_loader, 30, DEVICE)
_, te_acc_cnn_f, preds_cnn, labels_cnn = f_eval(cnn_f, te_loader, nn.CrossEntropyLoss(), DEVICE)
print(f'CNN Test Accuracy: {te_acc_cnn_f:.2f}%')
""")

    mk_cell(nb, """# ==================== 5. 训练 ResNet18 ====================
print('=== FashionResNet18 ===')
resnet_f = get_model('resnet18', 10)
resnet_f, hist_rn_f, acc_rn_f = f_train(resnet_f, 'ResNet18', tr_loader, v_loader, 30, DEVICE)
_, te_acc_rn_f, preds_rn, labels_rn = f_eval(resnet_f, te_loader, nn.CrossEntropyLoss(), DEVICE)
print(f'ResNet18 Test Accuracy: {te_acc_rn_f:.2f}%')
""")

    mk_cell(nb, """# ==================== 6. 训练曲线对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, key, title in [(axes[0,0],'train_loss','Training Loss'), (axes[0,1],'val_loss','Validation Loss'),
                        (axes[1,0],'train_acc','Training Accuracy'), (axes[1,1],'val_acc','Validation Accuracy')]:
    ax.plot(hist_cnn_f[key], label='CNN', color='#FF6B6B', linewidth=2)
    ax.plot(hist_rn_f[key], label='ResNet18', color='#4ECDC4', linewidth=2)
    ax.set_title(title, fontsize=13); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'fashion_model', 'checkpoints', '02_training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 7. 混淆矩阵 ====================
best_f = cnn_f if te_acc_cnn_f >= te_acc_rn_f else resnet_f
best_p = preds_cnn if te_acc_cnn_f >= te_acc_rn_f else preds_rn
best_l = labels_cnn if te_acc_cnn_f >= te_acc_rn_f else labels_rn
bname = 'CNN' if te_acc_cnn_f >= te_acc_rn_f else 'ResNet18'

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(best_l, best_p)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix - {bname} ({max(te_acc_cnn_f, te_acc_rn_f):.1f}%)', fontsize=14)
plt.xticks(rotation=45); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'fashion_model', 'checkpoints', '03_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print(classification_report(best_l, best_p, target_names=CLASS_NAMES))
""")

    mk_cell(nb, """# ==================== 8. 精度对比与保存 ====================
fig, ax = plt.subplots(figsize=(8, 5))
models_f = ['CNN', 'ResNet18']
accs_f = [te_acc_cnn_f, te_acc_rn_f]
bars = ax.bar(models_f, accs_f, color=['#FF6B6B','#4ECDC4'], edgecolor='white', width=0.4)
for b, v in zip(bars, accs_f): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)
ax.axhline(y=80, color='green', linestyle='--', label='Target 80%'); ax.legend()
ax.set_ylabel('Accuracy (%)'); ax.set_title('Test Accuracy Comparison (Target >= 80%)', fontsize=14); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'fashion_model', 'checkpoints', '04_accuracy_compare.png'), dpi=150, bbox_inches='tight')
plt.show()

CKPT = os.path.join(ROOT, 'backend', 'fashion_model', 'checkpoints')
torch.save({
    'cnn_state_dict': {k:v.cpu().clone() for k,v in cnn_f.state_dict().items()},
    'resnet18_state_dict': {k:v.cpu().clone() for k,v in resnet_f.state_dict().items()},
    'num_classes':10, 'class_names':CLASS_NAMES,
    'test_accuracy_cnn': te_acc_cnn_f/100., 'test_accuracy_resnet18': te_acc_rn_f/100.,
}, os.path.join(CKPT, 'best_model.pth'))
print(f'Saved. CNN: {te_acc_cnn_f:.1f}% | ResNet18: {te_acc_rn_f:.1f}%')
""")

    save_nb(nb, '03_fashion_classification.ipynb')


# ============================================================
# 4. 非线性回归 Notebook
# ============================================================
def build_regression_nb():
    nb = nbf.v4.new_notebook()
    mk_cell(nb, """# 非线性系统回归预测模型训练

## 任务：多输入多输出非线性系统映射学习
## 架构：MLP（多层感知机） + LSTM（时序记忆网络）
## 目标：R² ≥ 0.50
""", 'markdown')

    mk_cell(nb, """import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# Auto-detect project root
def get_project_root():
    current = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.exists(os.path.join(current, 'backend')) and os.path.exists(os.path.join(current, 'dataset')):
            return current
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return os.getcwd()
ROOT = get_project_root()
print(f'Project root: {ROOT}')

sys.path.insert(0, os.path.join(ROOT, 'backend', 'regression_model'))
from model import get_model

import warnings as _w; _w.filterwarnings('ignore', category=UserWarning)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
""")

    mk_cell(nb, """# ==================== 1. 生成合成非线性数据 ====================
np.random.seed(42); N = 5000
x1 = np.random.uniform(-3, 3, N); x2 = np.random.uniform(-3, 3, N); x3 = np.random.uniform(-3, 3, N)
y1 = np.sin(x1*1.5)*np.cos(x2*0.8) + x3**2*0.3 + np.exp(-np.abs(x2))*0.5 + np.sin(x1*x2*0.3)*0.4
y2 = np.cos(x3*1.2)*np.sin(x1*0.7) + x2**3*0.1 + np.tanh(x1+x2+x3)*0.6
y3 = np.sin(x1+x2)*np.cos(x3)*0.8 + np.abs(x1*x3)*0.15 + x2**2*0.2 + np.cos(x1*x2*x3*0.1)*0.3
for y in [y1,y2,y3]: y += np.random.normal(0, 0.05, N)
X = np.stack([x1,x2,x3],1).astype(np.float32); Y = np.stack([y1,y2,y3],1).astype(np.float32)
print(f'X: {X.shape}, Y: {Y.shape}')

# 数据可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (ax, xi) in enumerate(zip(axes, [x1,x2,x3])):
    ax.scatter(xi, y1, c=y2, cmap='coolwarm', s=2, alpha=0.5)
    ax.set_xlabel(f'x{i+1}'); ax.set_ylabel('y1')
    ax.set_title(f'y1 vs x{i+1} (color=y2)')
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints', '01_data_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()

# 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x1[::50], x2[::50], x3[::50], c=y1[::50], cmap='coolwarm', s=20)
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
ax.set_title('3D Input Space (color=y1)'); plt.colorbar(sc, label='y1')
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints', '01_3d_input.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 2. 数据预处理 ====================
scaler_X = StandardScaler(); scaler_Y = StandardScaler()
Xs = scaler_X.fit_transform(X); Ys = scaler_Y.fit_transform(Y)
X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(Xs, Ys, test_size=0.3, random_state=42)
X_v, X_te, Y_v, Y_te = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)
print(f'Train: {len(X_tr)} | Val: {len(X_v)} | Test: {len(X_te)}')

BATCH = 64
tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)), BATCH, True)
v_loader  = DataLoader(TensorDataset(torch.tensor(X_v), torch.tensor(Y_v)), BATCH, False)
te_loader  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(Y_te)), BATCH, False)
""")

    mk_cell(nb, """# ==================== 3. 训练 MLP 回归器 ====================
print('=== MLP Regressor (3->128->128->64->3) ===')
mlp_r = get_model('mlp', 3, 3).to(DEVICE)
crit = nn.MSELoss(); opt = optim.Adam(mlp_r.parameters(), lr=0.001)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)

hist_mlp = {'train_loss':[], 'val_loss':[]}
best_loss, best_st, wait = float('inf'), None, 0
for ep in range(200):
    mlp_r.train(); tl=0
    for x,y in tr_loader:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        loss=crit(mlp_r(x),y); loss.backward(); opt.step(); tl+=loss.item()
    tl/=len(tr_loader)
    mlp_r.eval(); vl=0
    with torch.no_grad():
        for x,y in v_loader: x,y=x.to(DEVICE),y.to(DEVICE); vl+=crit(mlp_r(x),y).item()
    vl/=len(v_loader); sched.step(vl)
    hist_mlp['train_loss'].append(tl); hist_mlp['val_loss'].append(vl)
    if vl<best_loss: best_loss=vl; best_st={k:v.cpu().clone() for k,v in mlp_r.state_dict().items()}; wait=0
    else: wait+=1
    if wait>=40: print(f'MLP early stop at {ep+1}'); break
    if (ep+1)%20==0: print(f'[MLP] E{ep+1:3d} | TL={tl:.4f} VL={vl:.4f}')
mlp_r.load_state_dict(best_st)

# Test R²
mlp_r.eval(); preds_mlp, truths = [], []
with torch.no_grad():
    for x,y in te_loader: x=x.to(DEVICE); preds_mlp.append(mlp_r(x).cpu().numpy()); truths.append(y.numpy())
preds_mlp = np.concatenate(preds_mlp); truths = np.concatenate(truths)
r2_mlp = r2_score(truths, preds_mlp, multioutput='uniform_average')
mse_mlp = mean_squared_error(truths, preds_mlp)
print(f'MLP Test R²={r2_mlp:.4f}, MSE={mse_mlp:.4f}')
""")

    mk_cell(nb, """# ==================== 4. 生成时序数据并训练 LSTM ====================
print('=== LSTM Regressor (seq_len=10, 3->64->64->1) ===')
# 生成时序数据
N2, SL = 5000, 10
t = np.linspace(0, 50, N2+SL)
s1 = np.sin(t*0.7)+0.5*np.sin(t*1.5)+np.random.normal(0,0.05,N2+SL)
s2 = np.cos(t*0.6)+0.3*np.sin(t*2.0)+np.random.normal(0,0.05,N2+SL)
s3 = np.sin(t*0.9+1.0)*np.cos(t*0.3)+np.random.normal(0,0.05,N2+SL)
target = np.sin(t*0.8)*np.cos(t*0.5)+np.exp(-0.1*t)*np.sin(t*1.2)+0.2*np.sin(t*2.1)*np.cos(t*0.4)

X_seq = np.stack([s1,s2,s3],1).astype(np.float32)
Y_seq = target.astype(np.float32)

# 滑动窗口
Xw, Yw = [], []
for i in range(N2):
    Xw.append(X_seq[i:i+SL]); Yw.append(Y_seq[i+SL])
Xw = np.array(Xw); Yw = np.array(Yw)

# 标准化时序数据
ns, sl, nf = Xw.shape
scaler_Xs = StandardScaler().fit(Xw.reshape(-1, nf))
Xws = scaler_Xs.transform(Xw.reshape(-1, nf)).reshape(ns, sl, nf)
scaler_Ys = StandardScaler().fit(Yw.reshape(-1,1))
Yws = scaler_Ys.transform(Yw.reshape(-1,1)).flatten()

X_tr_s, X_tmp_s, Y_tr_s, Y_tmp_s = train_test_split(Xws, Yws, test_size=0.3, random_state=42)
X_v_s, X_te_s, Y_v_s, Y_te_s = train_test_split(X_tmp_s, Y_tmp_s, test_size=0.5, random_state=42)

tr_loader_s = DataLoader(TensorDataset(torch.tensor(X_tr_s), torch.tensor(Y_tr_s).unsqueeze(1)), BATCH, True)
v_loader_s  = DataLoader(TensorDataset(torch.tensor(X_v_s), torch.tensor(Y_v_s).unsqueeze(1)), BATCH, False)
te_loader_s  = DataLoader(TensorDataset(torch.tensor(X_te_s), torch.tensor(Y_te_s).unsqueeze(1)), BATCH, False)

# 训练LSTM
lstm_r = get_model('lstm', 3, 1).to(DEVICE)
crit_s = nn.MSELoss(); opt_s = optim.Adam(lstm_r.parameters(), lr=0.001)
sched_s = optim.lr_scheduler.ReduceLROnPlateau(opt_s, mode='min', patience=10, factor=0.5)

hist_lstm = {'train_loss':[], 'val_loss':[]}
best_loss_s, best_st_s, wait_s = float('inf'), None, 0
for ep in range(200):
    lstm_r.train(); tl=0
    for x,y in tr_loader_s:
        x,y=x.to(DEVICE),y.to(DEVICE); opt_s.zero_grad()
        loss=crit_s(lstm_r(x),y); loss.backward(); opt_s.step(); tl+=loss.item()
    tl/=len(tr_loader_s)
    lstm_r.eval(); vl=0
    with torch.no_grad():
        for x,y in v_loader_s: x,y=x.to(DEVICE),y.to(DEVICE); vl+=crit_s(lstm_r(x),y).item()
    vl/=len(v_loader_s); sched_s.step(vl)
    hist_lstm['train_loss'].append(tl); hist_lstm['val_loss'].append(vl)
    if vl<best_loss_s: best_loss_s=vl; best_st_s={k:v.cpu().clone() for k,v in lstm_r.state_dict().items()}; wait_s=0
    else: wait_s+=1
    if wait_s>=40: print(f'LSTM early stop at {ep+1}'); break
    if (ep+1)%20==0: print(f'[LSTM] E{ep+1:3d} | TL={tl:.4f} VL={vl:.4f}')
lstm_r.load_state_dict(best_st_s)

# Test R²
lstm_r.eval(); preds_lstm, truths_s = [], []
with torch.no_grad():
    for x,y in te_loader_s: x=x.to(DEVICE); preds_lstm.append(lstm_r(x).cpu().numpy()); truths_s.append(y.numpy())
preds_lstm = np.concatenate(preds_lstm); truths_s = np.concatenate(truths_s)
r2_lstm = r2_score(truths_s, preds_lstm)
mse_lstm = mean_squared_error(truths_s, preds_lstm)
print(f'LSTM Test R²={r2_lstm:.4f}, MSE={mse_lstm:.4f}')
""")

    mk_cell(nb, """# ==================== 5. 训练Loss曲线对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, title in [(axes[0], 'train_loss', 'Training Loss'), (axes[1], 'val_loss', 'Validation Loss')]:
    ax.plot(hist_mlp[key], label='MLP', color='#FF6B6B', linewidth=2)
    ax.plot(hist_lstm[key], label='LSTM', color='#4ECDC4', linewidth=2)
    ax.set_title(f'{title} Comparison', fontsize=13); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints', '02_training_loss.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 6. R² 对比柱状图 ====================
fig, ax = plt.subplots(figsize=(8, 5))
models_r = ['MLP', 'LSTM']
r2s = [r2_mlp, r2_lstm]
bars = ax.bar(models_r, r2s, color=['#FF6B6B','#4ECDC4'], edgecolor='white', width=0.4)
for b, v in zip(bars, r2s): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}', ha='center', fontweight='bold', fontsize=14)
ax.axhline(y=0.5, color='green', linestyle='--', label='Target R²=0.5'); ax.legend()
ax.set_ylabel('R² Score'); ax.set_title('R² Score Comparison (Target >= 0.50)', fontsize=14); ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints', '03_r2_compare.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 7. 预测 vs 真实值散点图 (MLP) ====================
Y_pred_orig = scaler_Y.inverse_transform(preds_mlp)
Y_true_orig = scaler_Y.inverse_transform(truths)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes):
    ax.scatter(Y_true_orig[:,i], Y_pred_orig[:,i], c='#4ECDC4', s=5, alpha=0.5)
    ax.plot([Y_true_orig[:,i].min(), Y_true_orig[:,i].max()],
            [Y_true_orig[:,i].min(), Y_true_orig[:,i].max()], 'r--', lw=1, label='Ideal')
    ax.set_xlabel(f'True y{i+1}'); ax.set_ylabel(f'Predicted y{i+1}')
    ax.set_title(f'MLP: y{i+1} (R²={r2_score(Y_true_orig[:,i], Y_pred_orig[:,i]):.3f})'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints', '04_pred_vs_true.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 8. LSTM预测 vs 真实值 ====================
Y_pred_lstm_orig = scaler_Ys.inverse_transform(preds_lstm.reshape(-1,1)).flatten()
Y_true_lstm_orig = scaler_Ys.inverse_transform(truths_s.reshape(-1,1)).flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 散点图
axes[0].scatter(Y_true_lstm_orig, Y_pred_lstm_orig, c='#FF6B6B', s=5, alpha=0.5)
axes[0].plot([Y_true_lstm_orig.min(), Y_true_lstm_orig.max()],
             [Y_true_lstm_orig.min(), Y_true_lstm_orig.max()], 'k--', lw=1, label='Ideal')
axes[0].set_xlabel('True'); axes[0].set_ylabel('Predicted'); axes[0].set_title(f'LSTM Predictions (R²={r2_lstm:.3f})')
axes[0].legend()

# 前100点时间序列
axes[1].plot(Y_true_lstm_orig[:100], label='True', color='#333333', linewidth=1.5)
axes[1].plot(Y_pred_lstm_orig[:100], label='LSTM Pred', color='#4ECDC4', linewidth=1.2, alpha=0.8)
axes[1].set_xlabel('Sample'); axes[1].set_ylabel('Value')
axes[1].set_title('First 100 Predictions vs Ground Truth'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints', '05_lstm_results.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

    mk_cell(nb, """# ==================== 9. 保存模型 ====================
CKPT = os.path.join(ROOT, 'backend', 'regression_model', 'checkpoints')
torch.save({
    'mlp_state_dict': {k:v.cpu().clone() for k,v in mlp_r.state_dict().items()},
    'lstm_state_dict': {k:v.cpu().clone() for k,v in lstm_r.state_dict().items()},
    'r2_mlp': r2_mlp, 'r2_lstm': r2_lstm,
    'scaler_X_mlp': scaler_X, 'scaler_Y_mlp': scaler_Y,
    'scaler_X_lstm': scaler_Xs, 'scaler_Y_lstm': scaler_Ys,
}, os.path.join(CKPT, 'best_model.pth'))
print(f'Saved. MLP R²={r2_mlp:.4f} | LSTM R²={r2_lstm:.4f}')
""")

    save_nb(nb, '04_nonlinear_regression.ipynb')


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print('Building notebooks...')
    build_flower_nb()
    build_titanic_nb()
    build_fashion_nb()
    build_regression_nb()
    print(f'\\nAll 4 notebooks saved to: {OUT_DIR}')
