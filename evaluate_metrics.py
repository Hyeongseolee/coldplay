import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import librosa
import joblib

# ==========================================
# 1. Models & Datasets Definitions
# ==========================================
class PreprocessedSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        s_db = np.load(file_path)
        s_db_norm = (s_db - s_db.min()) / (s_db.max() - s_db.min() + 1e-6)
        s_db_norm = (s_db_norm - 0.5) * 2.0
        spec_tensor = torch.tensor(s_db_norm, dtype=torch.float32).unsqueeze(0)
        spec_tensor = spec_tensor.repeat(3, 1, 1) 
        y = torch.tensor(label, dtype=torch.long)
        return spec_tensor, y

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(AudioClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_features=False):
        features = self.features(x)
        features = features.view(features.size(0), -1) 
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

def extract_mfcc_features(file_path, chunk_duration=2.0, overlap_sec=1.0, sr=22050, n_mfcc=20):
    y, _ = librosa.load(file_path, sr=sr)
    chunk_length = int(sr * chunk_duration)
    step_size = int(sr * (chunk_duration - overlap_sec))
    features = []
    for start_idx in range(0, len(y), step_size):
        end_idx = start_idx + chunk_length
        chunk = y[start_idx:end_idx]
        if len(chunk) < chunk_length:
            if len(chunk) < sr * 0.5: continue
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant')
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        features.append(np.hstack([mfcc_mean, mfcc_var]))
    return features

def get_files_and_labels(data_dir):
    files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    labels = []
    for npy_file in files:
        path_lower = npy_file.lower()
        if 'nogas' in path_lower: label = 1
        elif 'jamming' in path_lower: label = 2
        elif 'empty' in path_lower: label = 3
        else: label = 0
        labels.append(label)
    return files, labels

# ==========================================
# 2. Plotting Utilities
# ==========================================
CLASS_NAMES = ['Normal', 'Gas Failure', 'Powder Excess', 'Powder Depletion']

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, title, save_path):
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    n_classes = y_true_bin.shape[1]
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne(features, y_true, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("hsv", 4)
    for i in range(4):
        idx = np.where(y_true == i)[0]
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=[colors[i]], label=CLASS_NAMES[i], alpha=0.7, edgecolors='w', s=50)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 3. Evaluation Pipelines
# ==========================================
def eval_file_level_cnn(base_dir, out_dir, device):
    print("Evaluating Model 1: File-level CNN (13% - True Data Leakage Check)")
    data_dir = os.path.join(base_dir, "processed_data")
    all_files, all_labels = get_files_and_labels(data_dir)
    
    train_groups = ['20231105_220031.896329Z', '20231105_220631.588041Z', '20231105_221100.678851Z', '20231105_221602.995174Z', '20231105_221603.050033Z']
    test_files, test_labels = [], []
    for f, l in zip(all_files, all_labels):
        if not any(tg in f for tg in train_groups):
            test_files.append(f)
            test_labels.append(l)
            
    test_loader = DataLoader(PreprocessedSpectrogramDataset(test_files, test_labels), batch_size=32, shuffle=False)
    
    model = AudioClassifier().to(device)
    model.load_state_dict(torch.load(os.path.join(base_dir, "resnet18_xai_best.pth"), map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred, y_prob, features_list = [], [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, feats = model(inputs, return_features=True)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            features_list.extend(feats.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    features = np.array(features_list)
    
    prefix = "M1_FileLevelCNN"
    plot_confusion_matrix(y_true, y_pred, "M1: File-Level CNN Confusion Matrix", os.path.join(out_dir, f"{prefix}_CM.png"))
    plot_roc_curve(y_true, y_prob, "M1: File-Level CNN ROC Curve", os.path.join(out_dir, f"{prefix}_ROC.png"))
    plot_tsne(features, y_true, "M1: File-Level CNN t-SNE", os.path.join(out_dir, f"{prefix}_tSNE.png"))

def eval_time_split_rf(base_dir, out_dir):
    print("Evaluating Model 2: Time-Split RF (99% - Environment Leakage Check)")
    raw_data_dir = os.path.join(base_dir, "Cold Spray")
    all_wav = glob.glob(os.path.join(raw_data_dir, "**", "*.wav"), recursive=True)
    wav_files = [f for f in all_wav if f.endswith('0.wav')]
    
    y_true, X_test = [], []
    for f in wav_files:
        path_lower = f.lower()
        if 'nogas' in path_lower: label = 1
        elif 'jamming' in path_lower: label = 2
        elif 'empty' in path_lower: label = 3
        else: label = 0
        
        chunk_feats = extract_mfcc_features(f)
        half_idx = len(chunk_feats) // 2
        test_chunks = chunk_feats[half_idx:]
        
        X_test.extend(test_chunks)
        y_true.extend([label] * len(test_chunks))
        
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    model = joblib.load(os.path.join(base_dir, "rf_model.pkl"))
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    prefix = "M2_TimeSplitRF"
    plot_confusion_matrix(y_true, y_pred, "M2: Time-Split RF Confusion Matrix", os.path.join(out_dir, f"{prefix}_CM.png"))
    plot_roc_curve(y_true, y_prob, "M2: Time-Split RF ROC Curve", os.path.join(out_dir, f"{prefix}_ROC.png"))
    # Skip t-SNE for RF, use MFCC features directly for visualization
    plot_tsne(X_test, y_true, "M2: Time-Split RF MFCC t-SNE", os.path.join(out_dir, f"{prefix}_tSNE.png"))

def eval_random_split_cnn(base_dir, out_dir, device):
    print("Evaluating Model 3: Random-Split CNN (91% - Steady-state Philosophy)")
    data_dir = os.path.join(base_dir, "processed_data")
    all_files, all_labels = get_files_and_labels(data_dir)
    
    _, test_files, _, test_labels = train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    test_loader = DataLoader(PreprocessedSpectrogramDataset(test_files, test_labels), batch_size=32, shuffle=False)
    
    model = AudioClassifier().to(device)
    model.load_state_dict(torch.load(os.path.join(base_dir, "cnn_random_split_best.pth"), map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred, y_prob, features_list = [], [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, feats = model(inputs, return_features=True)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            features_list.extend(feats.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    features = np.array(features_list)
    
    prefix = "M3_RandomSplitCNN"
    plot_confusion_matrix(y_true, y_pred, "M3: Random-Split CNN Confusion Matrix", os.path.join(out_dir, f"{prefix}_CM.png"))
    plot_roc_curve(y_true, y_prob, "M3: Random-Split CNN ROC Curve", os.path.join(out_dir, f"{prefix}_ROC.png"))
    plot_tsne(features, y_true, "M3: Random-Split CNN t-SNE", os.path.join(out_dir, f"{prefix}_tSNE.png"))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "evaluation_metrics_plots")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        eval_file_level_cnn(base_dir, out_dir, device)
    except Exception as e:
        print(f"Failed to evaluate Model 1: {e}")
        
    try:
        eval_time_split_rf(base_dir, out_dir)
    except Exception as e:
        print(f"Failed to evaluate Model 2: {e}")
        
    try:
        eval_random_split_cnn(base_dir, out_dir, device)
    except Exception as e:
        print(f"Failed to evaluate Model 3: {e}")
        
    print(f"All evaluation plots have been saved to {out_dir}")

if __name__ == "__main__":
    main()
