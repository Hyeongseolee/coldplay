import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. Custom Dataset Definition
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
        
        # SpecAugment
        if self.is_train:
            _, F, T = spec_tensor.shape
            
            # Frequency masking (max 15 bins)
            freq_mask_param = 15
            f_len = torch.randint(0, freq_mask_param, (1,)).item()
            if f_len > 0:
                f_0 = torch.randint(0, F - f_len, (1,)).item()
                spec_tensor[:, f_0:f_0 + f_len, :] = 0
                
            # Time masking (max 15 frames)
            time_mask_param = 15
            t_len = torch.randint(0, time_mask_param, (1,)).item()
            if t_len > 0:
                t_0 = torch.randint(0, T - t_len, (1,)).item()
                spec_tensor[:, :, t_0:t_0 + t_len] = 0

        spec_tensor = spec_tensor.repeat(3, 1, 1) 
        
        y = torch.tensor(label, dtype=torch.long)
        return spec_tensor, y

# ==========================================
# 2. Shallow CNN Model (LightAudioCNN)
# ==========================================
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

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, device='cpu'):
    model.to(device)
    print(f"Starting training for {epochs} epochs on {device}...")
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct.double() / len(val_loader.dataset)
        
        if epoch_val_acc >= best_acc:
            best_acc = epoch_val_acc
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_random_split_best.pth")
            torch.save(model.state_dict(), save_path)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
    return history

def get_files_and_labels(data_dir):
    files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    labels = []
    for npy_file in files:
        path_lower = npy_file.lower()
        if 'nogas' in path_lower: label = 1
        elif 'jamming' in path_lower: label = 2
        elif 'empty' in path_lower: label = 3
        elif 'normal' in path_lower: label = 0
        else: label = 0
        labels.append(label)
    return files, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "processed_data")
    
    print(f"Loading data from: {data_dir}")
    all_files, all_labels = get_files_and_labels(data_dir)
    if not all_files:
        print("Data not found.")
        return

    # 1. Random Split (80/20) - 오버랩/환경누수 전면 허용!
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"Random Split - Train: {len(train_files)}, Test: {len(test_files)}")

    train_dataset = PreprocessedSpectrogramDataset(train_files, train_labels, is_train=True)
    test_dataset = PreprocessedSpectrogramDataset(test_files, test_labels, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = AudioClassifier(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 2. Train the model
    epochs = 30
    history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs, device=device)
    
    # 3. Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss History (Random Split)')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy History (Random Split)')
    
    plt.savefig(os.path.join(base_dir, "cnn_training_history.png"))
    plt.close()

    # 4. Final Evaluation
    save_path = os.path.join(base_dir, "cnn_random_split_best.pth")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    target_names = ['Normal', 'Gas Failure', 'Powder Excess', 'Powder Depletion']
    print("\n==================================================")
    print(" [ CNN PERFORMANCE: RANDOM SPLIT (OVERLAP ALLOWED) ]")
    print("==================================================")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    print(f"Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
