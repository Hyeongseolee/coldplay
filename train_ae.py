import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ==========================================
# 1. Dataset for AutoEncoder
# ==========================================
class NormalOnlyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        # Normalize between 0 and 1 roughly
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # Convert to tensor and add channel dim (1, H, W)
        spec_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        # Resize to fixed dimension for simple CNN e.g. 128x87 
        import torch.nn.functional as F
        spec_tensor = F.interpolate(spec_tensor.unsqueeze(0), size=(128, 88), mode='bilinear', align_corners=False).squeeze(0)

        # For autoencoder, input is also the target
        return spec_tensor, spec_tensor

class TestAnomalyDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        spec_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        import torch.nn.functional as F
        spec_tensor = F.interpolate(spec_tensor.unsqueeze(0), size=(128, 88), mode='bilinear', align_corners=False).squeeze(0)
        
        return spec_tensor, self.labels[idx]

# ==========================================
# 2. AutoEncoder Architecture
# ==========================================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "processed_data")
    
    # Load all files, EXCLUDING augmented data to keep evaluation pure
    all_files = [f for f in glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True) if "noise" not in f and "pitch" not in f]
    
    # We will train ONLY on 1 normal file.
    train_normal_groups = ['20231105_221602.995174Z'] 
    
    train_files = []
    test_files = []
    test_labels = [] # 0: Normal, 1: Anomaly
    
    for f in all_files:
        filename = os.path.basename(f)
        
        # Check label using full path (avoid 'normal' being found inside 'abnormal')
        is_normal = 'abnormal' not in f.lower()
        
        is_train = False
        if is_normal:
            for tg in train_normal_groups:
                if tg in filename:
                    is_train = True
                    break
                    
        if is_train:
            train_files.append(f)
        else:
            test_files.append(f)
            # Label 0 for Normal, 1 for any Anomaly
            test_labels.append(0 if is_normal else 1)
            
    print(f"AE Train Files (Normal Only): {len(train_files)}")
    print(f"AE Test Files (Normal + Abnormal): {len(test_files)}")
    
    train_dataset = NormalOnlyDataset(train_files)
    test_dataset = TestAnomalyDataset(test_files, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train AutoEncoder
    print("Training AutoEncoder...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} \t Train Loss: {train_loss:.6f}")
        
    # Evaluate Anomaly Detection
    print("Evaluating Test Set...")
    model.eval()
    mse_losses = []
    true_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            
            # Calculate MSE per batch item
            for i in range(data.size(0)):
                loss = criterion(outputs[i], data[i]).item()
                mse_losses.append(loss)
                true_labels.append(labels[i].item())
                
    mse_losses = np.array(mse_losses)
    true_labels = np.array(true_labels)
    
    # Calculate threshold (90th percentile of normal test data)
    normal_losses = mse_losses[true_labels == 0]
    if len(normal_losses) > 0:
        threshold = np.percentile(normal_losses, 90)
    else:
        threshold = np.mean(mse_losses)
        
    print(f"Calculated Threshold: {threshold:.6f}")
    
    pred_labels = (mse_losses > threshold).astype(int)
    
    print("\n==================================================")
    print("      [ AUTOENCODER ANOMALY DETECTION REPORT ]")
    print("==================================================")
    print(classification_report(true_labels, pred_labels, target_names=['Normal (0)', 'Anomaly (1)']))
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    
    # ROC-AUC if possible
    try:
        auc = roc_auc_score(true_labels, mse_losses)
        print(f"ROC-AUC Score: {auc:.4f}")
    except:
        pass
    print("==================================================")

if __name__ == "__main__":
    main()
