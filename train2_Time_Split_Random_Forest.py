import os
import glob
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def get_label(file_path):
    path_lower = file_path.lower()
    if 'nogas' in path_lower: return 1
    elif 'jamming' in path_lower: return 2
    elif 'empty' in path_lower: return 3
    elif 'normal' in path_lower: return 0
    else: return 0

def extract_mfcc_features(file_path, chunk_duration=2.0, overlap_sec=1.0, sr=22050, n_mfcc=20):
    y, _ = librosa.load(file_path, sr=sr)
    chunk_length = int(sr * chunk_duration)
    step_size = int(sr * (chunk_duration - overlap_sec))
    
    features = []
    for start_idx in range(0, len(y), step_size):
        end_idx = start_idx + chunk_length
        chunk = y[start_idx:end_idx]
        
        if len(chunk) < chunk_length:
            if len(chunk) < sr * 0.5:
                continue
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant')
            
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        feature_vector = np.hstack([mfcc_mean, mfcc_var])
        features.append(feature_vector)
        
    return features

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "Cold Spray")
    
    # 청진기(stethoscope) 데이터만 수집 (파일명이 0.wav로 끝남)
    all_wav_files = glob.glob(os.path.join(raw_data_dir, "**", "*.wav"), recursive=True)
    wav_files = [f for f in all_wav_files if f.endswith('0.wav')]
    
    if not wav_files:
        print("No stethoscope .wav files found.")
        return

    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print(f"Extracting MFCC features from {len(wav_files)} STETHOSCOPE WAV files...")
    for f in wav_files:
        label = get_label(f)
        
        # 파일에서 순차적으로(시간순) 특징 추출
        chunk_features = extract_mfcc_features(f)
        
        # 🎯 핵심 변경점: 파일 내 절반 자르기 (Time-series Split)
        # 각 파일의 앞 50%는 Train, 뒤 50%는 Test로 할당
        half_idx = len(chunk_features) // 2
        
        train_chunks = chunk_features[:half_idx]
        test_chunks = chunk_features[half_idx:]
        
        X_train.extend(train_chunks)
        y_train.extend([label] * len(train_chunks))
        
        X_test.extend(test_chunks)
        y_test.extend([label] * len(test_chunks))
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Data Split - Train: {len(X_train)} chunks, Test: {len(X_test)} chunks")
    
    print("Training Random Forest Classifier on STETHOSCOPE ONLY (Time-Split)...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # 모델 저장 (웹 UI용)
    model_path = os.path.join(base_dir, "rf_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Feature Importance 시각화 (XAI)
    importances = rf_model.feature_importances_
    # 20 MFCC means + 20 MFCC vars = 40 features
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances (MFCC Mean & Variance)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices, rotation=90, fontsize=8)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "feature_importance.png"), dpi=300)
    plt.close()
    print("Feature importance plot saved to feature_importance.png")
    
    print("Evaluating Test Set...")
    y_pred = rf_model.predict(X_test)
    
    target_names = ['Normal', 'Gas Failure', 'Powder Excess', 'Powder Depletion']
    print("\n==================================================")
    print(" [ STETHOSCOPE: TIME-SPLIT ENVIRONMENT LEAKAGE ]")
    print("==================================================")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
