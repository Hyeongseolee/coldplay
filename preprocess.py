import librosa
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=2.0, n_mels=128):
        self.sr = sample_rate
        self.target_length = int(sample_rate * duration)
        self.n_mels = n_mels

    def bandpass_filter(self, data, lowcut=100, highcut=8000, order=5):
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def normalize(self, data):
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data

    def add_noise(self, data, noise_factor=0.01):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        return augmented_data

    def pitch_shift(self, data, n_steps=2.0):
        return librosa.effects.pitch_shift(y=data, sr=self.sr, n_steps=n_steps)

    def extract_mel_spectrograms_chunked(self, file_path, overlap_sec=1.0, augment=True):
        # 1. Load Audio
        y, _ = librosa.load(file_path, sr=self.sr)
        
        # 2. Band-pass Filter & Peak Normalization
        y_filtered = self.bandpass_filter(y)
        y_norm = self.normalize(y_filtered)
        
        y_variants = [("orig", y_norm)]
        
        if augment:
            y_variants.append(("noise", self.add_noise(y_norm, noise_factor=0.015)))
            y_variants.append(("pitch1", self.pitch_shift(y_norm, n_steps=2.0)))
            y_variants.append(("pitch2", self.pitch_shift(y_norm, n_steps=-2.0)))

        # 3. Chunking (Data Augmentation)
        chunk_length = self.target_length
        step_size = int(self.sr * (2.0 - overlap_sec)) # 1 second overlap by default
        if step_size <= 0:
            step_size = chunk_length
            
        mel_features_list = []
        suffixes_list = []
        
        for suffix, y_arr in y_variants:
            for start_idx in range(0, len(y_arr), step_size):
                end_idx = start_idx + chunk_length
                chunk = y_arr[start_idx:end_idx]
                
                if len(chunk) < chunk_length:
                    # 너무 짧은 꼬투리 데이터는 버림 (0.5초 미만)
                    if len(chunk) < self.sr * 0.5 and len(mel_features_list) > 0:
                        continue
                    # 2초가 안되는 데이터는 빈 공간을 0으로 채움(Padding)
                    chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant')

                # 4. Compute Log Mel-Spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=chunk, sr=self.sr, n_fft=2048, hop_length=512, n_mels=self.n_mels
                )
                mel_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_features_list.append(mel_db)
                suffixes_list.append(suffix)
            
        return mel_features_list, suffixes_list

def main():
    # 현재 작업 디렉토리를 기준으로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "Cold Spray")
    processed_data_dir = os.path.join(base_dir, "processed_data")

    preprocessor = AudioPreprocessor()

    # 모든 .wav 파일 찾기
    wav_files = glob.glob(os.path.join(r"C:\Users\hyeong seo lee\Documents\GitHub\coldplay\audio file", "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print(f"'{raw_data_dir}' 경로에서 .wav 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(wav_files)}개의 .wav 파일을 찾았습니다. 전처리를 시작합니다...")

    for file_path in wav_files:
        try:
            # 멜 스펙트로그램 특징 추출 (오디오 쪼개기 적용)
            # overlap_sec=1.0 & augment=True
            mel_features_list, suffixes_list = preprocessor.extract_mel_spectrograms_chunked(file_path, overlap_sec=1.0, augment=True)
            
            # 기존 폴더 구조(normal/abnormal 등) 유지하며 출력 경로 생성
            rel_path = os.path.relpath(file_path, raw_data_dir)
            rel_dir = os.path.dirname(rel_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 파일명 끝자리에 따라 센서 타입 분류
            if file_name.endswith('0'):
                sensor_type = 'stethoscope'
            elif file_name.endswith('1'):
                sensor_type = 'microphone'
            else:
                sensor_type = 'other'
            
            out_dir = os.path.join(processed_data_dir, sensor_type, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            
            # 잘라낸 여러 개의 조각들을 각각 개별 파일로 저장
            for i, (mel_features, suffix) in enumerate(zip(mel_features_list, suffixes_list)):
                out_path = os.path.join(out_dir, f"{file_name}_chunk{i:03d}_{suffix}.npy")
                np.save(out_path, mel_features)
                print(f"저장 완료: {out_path}")
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({file_path}): {e}")

    print("모든 데이터의 전처리가 완료되었습니다!")

if __name__ == "__main__":
    main()
