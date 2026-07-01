import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from joblib import Parallel, delayed

# Feature extraction fonksiyonunu import ediyoruz
from feature_extraction import extract_ecg_features

def add_real_noise(X, snr_db=None, add_wander=True, renormalize=False):
    """
    EKG sinyallerine gürültü ekler ve isteğe bağlı olarak tekrar [0, 1] aralığına normalize eder.
    
    Normalizasyon Tercih Açıklaması:
    - Neden Min-Max Normalizasyon seçildi: Sinyal değerlerini [0, 1] aralığına getirmek için np.clip(signal, 0, 1) 
      kullanılsaydı, baseline wander nedeniyle yukarı veya aşağı kaymış olan EKG dalgalarının en kritik bölgeleri 
      (örneğin R-tepesi veya S-çukuru) kesilecek ve düzleşecekti (hard clipping). Bu durum morfolojik bilgiyi kalıcı 
      olarak yok eder.
      Min-Max Normalizasyonu ise sinyalin genel geometrik şeklini ve oranlarını koruyarak tüm sinyali sıkıştırır ve 
      [0, 1] aralığına sığdırır. Sınıflandırma algoritması için şekil (morfoloji) genlikten çok daha önemli olduğundan 
      Min-Max normalizasyon gerçek hayatta daha gerçekçidir.
    """
    if snr_db is None and not add_wander:
        return X.copy()
        
    X_noisy = X.copy()
    N, signal_len = X.shape
    
    # 1. Baseline Wander ekle
    if add_wander:
        fs = 125
        t = np.arange(signal_len) / fs
        wander = 0.15 * np.sin(2 * np.pi * 0.5 * t)
        X_noisy = X_noisy + wander
        
    # 2. Gaussian Gürültü ekle
    if snr_db is not None:
        sig_powers = np.mean(X_noisy**2, axis=1, keepdims=True)
        snr_linear = 10**(snr_db / 10.0)
        noise_powers = sig_powers / snr_linear
        sigmas = np.sqrt(noise_powers)
        
        noise = np.random.normal(0, 1, size=X.shape) * sigmas
        X_noisy = X_noisy + noise
        
    # 3. Yeniden Normalizasyon (Düzeltme Adımı)
    if renormalize:
        row_mins = np.min(X_noisy, axis=1, keepdims=True)
        row_maxs = np.max(X_noisy, axis=1, keepdims=True)
        X_noisy = (X_noisy - row_mins) / (row_maxs - row_mins + 1e-8)
        
    return X_noisy

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - NOISE ROBUSTNESS TEST V2")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Adım 1] Veriler yükleniyor...")
    train_path = os.path.join("data", "mitbih_train.csv")
    test_path = os.path.join("data", "mitbih_test.csv")
    
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    label_col = train_df.shape[1] - 1
    
    X_train = train_df.iloc[:, :label_col].values
    y_train = train_df.iloc[:, label_col].values.astype(int)
    
    X_test = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    # 2. Train Models
    print("\n[Adım 2] Modeller test için hızlıca eğitiliyor...")
    
    # A) Baseline RF
    print(" -> Baseline RF eğitiliyor...")
    clf_baseline = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_baseline.fit(X_train, y_train)
    
    # B) Feature Engineered RF
    print(" -> Özellik Mühendisliği RF için özellikler çıkarılıyor...")
    X_train_feat = np.array(Parallel(n_jobs=-1)(delayed(extract_ecg_features)(row) for row in X_train))
    
    print(" -> Özellik Mühendisliği RF eğitiliyor...")
    clf_feat_eng = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_feat_eng.fit(X_train_feat, y_train)
    
    # C) 1D CNN
    print(" -> 1D CNN için SMOTE uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(" -> CNN hızlı eğitim için örnekleniyor (25,000 örnek)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train_res), size=25000, replace=False)
    X_train_cnn = X_train_res[sample_indices]
    y_train_cnn = y_train_res[sample_indices]
    
    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1)
    y_train_cnn_cat = to_categorical(y_train_cnn, num_classes=5)
    
    print(" -> 1D CNN eğitiliyor (5 Epoch)...")
    model_cnn = Sequential([
        Input(shape=(187, 1)),
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_cnn.fit(X_train_cnn, y_train_cnn_cat, epochs=5, batch_size=256, verbose=0)
    print(" -> Bütün modeller başarıyla hazırlandı.")
    
    # Scenarios to evaluate
    noise_scenarios = [
        {"name": "Clean", "snr_db": None, "add_wander": False},
        {"name": "20dB", "snr_db": 20.0, "add_wander": True},
        {"name": "15dB", "snr_db": 15.0, "add_wander": True},
        {"name": "10dB", "snr_db": 10.0, "add_wander": True},
        {"name": "5dB", "snr_db": 5.0, "add_wander": True},
        {"name": "0dB", "snr_db": 0.0, "add_wander": True}
    ]
    
    # 3. Run evaluation WITHOUT Renormalization
    print("\n[Adım 3] Gürültü testleri (DÜZELTİLMEMİŞ / NORMALİZASYONSUZ) başlatılıyor...")
    results_unnorm = {"Baseline RF": [], "1D CNN": [], "Feature Eng RF": []}
    
    for scenario in noise_scenarios:
        name = scenario["name"]
        snr_db = scenario["snr_db"]
        add_wander = scenario["add_wander"]
        
        X_test_noisy = add_real_noise(X_test, snr_db=snr_db, add_wander=add_wander, renormalize=False)
        
        # Eval Baseline RF
        y_pred_base = clf_baseline.predict(X_test_noisy)
        results_unnorm["Baseline RF"].append(f1_score(y_test, y_pred_base, average='macro'))
        
        # Eval CNN
        X_test_noisy_cnn = X_test_noisy.reshape(X_test_noisy.shape[0], X_test_noisy.shape[1], 1)
        y_pred_cnn_probs = model_cnn.predict(X_test_noisy_cnn, verbose=0)
        y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
        results_unnorm["1D CNN"].append(f1_score(y_test, y_pred_cnn, average='macro'))
        
        # Eval Feature Eng RF
        X_test_noisy_feat = np.array(Parallel(n_jobs=-1)(delayed(extract_ecg_features)(row) for row in X_test_noisy))
        y_pred_feat = clf_feat_eng.predict(X_test_noisy_feat)
        results_unnorm["Feature Eng RF"].append(f1_score(y_test, y_pred_feat, average='macro'))
        
    # 4. Run evaluation WITH Renormalization (Fixed version)
    print("\n[Adım 4] Gürültü testleri (DÜZELTİLMİŞ / YENİDEN NORMALİZE EDİLMİŞ) başlatılıyor...")
    results_norm = {"Baseline RF": [], "1D CNN": [], "Feature Eng RF": []}
    
    for scenario in noise_scenarios:
        name = scenario["name"]
        snr_db = scenario["snr_db"]
        add_wander = scenario["add_wander"]
        
        # renormalize=True adımı eklendi
        X_test_noisy = add_real_noise(X_test, snr_db=snr_db, add_wander=add_wander, renormalize=True)
        
        # Eval Baseline RF
        y_pred_base = clf_baseline.predict(X_test_noisy)
        f1_base = f1_score(y_test, y_pred_base, average='macro')
        results_norm["Baseline RF"].append(f1_base)
        
        # Eval CNN
        X_test_noisy_cnn = X_test_noisy.reshape(X_test_noisy.shape[0], X_test_noisy.shape[1], 1)
        y_pred_cnn_probs = model_cnn.predict(X_test_noisy_cnn, verbose=0)
        y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
        f1_cnn = f1_score(y_test, y_pred_cnn, average='macro')
        results_norm["1D CNN"].append(f1_cnn)
        
        # Eval Feature Eng RF
        X_test_noisy_feat = np.array(Parallel(n_jobs=-1)(delayed(extract_ecg_features)(row) for row in X_test_noisy))
        y_pred_feat = clf_feat_eng.predict(X_test_noisy_feat)
        f1_feat = f1_score(y_test, y_pred_feat, average='macro')
        results_norm["Feature Eng RF"].append(f1_feat)
        
        print(f" {name:<6} -> RF: {f1_base:.4f} | CNN: {f1_cnn:.4f} | FE RF: {f1_feat:.4f}")
        
    # 5. Plot side-by-side comparison
    print("\n[Adım 5] Karşılaştırmalı yan yana çizgi grafik oluşturuluyor...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    x_labels = [s["name"] for s in noise_scenarios]
    x_indices = range(len(x_labels))
    
    # Left subplot - Unnormalized
    axes[0].plot(x_indices, results_unnorm["Baseline RF"], marker='o', linewidth=2.5, color='blue', label='Baseline RF')
    axes[0].plot(x_indices, results_unnorm["1D CNN"], marker='s', linewidth=2.5, color='orange', label='1D CNN')
    axes[0].plot(x_indices, results_unnorm["Feature Eng RF"], marker='^', linewidth=2.5, color='purple', label='Özellik Mühendisliği RF')
    axes[0].set_title("A: Düzeltilmemiş Gürültü Testi\n(Sinyaller [0, 1] Sınırları Dışına Taşıyor)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Gürültü Seviyesi (Artan Gürültü ->)", fontsize=11)
    axes[0].set_ylabel("Macro F1-Score", fontsize=11)
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(x_labels)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # Right subplot - Normalized
    axes[1].plot(x_indices, results_norm["Baseline RF"], marker='o', linewidth=2.5, color='blue', label='Baseline RF')
    axes[1].plot(x_indices, results_norm["1D CNN"], marker='s', linewidth=2.5, color='orange', label='1D CNN')
    axes[1].plot(x_indices, results_norm["Feature Eng RF"], marker='^', linewidth=2.5, color='purple', label='Özellik Mühendisliği RF')
    axes[1].set_title("B: Düzeltilmiş Gürültü Testi\n(Sinyaller Min-Max ile [0, 1] Arasına Yeniden Normalize Edildi)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Gürültü Seviyesi (Artan Gürültü ->)", fontsize=11)
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(x_labels)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    plt.suptitle("Gürültü Sonrası Yeniden Normalizasyonun Modellere Etkisi Karşılaştırması", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("noise_robustness_comparison_v2.png", dpi=300)
    plt.close()
    print("Grafik 'noise_robustness_comparison_v2.png' olarak kaydedildi.")
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
