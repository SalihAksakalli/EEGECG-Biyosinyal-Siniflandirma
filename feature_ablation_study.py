import os
import time
import pandas as pd
import numpy as np
import scipy.stats
import scipy.fft
import pywt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from joblib import Parallel, delayed

def extract_all_ecg_features(signal):
    """
    Sinyalden Zaman, FFT, ve farklı Wavelet seviyelerindeki tüm özellikleri
    tek seferde çıkarır (verimlilik için).
    """
    # A) Zaman Alanı (9 özellik)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    var_val = np.var(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    ptp_val = max_val - min_val
    energy_val = np.sum(signal**2)
    skew_val = scipy.stats.skew(signal)
    kurt_val = scipy.stats.kurtosis(signal)
    
    # B) Frekans Alanı - FFT (5 özellik)
    fft_vals = np.abs(scipy.fft.fft(signal))
    half_len = len(signal) // 2
    dominant_idx = np.argmax(fft_vals[1:half_len]) + 1
    dominant_amp = fft_vals[dominant_idx]
    
    band_len = (half_len - 1) // 3
    band1 = fft_vals[1 : 1 + band_len]
    band2 = fft_vals[1 + band_len : 1 + 2 * band_len]
    band3 = fft_vals[1 + 2 * band_len : half_len]
    
    e1 = np.sum(band1**2)
    e2 = np.sum(band2**2)
    e3 = np.sum(band3**2)
    total_spectral_energy = e1 + e2 + e3 + 1e-8
    r1 = e1 / total_spectral_energy
    r2 = e2 / total_spectral_energy
    r3 = e3 / total_spectral_energy
    
    # C) Dalgacık Dönüşümleri (Wavelet)
    # L2 Wavelet (3 özellik)
    coeffs_l2 = pywt.wavedec(signal, 'db4', level=2)
    w_l2 = [np.sum(c**2) for c in coeffs_l2]
    
    # L4 Wavelet (5 özellik)
    coeffs_l4 = pywt.wavedec(signal, 'db4', level=4)
    w_l4 = [np.sum(c**2) for c in coeffs_l4]
    
    # L6 Wavelet (7 özellik)
    coeffs_l6 = pywt.wavedec(signal, 'db4', level=6)
    w_l6 = [np.sum(c**2) for c in coeffs_l6]
    
    # Birleştirilmiş vektör: Toplam 29 özellik
    return [
        mean_val, std_val, var_val, max_val, min_val, ptp_val, energy_val, skew_val, kurt_val, # 0-8
        float(dominant_idx), dominant_amp, r1, r2, r3,                                         # 9-13
        w_l2[0], w_l2[1], w_l2[2],                                                             # 14-16
        w_l4[0], w_l4[1], w_l4[2], w_l4[3], w_l4[4],                                           # 17-21
        w_l6[0], w_l6[1], w_l6[2], w_l6[3], w_l6[4], w_l6[5], w_l6[6]                          # 22-28
    ]

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - FEATURE ABLATION STUDY")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Adım 1] Veriler yükleniyor...")
    train_path = os.path.join("data", "mitbih_train.csv")
    test_path = os.path.join("data", "mitbih_test.csv")
    
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    label_col = train_df.shape[1] - 1
    
    X_train_raw = train_df.iloc[:, :label_col].values
    y_train = train_df.iloc[:, label_col].values.astype(int)
    
    X_test_raw = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    # 2. Extract Comprehensive Features
    print("\n[Adım 2] Kapsamlı özellik çıkarımı yapılıyor (Tüm boyutlar)...")
    start_time = time.time()
    X_train_full = np.array(Parallel(n_jobs=-1)(delayed(extract_all_ecg_features)(row) for row in X_train_raw))
    X_test_full = np.array(Parallel(n_jobs=-1)(delayed(extract_all_ecg_features)(row) for row in X_test_raw))
    end_time = time.time()
    print(f"Kapsamlı özellik çıkarımı tamamlandı. Geçen süre: {end_time - start_time:.2f} saniye.")
    
    # Deney senaryoları tanımlanıyor
    # Slices listesi, X_train_full özellik matrisinden hangi sütunları alacağımızı gösterir.
    scenarios = [
        {
            "id": 1,
            "name": "Sadece Zaman Alanı",
            "indices": list(range(9))
        },
        {
            "id": 2,
            "name": "Sadece FFT (Frekans)",
            "indices": list(range(9, 14))
        },
        {
            "id": 3,
            "name": "Sadece Wavelet (L4)",
            "indices": list(range(17, 22))
        },
        {
            "id": 4,
            "name": "Zaman + FFT",
            "indices": list(range(14))
        },
        {
            "id": 5,
            "name": "Zaman + Wavelet (L4)",
            "indices": list(range(9)) + list(range(17, 22))
        },
        {
            "id": 6,
            "name": "Tüm Özellikler (Zaman+FFT+Wavelet L4)",
            "indices": list(range(14)) + list(range(17, 22))
        },
        {
            "id": 7,
            "name": "Sadece Wavelet (L2)",
            "indices": list(range(14, 17))
        },
        {
            "id": 8,
            "name": "Sadece Wavelet (L6)",
            "indices": list(range(22, 29))
        }
    ]
    
    ablation_results = []
    
    # 3. Training and Evaluation loop for scenarios
    print("\n[Adım 3] Ablation senaryoları eğitiliyor ve değerlendiriliyor...")
    for s in scenarios:
        name = s["name"]
        indices = s["indices"]
        num_features = len(indices)
        
        # Sütunları seç
        X_tr = X_train_full[:, indices]
        X_te = X_test_full[:, indices]
        
        # Modeli eğit
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        
        t_start = time.time()
        clf.fit(X_tr, y_train)
        t_end = time.time()
        train_time = t_end - t_start
        
        # Tahmin et ve metrikleri hesapla
        y_pred = clf.predict(X_te)
        
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        # Sınıf bazlı F1 skorları
        f1_scores_class = f1_score(y_test, y_pred, average=None)
        f1_s = f1_scores_class[1] # Sınıf 1 (Supraventricular)
        f1_f = f1_scores_class[3] # Sınıf 3 (Fusion)
        
        print(f" Senaryo {s['id']}: {name:<40} | Özellik: {num_features:>2} | Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Süre: {train_time:.2f}s")
        
        ablation_results.append({
            "Senaryo": name,
            "Ozellik_Sayisi": num_features,
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Train_Time_Sec": train_time,
            "F1_S": f1_s,
            "F1_F": f1_f
        })
        
    # 4. Save results to CSV
    df_results = pd.DataFrame(ablation_results)
    df_results.to_csv("ablation_results.csv", index=False)
    print("\nSonuçlar 'ablation_results.csv' dosyasına kaydedildi.")
    
    # 5. Plot performance vs feature count
    print("\n[Adım 4] Özellik sayısı vs Performans grafiği çiziliyor...")
    plt.figure(figsize=(10, 6))
    
    # Grafikte senaryoları çizelim
    # Wavelet derinlik etkisini ayrı bir renkle göstermek için senaryoları ayırabiliriz, 
    # ya da hepsini bir scatter/line şeklinde çizebiliriz.
    for r in ablation_results:
        plt.scatter(r["Ozellik_Sayisi"], r["Macro_F1"], color='crimson', s=100, zorder=5)
        plt.text(r["Ozellik_Sayisi"] + 0.3, r["Macro_F1"], r["Senaryo"], fontsize=9, va='center', clip_on=True)
        
    plt.title("Özellik Sayısı ile Model Performansı (Macro F1) İlişkisi", fontsize=14, fontweight='bold')
    plt.xlabel("Özellik Sayısı (Boyut)", fontsize=12)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, 32])
    plt.tight_layout()
    plt.savefig("ablation_feature_count_vs_performance.png", dpi=300)
    plt.close()
    print("Grafik 'ablation_feature_count_vs_performance.png' olarak kaydedildi.")
    
    print("\n" + "=" * 60)
    print("ABLASYON ANALİZİ TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
