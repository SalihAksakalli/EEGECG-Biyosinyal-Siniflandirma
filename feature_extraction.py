import os
import time
import pandas as pd
import numpy as np
import scipy.stats
import scipy.fft
import pywt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

def extract_ecg_features(signal):
    # A) Zaman Alanı İstatistiksel Özellikler
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    var_val = np.var(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    ptp_val = max_val - min_val
    energy_val = np.sum(signal**2)
    skew_val = scipy.stats.skew(signal)
    kurt_val = scipy.stats.kurtosis(signal)
    
    # B) Frekans Alanı Özellikleri (FFT ile)
    fft_vals = np.abs(scipy.fft.fft(signal))
    half_len = len(signal) // 2
    # DC bileşeni (indeks 0) hariç tutularak baskın frekans bulunur
    dominant_idx = np.argmax(fft_vals[1:half_len]) + 1
    dominant_amp = fft_vals[dominant_idx]
    
    # Spektral enerjinin frekans bantlarına dağılımı (DC hariç 3 eşit banda bölüyoruz)
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
    
    # C) Zaman-Frekans Özellikleri (Dalgacık Dönüşümü - Wavelet)
    # db4 (Daubechies 4) dalgacığı kullanılarak 4 seviyeli ayrıştırma gerçekleştirilir.
    # Bu işlem sonucunda [cA4, cD4, cD3, cD2, cD1] katsayıları döner.
    # cA4: Yaklaşıklık (Aproximation - Düşük Frekans/Genel Eğilim)
    # cD4-cD1: Detay (Detail - Yüksek Frekans/Hızlı Değişimler) katsayılarıdır.
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    cA4_energy = np.sum(coeffs[0]**2)
    cD4_energy = np.sum(coeffs[1]**2)
    cD3_energy = np.sum(coeffs[2]**2)
    cD2_energy = np.sum(coeffs[3]**2)
    cD1_energy = np.sum(coeffs[4]**2)
    
    # Neden dalgacık dönüşümü (Wavelet) Fourier'den (FFT) daha uygundur?
    """
    WAVELET TRANSFORMATION VS FFT FOR ECG:
    Fourier Dönüşümü (FFT) sinyalin durağan (stationary) olduğunu varsayar. Ancak EKG gibi biyosinyaller 
    durağan değildir; zaman içinde ani ve geçici değişimler (aritmi, QRS kompleksi vb.) barındırır.
    FFT ile frekans bileşenlerini çok hassas bulabiliriz fakat bu frekansların *ne zaman* gerçekleştiğini 
    tamamen kaybederiz (zaman çözünürlüğü yoktur).
    Dalgacık Dönüşümü (WT) ise ölçeklenip kaydırılabilen dalgacıklar kullanarak hem zaman hem frekans çözünürlüğünü 
    aynı anda sunar. Yüksek frekanslı ani değişimler (R-tepeleri gibi) için iyi zaman çözünürlüğü, düşük frekanslı 
    yavaş dalgalanmalar için iyi frekans çözünürlüğü sağlar. Bu yönüyle geçici EKG anomalilerini yakalamak için 
    FFT'ye kıyasla çok daha uygundur.
    """
    
    # Toplam 19 adet anlamlı özellik birleştirilir
    return [
        mean_val, std_val, var_val, max_val, min_val, ptp_val, energy_val, skew_val, kurt_val,
        float(dominant_idx), dominant_amp, r1, r2, r3,
        cA4_energy, cD4_energy, cD3_energy, cD2_energy, cD1_energy
    ]

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - FEATURE ENGINEERING & RF MODEL")
    print("=" * 60)
    
    # 1. Load Data
    train_path = os.path.join("data", "mitbih_train.csv")
    test_path = os.path.join("data", "mitbih_test.csv")
    
    print("\n[Adım 1] Veriler yükleniyor...")
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    label_col = train_df.shape[1] - 1
    
    X_train_raw = train_df.iloc[:, :label_col].values
    y_train = train_df.iloc[:, label_col].values.astype(int)
    
    X_test_raw = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    # 2. Extract Features
    print("\n[Adım 2] Sinyallerden özellik çıkarımı yapılıyor...")
    
    start_feat_time = time.time()
    X_train_feat = np.array([extract_ecg_features(row) for row in X_train_raw])
    X_test_feat = np.array([extract_ecg_features(row) for row in X_test_raw])
    end_feat_time = time.time()
    
    print(f"Özellik çıkarımı tamamlandı. Geçen süre: {end_feat_time - start_feat_time:.2f} saniye.")
    print(f"Yeni Eğitim Özellik Şekli: {X_train_feat.shape} (187 sütundan 19 özellik sütununa indirgendi)")
    print(f"Yeni Test Özellik Şekli: {X_test_feat.shape}")
    
    # 3. Model Training
    print("\n[Adım 3] Random Forest Classifier modeli yeni özelliklerle eğitiliyor...")
    # Sınıf dengesizliğini dengelemek için class_weight='balanced' kullanıyoruz.
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    
    start_train_time = time.time()
    clf.fit(X_train_feat, y_train)
    end_train_time = time.time()
    
    train_duration = end_train_time - start_train_time
    print(f"Model eğitimi tamamlandı. Geçen süre: {train_duration:.2f} saniye.")
    
    # 4. Model Prediction
    print("\n[Adım 4] Test verisi üzerinde tahmin yapılıyor...")
    start_pred_time = time.time()
    y_pred = clf.predict(X_test_feat)
    end_pred_time = time.time()
    
    pred_duration = end_pred_time - start_pred_time
    print(f"Tahmin süresi: {pred_duration:.4f} saniye.")
    
    # 5. Model Evaluation
    print("\n[Adım 5] Model performansı değerlendiriliyor...")
    
    # Classification Report
    class_names = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
    print("\nSınıflandırma Raporu:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Confusion Matrix
    print("Sayısal Karmaşıklık Matrisi:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save Confusion Matrix Heatmap
    print("\nKarmaşıklık Matrisi görselleştiriliyor ve 'feature_engineered_confusion_matrix.png' olarak kaydediliyor...")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "S", "V", "F", "Q"])
    disp.plot(cmap=plt.cm.Purples, values_format='d', ax=plt.gca())
    plt.title("Özellik Mühendisliği RF - Karmaşıklık Matrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("feature_engineered_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Accuracy and Macro F1
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Genel Doğruluk (Accuracy)       : {accuracy:.4f} (%{accuracy*100:.2f})")
    print(f"Macro-Average F1-Score          : {macro_f1:.4f} (%{macro_f1*100:.2f})")
    
    # 6. Feature Importances
    print("\n[Adım 6] Kararda en etkili özellikler analiz ediliyor...")
    feature_names = [
        "mean", "std", "variance", "max", "min", "peak-to-peak", "energy", "skewness", "kurtosis",
        "dominant_freq_idx", "dominant_freq_amp", "spectral_band1_ratio", "spectral_band2_ratio", "spectral_band3_ratio",
        "cA4_energy (Wavelet)", "cD4_energy (Wavelet)", "cD3_energy (Wavelet)", "cD2_energy (Wavelet)", "cD1_energy (Wavelet)"
    ]
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Özellik önem derecelerini çiz
    plt.figure(figsize=(12, 6))
    plt.title("EKG Karar Sürecinde En Etkili Özellikler (Feature Importance)", fontsize=14, fontweight='bold')
    plt.bar(range(len(feature_names)), importances[indices], color='purple', align="center", edgecolor='black')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
    plt.xlim([-1, len(feature_names)])
    plt.ylabel("Önem Derecesi", fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()
    print("Grafik 'feature_importance.png' olarak kaydedildi.")
    
    print("\n" + "=" * 60)
    print("PROSES VE EĞİTİM TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
