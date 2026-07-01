import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_real_noise_vectorized(X, snr_db=None, add_wander=True):
    if snr_db is None and not add_wander:
        return X.copy()
        
    X_noisy = X.copy()
    N, signal_len = X.shape
    
    # 1. Baseline Wander
    if add_wander:
        fs = 125
        t = np.arange(signal_len) / fs
        wander = 0.15 * np.sin(2 * np.pi * 0.5 * t)
        X_noisy = X_noisy + wander
        
    # 2. Gaussian Noise
    if snr_db is not None:
        sig_powers = np.mean(X_noisy**2, axis=1, keepdims=True)
        snr_linear = 10**(snr_db / 10.0)
        noise_powers = sig_powers / snr_linear
        sigmas = np.sqrt(noise_powers)
        
        noise = np.random.normal(0, 1, size=X.shape) * sigmas
        X_noisy = X_noisy + noise
        
    return X_noisy

def main():
    print("=" * 60)
    print("ECG SIGNAL - NOISE VISUALIZATION & STATISTICS CHECK")
    print("=" * 60)
    
    # 1. Load Data
    test_path = os.path.join("data", "mitbih_test.csv")
    print(f"Test verisi yükleniyor: {test_path}")
    test_df = pd.read_csv(test_path, header=None)
    
    label_col = test_df.shape[1] - 1
    X_test = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    # 2. Apply Noise at SNR=5dB
    print("\nSNR=5dB (Gaussian + Baseline Wander) gürültü ekleniyor...")
    X_test_noisy = add_real_noise_vectorized(X_test, snr_db=5.0, add_wander=True)
    
    # 3. Print stats comparison for 3 samples
    print("\n--- Örnek Sinyal İstatistikleri (Temiz vs Gürültülü - SNR=5dB) ---")
    class_names = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusion (F)", 4: "Unknown (Q)"}
    
    sample_indices = [0, 1000, 2000] # Farklı sınıflardan 3 örnek seçelim
    
    for idx in sample_indices:
        clean_sig = X_test[idx]
        noisy_sig = X_test_noisy[idx]
        lbl = y_test[idx]
        
        print(f"\n[Örnek İndeks: {idx} | Sınıf: {lbl} ({class_names[lbl]})]")
        print(f"  Temiz Sinyal  -> Min: {clean_sig.min():.4f} | Max: {clean_sig.max():.4f} | Std: {clean_sig.std():.4f}")
        print(f"  Gürültülü (5dB)-> Min: {noisy_sig.min():.4f} | Max: {noisy_sig.max():.4f} | Std: {noisy_sig.std():.4f}")
        
    # 4. Check normalization range (0-1) across the entire noisy test set
    min_noisy_val = X_test_noisy.min()
    max_noisy_val = X_test_noisy.max()
    
    print("\n--- Normalizasyon Kontrolü (Orijinal Aralık: [0, 1]) ---")
    print(f"Gürültülü test kümesindeki minimum değer: {min_noisy_val:.4f}")
    print(f"Gürültülü test kümesindeki maksimum değer: {max_noisy_val:.4f}")
    
    if min_noisy_val < 0.0 or max_noisy_val > 1.0:
        print("\n[UYARI] Gürültülü sinyaller orijinal [0, 1] aralığının dışına taşmıştır!")
        print("  - Neden: Baseline wander (0.15 genlikli sinüs) ve Gaussian gürültü eklenmesi sinyal değerlerini")
        print("    0'ın altına düşürmüş ve 1'in üzerine çıkarmıştır.")
        print("  - Çözüm: Eğer model kesin olarak 0-1 arası girdi bekliyorsa sinyallerin yeniden normalize edilmesi gerekir.")
        print("    Ancak bizim modellerimiz (Random Forest ve 1D CNN) bu taşmaları belirli ölçüde tolere edebilmektedir.")
    else:
        print("\nGürültülü sinyaller [0, 1] aralığında kalmıştır.")
        
    # 5. Plot 3 samples overlaid
    print("\nSinyal grafikleri çiziliyor ve 'noisy_signal_sample.png' olarak kaydediliyor...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i, idx in enumerate(sample_indices):
        axes[i].plot(X_test[idx], label="Temiz Sinyal (Clean)", color='blue', linewidth=2)
        axes[i].plot(X_test_noisy[idx], label="Gürültülü Sinyal (Noisy, SNR=5dB)", color='orange', alpha=0.8, linestyle='--', linewidth=1.5)
        
        axes[i].set_title(f"Örnek {idx} - Sınıf: {class_names[y_test[idx]]}", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Genlik", fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend(loc="upper right")
        
    axes[2].set_xlabel("Zaman Noktaları (187 Nokta)", fontsize=12)
    plt.suptitle("EKG Sinyalleri Temiz vs Gürültülü Karşılaştırması (SNR=5dB + Baseline Wander)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("noisy_signal_sample.png", dpi=300)
    plt.close()
    print("Grafik kaydedildi.")
    
    print("\n" + "=" * 60)
    print("KONTROL VE GÖRSELLEŞTİRME TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
