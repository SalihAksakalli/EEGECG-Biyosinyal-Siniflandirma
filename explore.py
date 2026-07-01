import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 1. Load Data
    train_path = os.path.join("data", "mitbih_train.csv")
    test_path = os.path.join("data", "mitbih_test.csv")
    
    print(f"\n[Adım 1] Veriler yükleniyor...")
    print(f"Eğitim veri seti yolu: {train_path}")
    print(f"Test veri seti yolu: {test_path}")
    
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    print("\n--- Veri Seti Boyutları ---")
    print(f"Eğitim Veri Seti Boyutu (Satır, Sütun): {train_df.shape}")
    print(f"Test Veri Seti Boyutu (Satır, Sütun): {test_df.shape}")
    
    print("\n--- Sütun Yapısı ---")
    # Her satırda 187 sinyal zaman noktası ve 1 sınıf etiketi vardır.
    print(f"Toplam sütun sayısı: {train_df.shape[1]}")
    print("Sütunlar 0'dan 187'ye kadar indekslenmiştir.")
    print("0 - 186 arası sütunlar: ECG Sinyal Değerleri (Zaman Noktaları)")
    print("187. sütun (Son sütun): Sınıf Etiketi (0, 1, 2, 3, 4)")
    
    print("\n--- Eğitim Verisinin İlk 5 Satırı (İlk 5 ve son sütunlar) ---")
    # İlk 5 satırı ve ilk birkaç ile son sütunlarını gösterelim
    cols_to_show = list(range(5)) + [train_df.shape[1] - 1]
    print(train_df.iloc[:5, cols_to_show])
    
    # Sınıf sütununun adı
    label_col = train_df.shape[1] - 1
    
    # 2. Check for missing values (NaN)
    print(f"\n[Adım 2] Eksik (NaN) değer kontrolü yapılıyor...")
    train_nans = train_df.isnull().sum().sum()
    test_nans = test_df.isnull().sum().sum()
    print(f"Eğitim veri setindeki toplam NaN değer sayısı: {train_nans}")
    print(f"Test veri setindeki toplam NaN değer sayısı: {test_nans}")
    
    # 3. Class Distribution
    print(f"\n[Adım 3] Sınıf dağılımı analiz ediliyor...")
    train_class_counts = train_df[label_col].value_counts().sort_index()
    test_class_counts = test_df[label_col].value_counts().sort_index()
    
    # Sınıf isimleri (MIT-BIH veri seti açıklamasına göre)
    # 0: N (Normal / Normal beat)
    # 1: S (Supraventricular ectopic beat)
    # 2: V (Ventricular ectopic beat)
    # 3: F (Fusion beat)
    # 4: Q (Unknown beat / Unclassifiable beat)
    class_names = {
        0: "Normal (N)",
        1: "Supraventricular Ectopic (S)",
        2: "Ventricular Ectopic (V)",
        3: "Fusion (F)",
        4: "Unknown/Unclassifiable (Q)"
    }
    
    print("\nEğitim Seti Sınıf Dağılımı:")
    for cls, count in train_class_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f" Sınıf {cls} ({class_names[cls]}): {count} örnek ({percentage:.2f}%)")
        
    print("\nTest Seti Sınıf Dağılımı:")
    for cls, count in test_class_counts.items():
        percentage = (count / len(test_df)) * 100
        print(f" Sınıf {cls} ({class_names[cls]}): {count} örnek ({percentage:.2f}%)")
        
    # Bar grafiğini çiz
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in train_class_counts.index], train_class_counts.values, color='royalblue', edgecolor='black')
    plt.title("Eğitim Veri Seti Sınıf Dağılımı (MIT-BIH)", fontsize=14, fontweight='bold')
    plt.xlabel("Kalp Atışı Sınıfları", fontsize=12)
    plt.ylabel("Örnek Sayısı", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    
    # Barların üzerine sayıları ekle
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1000, f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300)
    print("\n[Grafik Kaydedildi] Sınıf dağılımı grafiği 'class_distribution.png' olarak kaydedildi.")
    plt.close()
    
    # 4. Draw random samples from each class
    print(f"\n[Adım 4] Her sınıftan rastgele birer ECG dalga formu görselleştiriliyor...")
    
    # Farklı renkler seçelim
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    for i in range(5):
        # Sınıfa ait tüm satırları filtrele
        class_subset = train_df[train_df[label_col] == i]
        # Rastgele bir satır seç
        random_sample = class_subset.sample(n=1, random_state=None)
        
        # İlk 187 sütun (sinyal değerleri)
        signal = random_sample.iloc[0, :187].values
        
        # Grafiği çiz
        axes[i].plot(signal, label=f"Sınıf {i}: {class_names[i]}", color=colors[i], linewidth=2)
        axes[i].set_title(f"Sınıf {i} Örneği - {class_names[i]}", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Genlik (Amplitüd)", fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend(loc="upper right")
        
    axes[4].set_xlabel("Zaman Noktaları (187 Nokta)", fontsize=12)
    plt.suptitle("Farklı Aritmi Sınıflarından ECG Dalga Formu Örnekleri (MIT-BIH)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("ecg_classes_comparison.png", dpi=300)
    print("[Grafik Kaydedildi] Karşılaştırma grafiği 'ecg_classes_comparison.png' olarak kaydedildi.")
    plt.close()
    
    print("\n" + "=" * 60)
    print("ANALİZ TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
