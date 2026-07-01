import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

def remove_padding_and_interpolate(signal, target_len=187):
    """
    Sinyalin sonundaki sıfır dolgusunu (zero-padding) tespit edip çıkarır
    ve kalan gerçek EKG sinyalini enterpolasyonla target_len (187) boyuta uzatır.
    """
    # Sonundan geriye doğru ilk sıfır olmayan elemanın indeksini buluyoruz
    non_zero_indices = np.where(signal != 0.0)[0]
    if len(non_zero_indices) == 0:
        # Eğer tamamı sıfırsa (beklenmeyen durum), sıfır dizisi dön
        return np.zeros(target_len)
        
    last_non_zero_idx = non_zero_indices[-1]
    active_signal = signal[:last_non_zero_idx + 1]
    
    # Zaten dolgu yoksa direkt döndür
    if len(active_signal) == target_len:
        return active_signal
        
    # Orijinal x ekseni ve hedef x ekseni tanımlanarak lineer enterpolasyon uygulanır
    x_original = np.linspace(0, 1, len(active_signal))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_original, active_signal)

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - ZERO PADDING CORRECTION & CNN")
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
    
    # 2. Draw 3 padded samples for comparison
    print("\n[Adım 2] Dolgulu örnekler tespit ediliyor ve görselleştiriliyor...")
    # Sıfır dolgusu barındıran örnekleri bul (aktif sinyal uzunluğu 187'den küçük olanlar)
    padded_indices = []
    for i in range(len(X_train_raw)):
        non_zero_indices = np.where(X_train_raw[i] != 0.0)[0]
        if len(non_zero_indices) < 170: # En az 17 sıfır içerenleri seçelim (fark net görünsün)
            padded_indices.append(i)
            if len(padded_indices) == 3:
                break
                
    if len(padded_indices) < 3:
        # Eğer bulunamazsa ilk 3 örneği al
        padded_indices = [0, 1, 2]
        
    print(f"Görselleştirme için seçilen örnek indeksleri: {padded_indices}")
    
    # Karşılaştırma grafiğini çiz
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, idx in enumerate(padded_indices):
        orig_sig = X_train_raw[idx]
        interp_sig = remove_padding_and_interpolate(orig_sig)
        
        # Orijinal (Sıfır dolgulu)
        axes[i, 0].plot(orig_sig, color='red', linewidth=1.8)
        axes[i, 0].set_title(f"Örnek {idx} - Orijinal (Sıfır Dolgulu)", fontsize=11, fontweight='bold')
        axes[i, 0].set_ylabel("Genlik", fontsize=10)
        axes[i, 0].grid(True, linestyle='--', alpha=0.5)
        
        # Düzeltilmiş (İnterpolasyonlu)
        axes[i, 1].plot(interp_sig, color='blue', linewidth=1.8)
        axes[i, 1].set_title(f"Örnek {idx} - Düzeltilmiş (İnterpolasyonlu)", fontsize=11, fontweight='bold')
        axes[i, 1].grid(True, linestyle='--', alpha=0.5)
        
    axes[2, 0].set_xlabel("Zaman Noktaları (187)", fontsize=11)
    axes[2, 1].set_xlabel("Zaman Noktaları (187)", fontsize=11)
    plt.suptitle("Sıfır Dolgulu (Zero-Padding) ve İnterpolasyonlu EKG Sinyalleri Karşılaştırması", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("padding_comparison.png", dpi=300)
    plt.close()
    print("Karşılaştırma grafiği 'padding_comparison.png' olarak kaydedildi.")
    
    # 3. Apply transformation on all datasets
    print("\n[Adım 3] Tüm veri kümelerindeki sinyallerden dolgular çıkarılıyor ve enterpole ediliyor...")
    start_time = time.time()
    X_train_clean = np.array(Parallel(n_jobs=-1)(delayed(remove_padding_and_interpolate)(row) for row in X_train_raw))
    X_test_clean = np.array(Parallel(n_jobs=-1)(delayed(remove_padding_and_interpolate)(row) for row in X_test_raw))
    end_time = time.time()
    print(f"Dönüşüm tamamlandı. Geçen süre: {end_time - start_time:.2f} saniye.")
    
    # 4. Apply SMOTE (Only on Training Set!)
    print("\n[Adım 4] İnterpolasyonlu eğitim verisine SMOTE uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train)
    print(f"SMOTE sonrası eğitim veri boyutu: {X_train_res.shape[0]}")
    
    # Data reshaping for Conv1D and Label Categorization
    X_train_res = X_train_res.reshape(X_train_res.shape[0], X_train_res.shape[1], 1)
    X_test_clean = X_test_clean.reshape(X_test_clean.shape[0], X_test_clean.shape[1], 1)
    
    y_train_cat = to_categorical(y_train_res, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)
    
    # 5. Build and Train CNN
    print("\n[Adım 5] CNN modeli yeni veri kümesi üzerinde yeniden eğitiliyor...")
    model = Sequential([
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
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    start_train_time = time.time()
    # Batch size 1024 yapılarak eğitim hızlandırılmıştır.
    history = model.fit(
        X_train_res, y_train_cat,
        epochs=30,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    end_train_time = time.time()
    
    train_duration = end_train_time - start_train_time
    print(f"\nModel eğitimi tamamlandı. Geçen süre: {train_duration:.2f} saniye.")
    
    # 6. Evaluation
    print("\n[Adım 6] Test kümesinde değerlendirme yapılıyor...")
    y_pred_probs = model.predict(X_test_clean, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification Report
    class_names = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
    print("\nSınıflandırma Raporu:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Save Confusion Matrix
    print("Karmaşıklık Matrisi kaydediliyor...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "S", "V", "F", "Q"])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title("Düzeltilmiş CNN (Zero-Padding Kaldırılmış) - Karmaşıklık Matrisi", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fixed_cnn_confusion_matrix.png", dpi=300)
    plt.close()
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nGenel Doğruluk (Accuracy)       : {accuracy:.4f} (%{accuracy*100:.2f})")
    print(f"Macro-Average F1-Score          : {macro_f1:.4f} (%{macro_f1*100:.2f})")
    
    print("\n" + "=" * 60)
    print("DÜZELTME VE DEĞERLENDİRME TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
