import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# fix_padding.py'deki ön-işleme fonksiyonunu import ediyoruz
from fix_padding import remove_padding_and_interpolate
# gradcam_explainability.py'deki Grad-CAM hesaplama fonksiyonunu import ediyoruz
from gradcam_explainability import make_gradcam_heatmap

def main():
    print("=" * 60)
    print("GRAD-CAM ON FIXED CNN MODEL (PADDING REMOVED & INTERPOLATED)")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Adım 1] Veriler yükleniyor...")
    train_df = pd.read_csv("data/mitbih_train.csv", header=None)
    test_df = pd.read_csv("data/mitbih_test.csv", header=None)
    
    label_col = train_df.shape[1] - 1
    X_train_raw = train_df.iloc[:, :label_col].values
    y_train = train_df.iloc[:, label_col].values.astype(int)
    X_test_raw = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    # 2. Preprocess (Remove padding & interpolate)
    print("\n[Adım 2] Zero-padding kaldırılıyor ve enterpolasyon uygulanıyor...")
    from joblib import Parallel, delayed
    X_train_clean = np.array(Parallel(n_jobs=-1)(delayed(remove_padding_and_interpolate)(row) for row in X_train_raw))
    X_test_clean = np.array(Parallel(n_jobs=-1)(delayed(remove_padding_and_interpolate)(row) for row in X_test_raw))
    
    # 3. SMOTE & Train CNN
    print("\n[Adım 3] SMOTE uygulanıyor ve CNN hızlıca eğitiliyor (5 Epoch)...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_clean, y_train)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_res), size=25000, replace=False)
    X_train_cnn = X_res[sample_indices].reshape(-1, 187, 1)
    y_train_cnn_cat = to_categorical(y_res[sample_indices], num_classes=5)
    
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train_cnn_cat, epochs=5, batch_size=256, verbose=0)
    print("Model eğitimi tamamlandı.")
    
    last_conv_layer_name = [l.name for l in model.layers if isinstance(l, Conv1D)][-1]
    
    # 4. Find Correct Predictions
    print("\n[Adım 4] Her sınıftan doğru tahmin edilen örnekler aranıyor...")
    X_test_clean_cnn = X_test_clean.reshape(-1, 187, 1)
    y_pred = np.argmax(model.predict(X_test_clean_cnn, verbose=0), axis=1)
    
    class_samples = {}
    class_names = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusion (F)", 4: "Unknown (Q)"}
    
    for cls in range(5):
        correct_indices = np.where((y_test == cls) & (y_pred == cls))[0]
        if len(correct_indices) > 0:
            class_samples[cls] = correct_indices[0]
            print(f" Sınıf {cls} ({class_names[cls]}) için indeks: {correct_indices[0]}")
        else:
            all_indices = np.where(y_test == cls)[0]
            class_samples[cls] = all_indices[0]
            
    # 5. Generate and Plot Grad-CAM for fixed model
    print("\n[Adım 5] Düzeltilmiş model için Grad-CAM haritaları çiziliyor...")
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    
    for cls in range(5):
        idx = class_samples[cls]
        signal = X_test_clean[idx]
        
        input_signal = signal.reshape(1, len(signal), 1)
        cam = make_gradcam_heatmap(model, input_signal, last_conv_layer_name, pred_index=cls)
        heatmap = np.interp(np.linspace(0, len(cam)-1, 187), np.arange(len(cam)), cam)
        
        # Plot EKG in gray
        axes[cls].plot(signal, color='gray', alpha=0.5, linewidth=1.5, label='Düzeltilmiş EKG')
        # Overlay heatmap
        sc = axes[cls].scatter(range(187), signal, c=heatmap, cmap='jet', s=20, zorder=5, label='Grad-CAM Önem')
        
        axes[cls].set_title(f"Düzeltilmiş Model - Sınıf {cls}: {class_names[cls]} (Örnek: {idx})", fontsize=12, fontweight='bold')
        axes[cls].set_ylabel("Genlik", fontsize=10)
        axes[cls].grid(True, linestyle='--', alpha=0.5)
        
        cbar = fig.colorbar(sc, ax=axes[cls], aspect=40, pad=0.01)
        cbar.set_label('Önem', fontsize=8)
        
    axes[4].set_xlabel("Zaman Noktaları (187 Nokta)", fontsize=12)
    plt.suptitle("Düzeltilmiş CNN Modeli İçin 1D Grad-CAM Açıklanabilirlik Görselleştirmesi", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("gradcam_fixed_model.png", dpi=300)
    plt.close()
    print("Grafik 'gradcam_fixed_model.png' olarak kaydedildi.")
    print("\nİŞLEM TAMAMLANDI.")

if __name__ == "__main__":
    main()
