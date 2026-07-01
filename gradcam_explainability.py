import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

def make_gradcam_heatmap(model, signal_array, last_conv_layer_name, pred_index=None):
    """
    1D sinyaller için Grad-CAM ısı haritası üretir.
    
    Grad-CAM Mantığı:
    - Modelin karar verirken hangi öznitelik haritalarına (feature maps) güvendiğini anlamak için, 
      hedef sınıfın skoru (y_c) ile son evrişim (Conv1D) katmanının aktivasyonlarının (A_k) gradyanları hesaplanır.
    - Bu gradyanlar, her bir filtrenin/kanalın hedef sınıf için ne kadar "önemli" olduğunu belirlemek üzere 
      zamansal boyut boyunca ortalanır (Global Average Pooling).
    - Elde edilen önem ağırlıklarıyla evrişim çıktısı ağırlıklandırılıp toplanır. 
      Sadece sınıf skorunu pozitif yönde etkileyen kısımları almak için ReLU uygulanır.
    - Böylece modelin kararlarını doğrudan etkileyen "gradyan x aktivasyon" önemi (ısı haritası) ortaya çıkar.
    """
    # 1. Manual forward pass to track gradients using tf.GradientTape
    x = tf.convert_to_tensor(signal_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs = None
        current_tensor = x
        
        # Pass the input through the sequential layers one by one
        for layer in model.layers:
            current_tensor = layer(current_tensor)
            if layer.name == last_conv_layer_name:
                conv_outputs = current_tensor
                # Watch the conv layer output to compute gradients with respect to it
                tape.watch(conv_outputs)
                
        # The final output is predictions
        predictions = current_tensor
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 2. Compute the gradient of the class score with respect to conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # 3. Average the gradients along the temporal axis (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # 4. Weight the feature maps and sum them
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Sadece pozitif katkıları almak için ReLU (maximum 0) ve 0-1 normalizasyonu
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - CNN GRAD-CAM EXPLAINABILITY")
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
    
    # 2. Train CNN
    # Gerekli gradyan akışını sağlayabilmek için CNN modelimizi burada hızlıca (5 epoch) eğitiyoruz.
    print("\n[Adım 2] CNN modeli hızlıca eğitiliyor (5 Epoch)...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Zaman kazanmak için 25.000 örnek seçiyoruz
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train_res), size=25000, replace=False)
    X_train_cnn = X_train_res[sample_indices]
    y_train_cnn = y_train_res[sample_indices]
    
    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1)
    y_train_cnn_cat = to_categorical(y_train_cnn, num_classes=5)
    
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
    
    # Son Conv1D katmanının adını dinamik olarak bul
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv_layer_name = layer.name
            break
    print(f"Hedef son Conv1D katmanı: {last_conv_layer_name}")
    
    # 3. Find Correct Predictions for Each Class
    print("\n[Adım 3] Her sınıftan doğru tahmin edilmiş birer örnek aranıyor...")
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_pred_probs = model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    class_samples = {}
    class_names = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusion (F)", 4: "Unknown (Q)"}
    
    for cls in range(5):
        # Gerçek sınıfı 'cls' olan ve modelin doğru bildiği (y_test == cls ve y_pred == cls) indeksleri bul
        correct_indices = np.where((y_test == cls) & (y_pred == cls))[0]
        if len(correct_indices) > 0:
            # Rastgele bir tanesini seçelim (ilkini alıyoruz)
            class_samples[cls] = correct_indices[0]
            print(f" Sınıf {cls} ({class_names[cls]}) için örnek indeks: {correct_indices[0]}")
        else:
            # Doğru tahmin yoksa herhangi birini al
            all_indices = np.where(y_test == cls)[0]
            class_samples[cls] = all_indices[0]
            print(f" Sınıf {cls} için doğru tahmin bulunamadı! İlk örnek alındı, indeks: {all_indices[0]}")
            
    # 4. Generate and Plot Grad-CAM for each class
    print("\n[Adım 4] Her sınıf için Grad-CAM ısı haritaları üretiliyor ve çiziliyor...")
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    
    for cls in range(5):
        idx = class_samples[cls]
        signal = X_test[idx]
        
        # Grad-CAM için girdi boyutunu (1, 187, 1) yapıyoruz
        input_signal = signal.reshape(1, len(signal), 1)
        
        # Isı haritasını hesapla
        cam = make_gradcam_heatmap(model, input_signal, last_conv_layer_name, pred_index=cls)
        
        # Isı haritasını (örn. 18 boyuttan) orijinal sinyal boyutu olan 187'ye interpolate et
        # np.interp yardımıyla zamansal boyutu genişletiyoruz
        heatmap = np.interp(np.linspace(0, len(cam)-1, 187), np.arange(len(cam)), cam)
        
        # Plot EKG
        # Sinyali arka planda gri ince çizgi olarak çiziyoruz
        axes[cls].plot(signal, color='gray', alpha=0.5, linewidth=1.5, label='EKG Sinyali')
        
        # Grad-CAM önem derecesini renkli noktalar olarak sinyalin üzerine bindiriyoruz (Jet haritası ile)
        # Kırmızı: Yüksek Önem, Mavi: Düşük Önem
        sc = axes[cls].scatter(range(187), signal, c=heatmap, cmap='jet', s=20, zorder=5, label='Grad-CAM Önem Derecesi')
        
        axes[cls].set_title(f"Sınıf {cls}: {class_names[cls]} (Örnek İndeks: {idx})", fontsize=12, fontweight='bold')
        axes[cls].set_ylabel("Genlik", fontsize=10)
        axes[cls].grid(True, linestyle='--', alpha=0.5)
        
        # Renk çubuğu ekle
        cbar = fig.colorbar(sc, ax=axes[cls], aspect=40, pad=0.01)
        cbar.set_label('Önem', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
    axes[4].set_xlabel("Zaman Noktaları (187 Nokta)", fontsize=12)
    plt.suptitle("EKG Dalga Sınıfları İçin 1D Grad-CAM Görselleştirmesi", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("gradcam_all_classes.png", dpi=300)
    plt.close()
    print("Grafik 'gradcam_all_classes.png' olarak kaydedildi.")
    
    # 5. Ventricular (V) sınıfının incelenmesi
    # Ventricular (V) atımlarında beklenen durum, genişlemiş ve bizzar QRS kompleksine odaklanılmasıdır.
    print("\n[Adım 5] Ventricular (V) sınıfı Grad-CAM analizi...")
    idx_v = class_samples[2]
    signal_v = X_test[idx_v]
    input_v = signal_v.reshape(1, len(signal_v), 1)
    cam_v = make_gradcam_heatmap(model, input_v, last_conv_layer_name, pred_index=2)
    heatmap_v = np.interp(np.linspace(0, len(cam_v)-1, 187), np.arange(len(cam_v)), cam_v)
    
    # QRS kompleksi genellikle sinyalin orta kısımlarında (örneğin 40-100 indeksleri arasında) yer alır.
    qrs_region = heatmap_v[40:100]
    rest_region = np.concatenate([heatmap_v[:40], heatmap_v[100:]])
    print(f"QRS Bölgesi (40-100) Ortalama Grad-CAM Önemi  : {np.mean(qrs_region):.4f}")
    print(f"Diğer Bölgeler Ortalama Grad-CAM Önemi          : {np.mean(rest_region):.4f}")
    
    print("\n" + "=" * 60)
    print("GÖRSELLEŞTİRME VE ANALİZ TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
