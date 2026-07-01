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
from collections import Counter

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - 1D CNN MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load Data
    train_path = os.path.join("data", "mitbih_train.csv")
    test_path = os.path.join("data", "mitbih_test.csv")
    
    print("\n[Adım 1] Veriler yükleniyor...")
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    # 2. Split Features and Labels
    print("\n[Adım 2] Özellikler ve etiketler ayrıştırılıyor...")
    label_col = train_df.shape[1] - 1
    
    X_train = train_df.iloc[:, :label_col].values
    y_train = train_df.iloc[:, label_col].values.astype(int)
    
    X_test = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    # 3. Apply SMOTE (Only on Training Set!)
    print("\n[Adım 3] Eğitim verisine SMOTE uygulanıyor...")
    # VERİ SIZINTISINI ENGELLEMEK İÇİN: SMOTE sadece eğitim verisine uygulanır.
    # Test verisi kesinlikle orijinal haliyle bırakılır.
    smote = SMOTE(random_state=42)
    start_smote_time = time.time()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    end_smote_time = time.time()
    print(f"SMOTE işlemi tamamlandı. Geçen süre: {end_smote_time - start_smote_time:.2f} saniye.")
    print(f"Eğitim veri boyutu (Öncesi -> Sonrası): {X_train.shape[0]} -> {X_train_res.shape[0]}")
    
    # 4. Reshape data for Conv1D and One-Hot encode labels
    print("\n[Adım 4] Veriler CNN için yeniden şekillendiriliyor...")
    # Conv1D girdi olarak 3D tensor bekler: (Örnek Sayısı, Zaman Adımı/Sinyal Uzunluğu, Kanal Sayısı)
    # Bizim durumumuzda sinyal uzunluğu 187 ve kanal sayısı 1'dir.
    X_train_res = X_train_res.reshape(X_train_res.shape[0], X_train_res.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Etiketleri one-hot encoded formatına çeviriyoruz. (5 sınıf için)
    y_train_cat = to_categorical(y_train_res, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)
    
    print(f"Eğitim özellikleri şekli: {X_train_res.shape}, Eğitim etiketleri (one-hot) şekli: {y_train_cat.shape}")
    print(f"Test özellikleri şekli: {X_test.shape}, Test etiketleri (one-hot) şekli: {y_test_cat.shape}")
    
    # 5. Build 1D CNN Architecture
    print("\n[Adım 5] 1D CNN Model mimarisi oluşturuluyor...")
    
    """
    1D CNN MIMARISI ACIKLAMASI:
    - Input: Sinyal boyutu (187, 1). 187 zaman noktası, 1 kanal.
    - Conv1D Katmanları: Sinyal üzerindeki yerel zamansal örüntüleri (QRS kompleksi, P/T dalgaları gibi 
      kalp atışına özgü geometrik şekilleri) yakalamak için evrişim (convolution) işlemini uygular.
      Katmanlar ilerledikçe filtre sayısının artması (32 -> 64 -> 128) modelin basitten karmaşığa 
      daha üst düzey özellikleri öğrenmesini sağlar.
    - MaxPooling1D Katmanları: Evrişimden çıkan haritaların boyutunu yarı yarıya azaltır (downsampling). 
      Bu işlem parametre sayısını ve hesaplama maliyetini düşürür, aşırı öğrenmeyi (overfitting) 
      azaltır ve yerel ötelemelere karşı duyarsızlık (translation invariance) sağlar.
    - Dropout Katmanları: Belirtilen oranda (örn. %30) nöronu her eğitim adımında rastgele devre dışı bırakır.
      Bu sayede nöronların birbirine aşırı bağımlı hale gelmesi (co-adaptation) engellenir ve model 
      daha genel, sağlam özellikler öğrenmeye zorlanarak overfitting önlenir.
    - Flatten: Çok boyutlu çıktıları düzleştirerek (vektör haline getirerek) tam bağlantılı (Dense) katmana aktarır.
    - Dense Katmanı: Öğrenilen öznitelikleri sınıflandırma için birleştirir.
    - Son Dense Katmanı (Softmax): 5 nöronludur ve her bir sınıf için (0-4) olasılık değerleri üretir.
    """
    
    model = Sequential([
        Input(shape=(187, 1)),
        
        # 1. Evrişim Bloğu
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # 2. Evrişim Bloğu
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # 3. Evrişim Bloğu
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Düzleştirme ve Sınıflandırma
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    
    model.summary()
    
    # 6. Compile and Train Model
    print("\n[Adım 6] Model derleniyor ve eğitiliyor...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early Stopping: Val loss 5 epoch boyunca iyileşmezse eğitimi durdurur ve en iyi ağırlıkları yükler.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    start_train_time = time.time()
    # Batch size 512 seçilerek CPU'da eğitimin hızlı akması sağlanır.
    history = model.fit(
        X_train_res, y_train_cat,
        epochs=30,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    end_train_time = time.time()
    
    train_duration = end_train_time - start_train_time
    print(f"\nModel eğitimi tamamlandı. Geçen süre: {train_duration:.2f} saniye.")
    
    # 7. Plot Training History
    print("\n[Adım 7] Eğitim geçmişi grafiği oluşturuluyor...")
    plt.figure(figsize=(12, 5))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)', color='blue')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)', color='orange')
    plt.title('Model Kayıp Değişimi (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp (Loss)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu (Train Acc)', color='blue')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu (Val Acc)', color='orange')
    plt.title('Model Doğruluk Değişimi (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("cnn_training_history.png", dpi=300)
    plt.close()
    print("Grafik 'cnn_training_history.png' olarak kaydedildi.")
    
    # 8. Model Evaluation
    print("\n[Adım 8] Test verisi üzerinde tahmin yapılıyor...")
    start_pred_time = time.time()
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    end_pred_time = time.time()
    
    pred_duration = end_pred_time - start_pred_time
    print(f"Tahmin süresi: {pred_duration:.4f} saniye ({len(X_test)} örnek için).")
    
    # Classification Report
    class_names_short = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
    print("\nSınıflandırma Raporu:")
    report = classification_report(y_test, y_pred, target_names=class_names_short)
    print(report)
    
    # Confusion Matrix
    print("Sayısal Karmaşıklık Matrisi (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save Confusion Matrix Heatmap
    print("\nKarmaşıklık Matrisi görselleştiriliyor ve 'cnn_confusion_matrix.png' olarak kaydediliyor...")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "S", "V", "F", "Q"])
    disp.plot(cmap=plt.cm.Oranges, values_format='d', ax=plt.gca())
    plt.title("1D CNN Model - Karmaşıklık Matrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Accuracy & Macro F1
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n--- Önemli Metrikler ---")
    print(f"Genel Doğruluk (Accuracy)       : {accuracy:.4f} (%{accuracy*100:.2f})")
    print(f"Macro-Average F1-Score          : {macro_f1:.4f} (%{macro_f1*100:.2f})")
    
    print("\n" + "=" * 60)
    print("ANALİZ VE EĞİTİM TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    main()
