import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - BASELINE MODEL TRAINING")
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
    
    print(f"Eğitim özellik boyutu: {X_train.shape}, Etiket boyutu: {y_train.shape}")
    print(f"Test özellik boyutu: {X_test.shape}, Etiket boyutu: {y_test.shape}")
    
    # 3. Model Training
    print("\n[Adım 3] Random Forest Classifier modeli eğitiliyor...")
    # class_weight='balanced' parametresi sınıf dengesizliğini telafi etmek için kullanılır.
    # n_jobs=-1 tüm CPU çekirdeklerini kullanarak eğitimi hızlandırır.
    # random_state=42 tekrarlanabilirliği sağlar.
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    
    start_train_time = time.time()
    clf.fit(X_train, y_train)
    end_train_time = time.time()
    
    train_duration = end_train_time - start_train_time
    print(f"Model eğitimi tamamlandı. Geçen süre: {train_duration:.2f} saniye.")
    
    # 4. Model Prediction
    print("\n[Adım 4] Test verisi üzerinde tahmin yapılıyor...")
    start_pred_time = time.time()
    y_pred = clf.predict(X_test)
    end_pred_time = time.time()
    
    pred_duration = end_pred_time - start_pred_time
    print(f"Tahmin süresi: {pred_duration:.4f} saniye ({len(X_test)} örnek için).")
    print(f"Örnek başına ortalama tahmin süresi: {(pred_duration / len(X_test)) * 1000:.4f} milisaniye.")
    
    # 5. Evaluation
    print("\n[Adım 5] Model performansı değerlendiriliyor...")
    
    # Classification Report
    class_names = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
    print("\nSınıflandırma Raporu:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Numerical Confusion Matrix
    print("Sayısal Karmaşıklık Matrisi (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visual Confusion Matrix (Heatmap)
    print("\nKarmaşıklık Matrisi görselleştiriliyor ve 'confusion_matrix.png' olarak kaydediliyor...")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "S", "V", "F", "Q"])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title("Random Forest Baseline Model - Karmaşıklık Matrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    
    # Accuracy vs Macro-average F1-score
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n--- Önemli Metrikler ---")
    print(f"Genel Doğruluk (Accuracy)       : {accuracy:.4f} (%{accuracy*100:.2f})")
    print(f"Macro-Average F1-Score          : {macro_f1:.4f} (%{macro_f1*100:.2f})")
    
    # Neden ikisi arasındaki farkın önemli olduğuyla ilgili açıklama
    """
    ACIKLAMA:
    Sınıf dengesizliği (class imbalance) olan veri setlerinde Genel Doğruluk (Accuracy) tek başına 
    yanıltıcı bir metriktir. Bizim veri setimizde normal atımların oranı %82.76 civarındadır. 
    Hiçbir şey öğrenmeyen bir model bile her şeye normal atım derse %82.76 doğruluk elde eder.
    
    Macro-Average F1-score ise her sınıfın F1-skorunun (precision ve recall dengesi) aritmetik 
    ortalamasını alır. Bu metrik tüm sınıflara (örnek sayısından bağımsız olarak) eşit ağırlık 
    verdiğinden, azınlık sınıflardaki (örneğin %0.74 olan Fusion sınıfı) düşük performansı 
    hemen yansıtır. Accuracy ile Macro-Average F1-Score arasındaki büyük fark, azınlık 
    sınıfların doğru sınıflandırılıp sınıflandırılamadığını gösteren en iyi göstergedir.
    """

if __name__ == "__main__":
    main()
