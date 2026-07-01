import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter

def main():
    print("=" * 60)
    print("ECG SIGNAL CLASSIFICATION - SMOTE MODEL TRAINING")
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
    
    # 3. Apply SMOTE
    print("\n[Adım 3] Eğitim verisine SMOTE uygulanıyor...")
    
    # Sınıf isimleri
    class_names = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
    
    # SMOTE öncesi sınıf dağılımı
    print("\nSMOTE Öncesi Eğitim Kümesi Sınıf Dağılımı:")
    dist_before = Counter(y_train)
    for cls in sorted(dist_before.keys()):
        print(f" Sınıf {cls} ({class_names[cls]}): {dist_before[cls]} örnek")
        
    """
    ÖNEMLİ KURAL (VERİ SIZINTISI - DATA LEAKAGE UYARISI):
    SMOTE gibi aşırı örnekleme (oversampling) yöntemleri SADECE eğitim verisine (X_train, y_train) uygulanmalıdır.
    Test verisi (X_test, y_test) tamamen dokunulmamış, orijinal haliyle bırakılmalıdır. 
    Eğer SMOTE test verisine de uygulanırsa, sentetik olarak türetilmiş ve eğitim verisine çok benzeyen örnekler 
    test kümesine sızar. Bu durum modelin başarısını (özellikle genelleştirme yeteneğini) yapay olarak yüksek 
    gösterir (Overoptimistic evaluation) fakat gerçek dünyada model başarısız olur.
    """
    
    # SMOTE tanımla ve uygula
    # random_state=42 tekrarlanabilirliği sağlar.
    smote = SMOTE(random_state=42)
    
    start_smote_time = time.time()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    end_smote_time = time.time()
    
    print(f"SMOTE işlemi tamamlandı. Geçen süre: {end_smote_time - start_smote_time:.2f} saniye.")
    
    # SMOTE sonrası sınıf dağılımı
    print("\nSMOTE Sonrası Eğitim Kümesi Sınıf Dağılımı:")
    dist_after = Counter(y_train_res)
    for cls in sorted(dist_after.keys()):
        print(f" Sınıf {cls} ({class_names[cls]}): {dist_after[cls]} örnek")
        
    print(f"Toplam eğitim veri boyutu (Öncesi -> Sonrası): {X_train.shape[0]} -> {X_train_res.shape[0]}")
    
    # 4. Model Training
    print("\n[Adım 4] Random Forest Classifier modeli eğitiliyor...")
    # Veri setimiz SMOTE ile dengelendiği için bu sefer class_weight='balanced' parametresini kullanmıyoruz.
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    start_train_time = time.time()
    clf.fit(X_train_res, y_train_res)
    end_train_time = time.time()
    
    train_duration = end_train_time - start_train_time
    print(f"Model eğitimi tamamlandı. Geçen süre: {train_duration:.2f} saniye.")
    
    # 5. Model Prediction
    print("\n[Adım 5] Orijinal (Dokunulmamış) Test verisi üzerinde tahmin yapılıyor...")
    start_pred_time = time.time()
    y_pred = clf.predict(X_test)
    end_pred_time = time.time()
    
    pred_duration = end_pred_time - start_pred_time
    print(f"Tahmin süresi: {pred_duration:.4f} saniye ({len(X_test)} örnek için).")
    print(f"Örnek başına ortalama tahmin süresi: {(pred_duration / len(X_test)) * 1000:.4f} milisaniye.")
    
    # 6. Evaluation
    print("\n[Adım 6] Model performansı değerlendiriliyor...")
    
    # Classification Report
    print("\nSınıflandırma Raporu:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Numerical Confusion Matrix
    print("Sayısal Karmaşıklık Matrisi (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visual Confusion Matrix (Heatmap)
    print("\nKarmaşıklık Matrisi görselleştiriliyor ve 'smote_confusion_matrix.png' olarak kaydediliyor...")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "S", "V", "F", "Q"])
    disp.plot(cmap=plt.cm.Greens, values_format='d', ax=plt.gca())
    plt.title("Random Forest SMOTE Model - Karmaşıklık Matrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("smote_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Accuracy & Macro F1-score
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
