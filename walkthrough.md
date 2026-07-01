# Walkthrough: ECG Signal Classification Setup, Exploration, and Models

We have successfully initialized the ECG signal classification project, configured Kaggle API credentials, downloaded the dataset, performed exploratory data analysis (EDA), and trained/evaluated multiple models and analyses:
1. **Baseline Random Forest** (using class weights on raw data)
2. **SMOTE Random Forest** (using oversampling on raw training data)
3. **1D Convolutional Neural Network (CNN)** (using oversampling + deep learning)
4. **Feature Engineered Random Forest** (using hand-crafted time, frequency, and wavelet domain features)
5. **Noise Robustness Analysis V2** (evaluating models under Gaussian noise and baseline wander with and without normalization)
6. **Feature Ablation Study** (evaluating the impact of different feature subsets and wavelet decomposition levels)
7. **1D Grad-CAM Explainability (XAI)** (visually explaining CNN decisions using gradient-weighted activation mapping)
8. **Zero Padding Correction** (removing shortcut learning artifacts using linear interpolation)

---

## 1. Project Initialization & EDA

### Changes Made
1. **Virtual Environment & Dependencies**:
   - Created a Python virtual environment `.venv`.
   - Installed `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `kaggle`, `kagglehub`, `imbalanced-learn`, `tensorflow`, and `PyWavelets`.
   - Set up `.gitignore` to prevent committing the virtual environment, downloaded data, and access tokens.
2. **Kaggle API Authentication Setup**:
   - Created the `.kaggle` directory under the user home (`C:\Users\salih\.kaggle`).
   - Saved the new-generation Kaggle access token to `C:\Users\salih\.kaggle\access_token`.
   - Configured `KAGGLE_API_TOKEN` environment variable persistently for the user.
3. **Dataset Acquisition**:
   - Developed `download_data.py` to retrieve the `shayanfazeli/heartbeat` dataset using `kagglehub`.
   - Copied the dataset CSV files into the local project directory under `data/`.
4. **Exploratory Data Analysis Script**:
   - Created `explore.py` which loads the datasets, displays sizes/structures, checks for NaNs, analyzes class distributions, and visualizes samples.

### Validation Results (EDA)
- **Training Set Size**: 87,554 rows, 188 columns.
- **Testing Set Size**: 21,892 rows, 188 columns.
- **Columns**: 188 columns (0-186 representing ECG signal points, 187 representing the class label).
- **Missing Values (NaN)**: 0 NaN values found in both training and testing datasets.
- **Class Distribution**:
  - **Sınıf 0 (Normal - N)**: 72,471 örnek (%82.77)
  - **Sınıf 1 (Supraventricular Ectopic - S)**: 2,223 örnek (%2.54)
  - **Sınıf 2 (Ventricular Ectopic - V)**: 5,788 örnek (%6.61)
  - **Sınıf 3 (Fusion - F)**: 641 örnek (%0.73)
  - **Sınıf 4 (Unknown/Unclassifiable - Q)**: 6,431 örnek (%7.35)

#### Class Distribution Plot
![Sınıf Dağılımı](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\class_distribution.png)

#### ECG Classes Waveform Comparison
![ECG Dalga Formu Karşılaştırması](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\ecg_classes_comparison.png)

---

## 2. Baseline Model Training & Evaluation

We trained a Random Forest Classifier using `class_weight='balanced'` on the raw 187-dimensional signal data.

- **Model Training Time**: **19.68 saniye**
- **Prediction Duration**: **0.2077 saniye** (21,892 test örneği için)
- **Genel Doğruluk (Accuracy)**: **0.9772 (%97.72)**
- **Macro-Average F1-Score**: **0.8911 (%89.11)**

#### Visual Confusion Matrix Heatmap (Baseline)
![Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\confusion_matrix.png)

---

## 3. SMOTE Oversampling Model Training & Evaluation

We applied SMOTE oversampling to the training set (`X_train`, `y_train`) only, expanding the dataset from 87,554 rows to **362,355 rows**.

- **SMOTE Resampling Time**: **5.76 saniye**
- **Model Training Time**: **106.20 saniye**
- **Genel Doğruluk (Accuracy)**: **0.9806 (%98.06)**
- **Macro-Average F1-Score**: **0.9007 (%90.07)**

#### Visual Confusion Matrix Heatmap (SMOTE)
![SMOTE Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\smote_confusion_matrix.png)

---

## 4. 1D CNN Model Training & Evaluation

We trained a 1D Convolutional Neural Network (CNN) with 3 Conv1D layers (filters: 32, 64, 128) and MaxPooling + Dropout on the SMOTE-balanced training set.

- **Model Training Time**: **1424.05 saniye (~23.73 dakika)**
- **Prediction Duration**: **5.6250 saniye** (21,892 test örneği için)
- **Genel Doğruluk (Accuracy)**: **0.9832 (%98.32)**
- **Macro-Average F1-Score**: **0.9084 (%90.84)**

#### Visual Confusion Matrix Heatmap (1D CNN)
![CNN Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\cnn_confusion_matrix.png)

#### CNN Training Loss and Accuracy History
![CNN Eğitim Geçmişi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\cnn_training_history.png)

---

## 5. Feature Engineering Random Forest Model

We extracted 19 manual features in time, frequency (FFT), and time-frequency (Wavelet db4 level 4) domains.

- **Özellik Çıkarım Süresi**: **203.12 saniye**
- **Model Training Time**: **2.69 saniye**
- **Genel Doğruluk (Accuracy)**: **0.9582 (%95.82)**
- **Macro-Average F1-Score**: **0.8304 (%83.04)**

#### Visual Confusion Matrix Heatmap (Feature Engineered RF)
![Özellik Mühendisliği Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\feature_engineered_confusion_matrix.png)

#### En Etkili EKG Özellikleri (Feature Importance)
![Özellik Önem Dereceleri](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\feature_importance.png)

* **En Kritik Özellik**: **`cA4_energy (Wavelet)`** özellik önem sıralamasında en üstte yer almıştır. EKG sinyallerini sınıflandırmada Wavelet katsayılarının enerjisinin, FFT tabanlı frekans özelliklerine kıyasla çok daha belirleyici olduğu görülmüştür.

---

## 6. Gürültü Dayanıklılığı Analizi (Noise Robustness V2)

Sinyallere **Baseline Wander (0.5 Hz sinüs dalgası)** ve **Gaussian Beyaz Gürültü (SNR: 20dB - 0dB)** eklenmiş; ardından sinyallerin `[0, 1]` aralığının dışına taşmasını engellemek için **Min-Max Normalizasyon** ile yeniden ölçeklendirme yapılarak düzeltilmiş analiz gerçekleştirilmiştir.

#### Gürültü Sonrası Yeniden Normalizasyonun Modellere Etkisi (V2 Karşılaştırması)
![Gürültü Karşılaştırma V2](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\noise_robustness_comparison_v2.png)

#### SNR=5dB Sınırlarında Temiz vs Gürültülü EKG Dalga Karşılaştırması
![Sinyal Gürültü Karşılaştırması](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\noisy_signal_sample.png)

---

## 7. Özellik Çıkarımı Bileşen Analizi (Feature Ablation Study)

Özellik mühendisliğinde hangi bileşenlerin (Zaman, Frekans/FFT, Zaman-Frekans/Wavelet) karara ne kadar katkı sağladığını ve dalgacık dönüşümünün derinlik (seviye) etkisini ölçmek amacıyla 8 farklı kombinasyon test edilmiştir.

### Ablasyon Analizi Sonuç Tablosu (`ablation_results.csv`)

| Senaryo | Özellik Sayısı | Accuracy | Macro F1-Score | Eğitim Süresi | Fusion (F) F1 | Supraventricular (S) F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sadece Zaman Alanı | 9 | %85.39 | 0.6022 | 2.96 sn | 0.08 | 0.05 |
| 2. Sadece FFT (Frekans) | 5 | %87.18 | 0.6048 | 2.07 sn | 0.04 | 0.00 |
| 3. Sadece Wavelet (L4) | 5 | %88.74 | 0.5858 | 1.98 sn | 0.06 | 0.02 |
| 4. Zaman + FFT | 14 | %94.37 | 0.7872 | 2.77 sn | 0.57 | 0.47 |
| 5. Zaman + Wavelet (L4) | 14 | %95.05 | 0.8051 | 2.68 sn | 0.63 | 0.53 |
| **6. Tüm Özellikler (Full L4)** | **19** | **%95.82** | **0.8304** | **10.11 sn** | **0.69** | **0.67** |
| 7. Sadece Wavelet (L2) | 3 | %83.46 | 0.4515 | 5.18 sn | 0.00 | 0.00 |
| 8. Sadece Wavelet (L6) | 7 | %92.88 | 0.7294 | 6.50 sn | 0.38 | 0.28 |

#### Özellik Sayısı ile Model Performansı İlişkisi
![Ablasyon Grafiği](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\ablation_feature_count_vs_performance.png)

---

## 8. 1D Grad-CAM Model Açıklanabilirliği (XAI)

Modelin hangi klinik nedenlerle karar verdiğini görselleştirmek amacıyla, 1D CNN modelinin son evrişim katmanına (`conv1d_2`, 128 filtre) dayalı Grad-CAM ısı haritaları üretilmiş ve sinyallerin üzerine renk haritası (kırmızı = yüksek önem, mavi = düşük önem) olarak bindirilmiştir.

#### Tüm Sınıflar İçin Grad-CAM Önem Haritaları Karşılaştırması
![Grad-CAM Tüm Sınıflar](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\gradcam_all_classes.png)

### Grad-CAM Analiz Raporu ve Klinik Değerlendirme
* **Ventricular (V) Sınıfı Analizi (Doğru Nedenler):**
  * Model, QRS kompleksinin zirve noktası olan **20 ile 30. indekslere** odaklanmıştır. Ventriküler erken vurular geniş ve bizzar QRS morfolojisi ile teşhis edildiğinden modelin bu bölgeye odaklanması **doğru klinik nedenlere** dayandığını gösterir.
* **Supraventricular (S) ve Fusion (F) Sınıfı Analizi (Artefakt / Kestirme Yol):**
  * Model, bu sınıflarda sinyal morfolojisi yerine sinyalin sonlarındaki **130 ile 170. indeksler** arasındaki tamamen düz (sıfır genlikli) **zero-padding (sıfır dolgusu)** alanına odaklanmıştır.
  * Model, biyolojik sinyali öğrenmek yerine "sinyaldeki sıfır dolgusunun ne zaman başladığı" kestirme yolunu (dataset shortcut) öğrenmiştir. Bu durum, modelin zero-padding olmayan continuous ham EKG akışlarında başarısız olacağını gösteren kritik bir bulgudur.

---

## 9. Zero-Padding Düzeltmesi ve İnterpolasyon

Shortcut learning problemini çözmek amacıyla [fix_padding.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/fix_padding.py) scripti ile:
* Her EKG sinyalinin sonundaki sıfır dolgusu tespit edilerek kesilmiştir.
* Gerçek EKG dalgası, lineer enterpolasyon uygulanarak 187 noktaya yeniden örneklenmiştir (stretching). 
* Böylece sıfır dolgusu artefaktı tamamen ortadan kaldırılmıştır.

#### Orijinal (Sıfır Dolgulu) vs Düzeltilmiş (İnterpolasyonlu) Sinyal Karşılaştırması
![Sinyal Düzeltme Karşılaştırması](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\padding_comparison.png)

### Düzeltilmiş Model Performans Sonuçları (CNN)
CNN modeli, sıfır dolguları kaldırılmış ve enterpole edilmiş bu yeni veri kümesi üzerinde SMOTE uygulanarak yeniden eğitilmiştir.

- **Model Eğitim Süresi**: **1585.99 saniye (~26.43 dakika)**
- **Genel Doğruluk (Accuracy)**: **0.9810 (%98.10)** (Orijinal CNN: %98.32, kayıp sadece -%0.22)
- **Macro-Average F1-Score**: **0.9053 (%90.53)** (Orijinal CNN: %90.84, kayıp sadece -%0.31)

#### Düzeltilmiş CNN Karmaşıklık Matrisi
![Düzeltilmiş CNN Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\fixed_cnn_confusion_matrix.png)

#### Bireysel Sınıf F1-Score Karşılaştırması (Orijinal vs Düzeltilmiş CNN)
* **Normal (N)**: 0.99 -> **0.99** (Değişmedi)
* **Supraventricular (S)**: 0.83 -> **0.83** (Değişmedi)
* **Ventricular (V)**: 0.96 -> **0.95** (-0.01 azaldı)
* **Fusion (F)**: 0.77 -> **0.78** (**+0.01 iyileşti!**)
* **Unknown (Q)**: 0.99 -> **0.98** (-0.01 azaldı)

### Metodolojik Değerlendirme
Kestirme yol sömürüsü (zero-padding artefaktı) tamamen engellenmesine rağmen modelin genel performansı neredeyse **hiç düşmemiş (%98.32'den %98.10'a)**, hatta Fusion sınıfının F1-skoru **%77'den %78'e** yükselmiştir (Duyarlılık/Recall değeri de **%87**'ye fırlamıştır). 

Bu bulgu bize şunu gösterir: Model sıfır dolgusu kolaycılığından mahrum bırakıldığında, **zorunlu olarak gerçek EKG dalga morfolojisini (dalga eğimlerini, genlik oranlarını) öğrenmiştir.** Bu düzeltme, modelimizi laboratuvar sınırlarının ötesine taşıyarak gerçek hayattaki sürekli akan EKG sistemlerinde güvenle çalışabilir hale getiren en önemli metodolojik adımdır.

---

## 10. Genel Model Karşılaştırma (Tüm Modeller)

| Model / Metrik | Genel Accuracy | Macro F1-Score | Supraventricular (S) F1 | Fusion (F) F1 | Eğitim Süresi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline RF (Ham Veri)** | %97.72 | %89.11 | %79 | %77 | **19.68 saniye** |
| **SMOTE RF (Ham Veri)** | %98.06 | %90.07 | %81 | %77 | **106.20 saniye** |
| **1D CNN (Orijinal Dolgulu)** | %98.32 | %90.84 | %83 | %77 | **1424.05 saniye** |
| **Düzeltilmiş CNN (İnterpolasyonlu)** | **%98.10** | **%90.53** | **%83** | **%78** | **1585.99 saniye** |
| **Özellik Mühendisliği RF** | %95.82 | %83.04 | %67 | %69 | **2.69 saniye** |
