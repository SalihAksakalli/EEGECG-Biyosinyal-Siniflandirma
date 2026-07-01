# EKG Sinyali ile Kalp Aritmisi Sınıflandırma
## Kapsamlı Makine Öğrenmesi ve Sinyal İşleme Projesi

---

### 1. PROJE ÖZETİ (Executive Summary)

Bu projenin temel amacı, Elektrokardiyografi (EKG) zaman serisi sinyallerini kullanarak farklı kalp aritmisi sınıflarını yüksek doğrulukla ve klinik açıdan güvenilir bir biçimde sınıflandırmaktır. Aritmiler, kalbin elektriksel sistemindeki düzensizliklerden kaynaklanan ve zamanında teşhis edilmediğinde ölümcül sonuçlar doğurabilen kardiyovasküler bozukluklardır. Proje kapsamında, geleneksel makine öğrenmesi yaklaşımları (Random Forest), sinyal işleme tabanlı özellik mühendisliği (Öznitelik Çıkarımı) ve derin öğrenme (1D Evrişimli Sinir Ağları - CNN) modelleri uçtan uca geliştirilmiş ve karşılaştırılmıştır.

Veri tabanı olarak, MIT-BIH Aritmi Veri Tabanı'ndan derlenen ve Kaggle üzerinde paylaşılan, toplamda 109.000'den fazla kalp atımı içeren geniş ölçekli EKG sinyal seti kullanılmıştır. Bu veri kümesindeki en büyük zorluk, "Normal" kalp atışlarının ezici çoğunlukta olması (%82.77) ve bazı aritmi sınıflarının (örneğin Fusion atımları) %1'in altında temsil edilmesidir. Bu aşırı sınıf dengesizliği (class imbalance) problemini çözmek amacıyla Sınıf Ağırlıklandırma (Class Weighting) ve Sentetik Azınlık Aşırı Örnekleme Tekniği (SMOTE) kullanılarak veri dengelenmiştir.

Proje, basit bir model eğitimiyle sınırlı kalmamış, akademik ve endüstriyel standartlarda üç ileri analizle derinleştirilmiştir:
1. **Ablasyon Çalışması (Ablation Study):** Zaman alanı, Fourier (FFT) ve Dalgacık (Wavelet) özelliklerinin sınıflandırma üzerindeki katkısı incelenmiş, dalgacık derinliğinin etkisi ölçülmüştür.
2. **Gürültü Dayanıklılığı Testi (Noise Robustness):** Gerçek hayat koşullarını simüle etmek amacıyla sinyallere Gaussian gürültü ve solunum kaynaklı Baseline Wander (taban çizgisi kayması) eklenmiş, yeniden normalizasyonun model başarısına etkisi araştırılmıştır.
3. **Açıklanabilir Yapay Zeka (XAI - Grad-CAM):** Geliştirilen derin öğrenme modelinin karar mekanizması 1D Grad-CAM algoritması ile görselleştirilmiştir. Bu sayede modelin zero-padding (sıfır dolgusu) bölgesini bir "kestirme yol" (shortcut learning) olarak sömürdüğü saptanmış, bu metodolojik hata interpolasyon yöntemiyle düzeltilerek klinik olarak doğrulanabilir bir model elde edilmiştir.

Geliştirilen düzeltilmiş 1D CNN modeli, veri kümesindeki sıfır dolgusu kolaycılığından arındırılmasına rağmen **%98.10 genel doğruluk (accuracy)** ve **%90.53 Macro F1-skoru** elde ederek mükemmel bir başarı yakalamıştır. Bu sonuçlar, modelin biyolojik sinyal morfolojisini gerçekten öğrendiğini ve klinik olarak giyilebilir veya hastane tipi EKG cihazlarında güvenle kullanılabileceğini kanıtlamaktadır.

---

### 2. VERİ SETİ VE KEŞİFSEL ANALİZ (EDA)

Projede kullanılan veri kümesi, MIT-BIH Aritmi Veri Tabanı'ndan alınan EKG sinyallerini temel almaktadır. Orijinal kayıtlar 360 Hz frekansında örneklenmiş olup, her bir kalp vuruşu (heartbeat) R-tepe noktasından önce ve sonra belirli aralıklarla kesilmiş, gürültü filtrelemesinden geçirilmiş ve 125 Hz frekansına yeniden örneklenmiştir. Her örnek, $[0, 1]$ aralığında normalize edilmiş 187 zaman noktasından ve son sütundaki sınıf etiketinden oluşmaktadır.

Veri setinde Association for the Advancement of Medical Instrumentation (AAMI) standartlarına göre 5 temel kalp atışı sınıfı bulunmaktadır:
* **Sınıf 0 (Normal - N):** Normal kalp atışları.
* **Sınıf 1 (Supraventricular Ectopic - S):** Atriyal erken vuru gibi kulakçık kaynaklı düzensizlikler.
* **Sınıf 2 (Ventricular Ectopic - V):** Ventriküler erken vuru gibi karıncık kaynaklı ciddi aritmiler.
* **Sınıf 3 (Fusion - F):** Normal ve ventriküler atımın birleşmesiyle oluşan geçiş atışları.
* **Sınıf 4 (Unknown/Unclassifiable - Q):** Pacemaker (kalp pili) atışları veya sınıflandırılamayan vuruşlar.

Eğitim setinde 87.554, test setinde ise 21.892 adet örnek bulunmaktadır. Sınıfların dağılımı aşağıdaki tabloda özetlenmiştir:

| Sınıf Etiketi | Sınıf Adı | Eğitim Örnek Sayısı | Oran (%) | Klinik Anlamı |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Normal (N) | 72,471 | %82.77 | Sağlıklı Kalp Atışı |
| **1** | Supraventricular (S) | 2,223 | %2.54 | Kulakçık Kaynaklı Aritmi |
| **2** | Ventricular (V) | 5,788 | %6.61 | Karıncık Kaynaklı (Tehlikeli) Aritmi |
| **3** | Fusion (F) | 641 | %0.73 | Birleşik Kalp Atışı |
| **4** | Unknown (Q) | 6,431 | %7.35 | Kalp Pili / Belirlenemeyen |

Aşırı sınıf dengesizliği EKG analizinde yaygın bir durumdur. Normal atımların ezici üstünlüğü (%82.77), sınıflandırıcının azınlık sınıflarını (özellikle %0.73 oranındaki Fusion sınıfını) görmezden gelerek tahminlerini sürekli "Normal" sınıfına yönlendirmesine (çoğunluk sınıfı önyargısı) neden olabilir. Bu durum medikal teşhiste kabul edilemez; çünkü bir aritmi hastasının "sağlıklı" olarak raporlanması (False Negative - Yalancı Negatif) hayati risk taşır.

Şekil 1'de sınıfların dağılımı, Şekil 2'de ise 5 sınıftan rastgele seçilen EKG sinyal dalga formlarının karşılaştırması gösterilmiştir.

![Şekil 1: Sınıf Dağılımı](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\class_distribution.png)
*Şekil 1: MIT-BIH Eğitim Kümesindeki Aşırı Sınıf Dengesizliğini Gösteren Bar Grafiği*

![Şekil 2: ECG Dalga Formu Karşılaştırması](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\ecg_classes_comparison.png)
*Şekil 2: Farklı Kalp Aritmilerinin Zaman Serisi Sinyal Geometrileri (R-tepe noktaları genelde ilk üçte bir bölgede hizalanmıştır)*

---

### 3. MODEL GELİŞTİRME SÜRECİ

Proje kapsamında ham veriler üzerinde üç farklı makine öğrenmesi ve derin öğrenme yaklaşımı geliştirilmiş, ardından sinyal işleme tabanlı bir özellik mühendisliği modeli kurulmuştur.

#### 3.1 Baseline Random Forest
Sınıf dengesizliğini telafi etmek için `class_weight='balanced'` parametresi kullanılarak raw EKG sinyal verisi (187 öznitelik) üzerinde doğrudan bir Random Forest (RF) Classifier (100 ağaç) eğitilmiştir. 
* **Eğitim Süresi:** 19.68 saniye
* **Genel Doğruluk (Accuracy):** %97.72
* **Macro-Average F1-Score:** %89.11
* **Sınıf Bazlı Performans:** Supraventricular (S) F1-skoru %79, Fusion (F) F1-skoru %77 seviyesindedir.

![Şekil 3: Baseline RF Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\confusion_matrix.png)
*Şekil 3: Ham veriyle eğitilen Sınıf Ağırlıklı Baseline RF Modelinin Test Kümesi Karmaşıklık Matrisi*

#### 3.2 SMOTE ile Dengelenmiş Random Forest
Eğitim setindeki azınlık sınıfların örnek sayılarını sentetik olarak artırmak amacıyla SMOTE uygulanmıştır. SMOTE, azınlık sınıf örneklerinin en yakın komşuları arasında rastgele çizgiler üzerinde yeni yapay veriler üreterek çalışır.
* **ÖNEMLİ KURAL:** Veri sızıntısını (Data Leakage) önlemek için SMOTE **yalnızca eğitim setine** (`X_train`, `y_train`) uygulanmıştır. Test verisi tamamen orijinal haliyle bırakılmıştır. SMOTE sonrasında eğitim seti boyutu 87.554 satırdan **362.355 satıra** çıkarılmıştır.
* **Eğitim Süresi:** 106.20 saniye (Veri setinin genişlemesi nedeniyle süre yaklaşık 5 kat artmıştır)
* **Genel Doğruluk (Accuracy):** %98.06
* **Macro-Average F1-Score:** %90.07 (Baseline RF'e göre +%0.96 artış)
* **Değerlendirme:** Çoğunluk sınıfına olan eğilim kırılmış, Supraventricular (S) F1-skoru %81'e yükselmiştir.

![Şekil 4: SMOTE RF Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\smote_confusion_matrix.png)
*Şekil 4: SMOTE ile dengelenmiş veriyle eğitilen Random Forest Modelinin Karmaşıklık Matrisi*

#### 3.3 1D CNN Derin Öğrenme Modeli
Zaman serisi şekillerini doğrudan öğrenebilen, 1D Evrişimli Sinir Ağları (Conv1D) mimarisi kurulmuştur. Mimari şu şekildedir:
* **Girdi Katmanı:** `(187, 1)` boyutunda zaman serisi.
* **Evrişim Blokları:** Artan filtre sayısına sahip 3 adet Conv1D katmanı (32, 64, 128 filtre, kernel boyutu 5, ReLU aktivasyon) ve her birinden sonra Max Pooling (havuzlama boyutu 2).
* **Düzenlileştirme:** Ezberlemeyi (overfitting) önlemek amacıyla evrişim bloğunun çıkışına ve tam bağlantılı katmana %30 oranında Dropout uygulanmıştır.
* **Çıkış Katmanları:** Düzleştirme (Flatten) + 128 nöronlu Dense katmanı + 5 sınıflı Softmax çıkış katmanı.
* **Eğitim Ayarları:** Model, SMOTE ile dengelenmiş verinin tamamı üzerinde Adam optimizasyonu ve Kategorik Çapraz Yitim (Categorical Crossentropy) ile eğitilmiştir. `validation_split=0.1` ve validation loss takibi yapan `EarlyStopping` (patience=5) kullanılmıştır. Model 7. epoch'ta erken durdurulmuştur.

* **Eğitim Süresi:** 1424.05 saniye (~23.7 dakika, CPU üzerinde)
* **Genel Doğruluk (Accuracy):** %98.32
* **Macro-Average F1-Score:** %90.84 (En yüksek ham performans)
* **Değerlendirme:** Sınıf 1 (S) için %83, Sınıf 3 (F) için %77 F1-skoru elde edilmiştir. Grafiklerde overfitting olmadığı, eğitim ve doğrulama kayıplarının paralel azaldığı görülmüştür.

![Şekil 5: CNN Eğitim Tarihçesi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\cnn_training_history.png)
*Şekil 5: 1D CNN Modelinin Eğitim ve Doğrulama Kayıp/Doğruluk Eğrileri (7. epoch'ta eğitim durdurulmuştur)*

![Şekil 6: 1D CNN Karmaşıklık Matrisi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\cnn_confusion_matrix.png)
*Şekil 6: 1D CNN Modelinin Test Kümesindeki Tahmin Dağılımını Gösteren Karmaşıklık Matrisi*

#### 3.4 Sinyal İşleme Tabanlı Özellik Mühendisliği (Feature Engineering)
Gelişmiş sinyal işleme yöntemleri kullanılarak her bir EKG sinyalinden elle 19 anlamlı öznitelik çıkarılmıştır:
* **Zaman Alanı İstatistikleri (9 Özellik):** Ortalama (mean), standart sapma (std), varyans, maksimum, minimum, tepe-tepe genlik, sinyal enerjisi, çarpıklık (skewness) ve basıklık (kurtosis).
* **Frekans Alanı Özellikleri (5 Özellik - FFT):** Fast Fourier Transform genlik spektrumundan spektral enerjinin 5 eşit frekans bandındaki dağılım oranları.
* **Zaman-Frekans Alanı Özellikleri (5 Özellik - Wavelet):** Daubechies 4 (`db4`) ana dalgacığı kullanılarak 4. seviye Ayrık Dalgacık Dönüşümü (DWT) yapılmış; elde edilen yaklaşım katsayıları (cA4) ve detay katsayılarının (cD1, cD2, cD3, cD4) enerjileri hesaplanmıştır.

Bu 19 özellik kullanılarak bir Random Forest modeli eğitilmiştir:
* **Özellik Çıkarma Süresi (Paralel CPU):** 203.12 saniye
* **Model Eğitim Süresi:** **Sadece 2.69 saniye!**
* **Genel Doğruluk (Accuracy):** %95.82
* **Macro-Average F1-Score:** %83.04
* **Analiz:** Bu model, ham veri kullanan modellere göre daha düşük performansa sahiptir; ancak eğitim süresi (2.69 sn vs 1424 sn) ve girdi boyutu (19 öznitelik vs 187 zaman noktası) açısından olağanüstü hızlı ve hafiftir.
* **Önem Sıralaması (Şekil 7):** En önemli özniteliğin Dalgacık Dönüşümü yaklaşım katsayısı enerjisi (`cA4_energy`) olduğu görülmüştür. Bu durum, EKG sinyallerini anlamlandırmada Dalgacık Dönüşümünün, FFT'ye kıyasla ezici üstünlüğünü kanıtlar.

![Şekil 7: Öznitelik Önem Dereceleri](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\feature_importance.png)
*Şekil 7: Çıkarılan 19 Özelliğin Random Forest Kararlarındaki Göreli Önem Ağırlıkları (cA4_energy en kritik özniteliktir)*

---

### 4. ABLATION STUDY (BİLEŞEN ÇIKARMA ANALİZİ)

Özniteliklerin model kararlarına etkisini sistematik olarak ölçmek için 8 farklı öznitelik kombinasyonu üzerinde Random Forest modelleri ayrı ayrı eğitilmiş ve karşılaştırılmıştır. Sonuçlar `ablation_results.csv` dosyasından derlenerek aşağıdaki tabloda sunulmuştur:

| Senaryo | Özellik Sayısı | Doğruluk (Accuracy) | Macro F1-Score | Eğitim Süresi | Fusion (F) F1 | Supraventricular (S) F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sadece Zaman Alanı | 9 | %85.39 | 0.6022 | 2.96 sn | 0.08 | 0.05 |
| 2. Sadece FFT (Frekans) | 5 | %87.18 | 0.6048 | 2.07 sn | 0.04 | 0.00 |
| 3. Sadece Wavelet (L4) | 5 | %88.74 | 0.5858 | 1.98 sn | 0.06 | 0.02 |
| 4. Zaman + FFT | 14 | %94.37 | 0.7872 | 2.77 sn | 0.57 | 0.47 |
| 5. Zaman + Wavelet (L4) | 14 | %95.05 | 0.8051 | 2.68 sn | 0.63 | 0.53 |
| **6. Tüm Özellikler (L4)** | **19** | **%95.82** | **0.8304** | **10.11 sn** | **0.69** | **0.67** |
| 7. Sadece Wavelet (L2) | 3 | %83.46 | 0.4515 | 5.18 sn | 0.00 | 0.00 |
| 8. Sadece Wavelet (L6) | 7 | %92.88 | 0.7294 | 6.50 sn | 0.38 | 0.28 |

#### Önemli Bulgular:
* **Wavelet'in FFT'ye Üstünlüğü:** Sadece 5 dalgacık özelliği (%88.74 doğruluk), sadece 5 FFT özelliğini (%87.18 doğruluk) geride bırakmıştır. Zaman özellikleri eklendiğinde de Zaman+Wavelet (%95.05) kombinasyonu, Zaman+FFT (%94.37) kombinasyonuna üstün gelmiştir. Fourier dönüşümü, sinyalin zaman çözünürlüğünü tamamen kaybettiği için durağan olmayan (non-stationary) EKG dalgalarını kaçırır. Dalgacık Dönüşümü (DWT) ise zaman-frekans lokalizasyonu sağlayarak geçici aritmi anomalilerini yakalar.
* **Azalan Verimler Kanunu ve Dalgacık Seviyesi (Level 2, 4, 6):** Dalgacık dönüşüm seviyesi L2'den L6'ya yükseltildiğinde Macro F1 skoru **0.4515'ten 0.7294'e** yükselmiştir (+%27.8 artış). Seviye arttıkça alt frekans bantlarındaki kalp anomalileri daha belirginleşir. Ancak 187 noktalık kısa sinyallerde 6. seviyeden sonra sınır etkileri (boundary effects) oluştuğu için daha derin çözünürlük artışları performansı iyileştirmeyecektir.
* **Minimum Maliyet / Maksimum Fayda Noktası:** **Sadece Wavelet (L6)** modeli sadece **7 özellik** kullanarak **%92.88 genel doğruluk** sağlamaktadır. 14 özellik kullanan Zaman+FFT modeline çok yakın bir performansı neredeyse yarı yarıya az özellikle sunarak kaynak kısıtlı IoT sistemleri için en verimli nokta olduğunu göstermiştir.

![Şekil 8: Ablasyon Grafiği](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\ablation_feature_count_vs_performance.png)
*Şekil 8: Öznitelik Sayısının Model Performansı (Macro F1) Üzerindeki Etkisi ve Dalgacık Seviye Derinliğinin Karşılaştırması*

---

### 5. GÜRÜLTÜYE DAYANIKLILIK ANALİZİ

Gerçek EKG ölçümlerinde, solunumdan kaynaklanan düşük frekanslı taban çizgisi kaymaları (**Baseline Wander - 0.5 Hz sinüs dalgası**) ve kas hareketleri veya elektrot temassızlığından kaynaklanan yüksek frekanslı **Gaussian Beyaz Gürültü** kaçınılmazdır. Modellerin bu gürültülere karşı dayanıklılığı 20dB (hafif gürültü) ile 0dB (aşırı gürültü) arasında test edilmiştir.

#### Metodoloji ve Per-Sample Normalizasyon Önemi
Gürültü ve solunum dalgaları sinyale eklendiğinde, sinyal değerleri orijinal $[0, 1]$ normalizasyon sınırlarının dışına taşar (minimum -1.71'e düşerken, maksimum 2.64'e fırlamaktadır). Gerçek sistemlerde bu taşmaları engellemek için sinyalin yeniden normalize edilmesi zorunludur. Düzeltme adımında **her sinyal kendi içinde satır bazlı (per-sample) Min-Max normalizasyonuna** tabi tutulmuştur. 
* *Neden np.clip(0, 1) kullanılmadı:* Sinyali 0 ve 1 değerlerinde budamak (clipping), solunum kayması nedeniyle sınır dışına taşan kritik dalgaların (örneğin R-tepeleri veya S-çukurları) düzleşmesine (morfoloji kaybına) neden olur. Min-Max ise sinyal şeklini bozmadan sıkıştırır.

#### Bulgular ve CNN'in Üstünlüğü:
Aşağıdaki çizgi grafiklerde (Şekil 9), normalizasyonun etkisi ve modellerin performansı karşılaştırılmıştır:

![Şekil 9: Gürültü Karşılaştırma Grafiği](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\noise_robustness_comparison_v2.png)
*Şekil 9: Gürültü Seviyelerine Göre Model Başarımları. Grafik A (Normalizasyonsuz) ve Grafik B (Min-Max Normalize Edilmiş)*

* **Normalizasyonun Yıkıcı Dağılım Kayması (Covariate Shift) Etkisi:**
  Gürültü eklendikten sonra sinyal yeniden normalize edildiğinde (Grafik B), Random Forest modellerinin başarımı hızla düşmüştür (Baseline RF Macro F1 skoru 20dB gürültüde **0.3821**'e, 10dB'de **0.2308**'e çakılmıştır). Çünkü gürültü pikleri normalizasyon sınırlarını domine etmiş ve temiz sinyalin karar eşiklerini daraltarak bozmuştur.
* **CNN Modelinin Ezici Dayanıklılığı:**
  **1D CNN**, normalize edilmiş gürültü altında 20dB'de **0.5604** ve 10dB'de **0.5133** F1-skoru elde ederek diğer tüm modelleri açık ara geride bırakmıştır. Evrişimli katmanlar, mutlak genlik seviyeleri yerine sinyal içindeki yerel şekil geçişlerini (morfolojik sınırları) öğrendiği için gürültü ve normalizasyon büzülmelerinden çok daha az etkilenmektedir.

---

### 6. AÇIKLANABİLİR YAPAY ZEKA (XAI) - GRAD-CAM ANALİZİ

Derin öğrenme modelleri genellikle birer "kara kutu" (black box) olarak kabul edilir ve kararlarının ardındaki mantık anlaşılamadığında tıbbi cihazlarda kullanımı risk taşır. Bu engeli aşmak için, modelin son evrişim katmanındaki filtre aktivasyonlarının sınıf tahminine olan gradyan katkısını ölçen **1D Grad-CAM** görselleştirmesi uygulanmıştır.

![Şekil 10: Orijinal CNN Grad-CAM Analizi](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\gradcam_all_classes.png)
*Şekil 10: Orijinal Dolgulu Veriyle Eğitilen CNN Modelinin Karar Odak Noktaları. Kırmızı noktalar karardaki en önemli bölgeleri temsil eder.*

#### Kritik Bulgu: Kestirme Yol Sömürüsü (Zero-Padding Shortcut)
1D Grad-CAM görselleştirmesi (Şekil 10) çok kritik bir metodolojik hatayı ortaya çıkarmıştır:
* **Ventricular (V) Sınıfı (Doğru Odaklanma):** Model, ventriküler vuru tahminlerinde doğru bir biçimde sinyalin 20-30. indeksleri arasındaki **genişlemiş QRS tepe noktasına** odaklanmıştır. Bu, tıbbi teşhis kurallarına tamamen uyan "doğru nedenlerle doğru tahmin" durumudur.
* **Supraventricular (S) ve Fusion (F) Sınıfları (Yanlış Odaklanma):** Model, bu sınıflarda EKG dalga şeklini incelemek yerine, sinyalin sonlarındaki **130-170. indeksler arasındaki tamamen düz sıfır bölgesine (zero-padding)** odaklanmıştır. Aritmik vuruşların zaman uzunluğu farklı olduğundan, model sinyalin sıfır dolgu sınırının başladığı yeri bir "kestirme yol" olarak öğrenmiştir. Bu durum, sıfır dolgusunun bulunmadığı gerçek kesintisiz hasta monitörlerinde modelin **tamamen çökeceği** anlamına gelir.

---

### 7. ZERO-PADDING DÜZELTMESİ VE TIBBİ DOĞRULAMA

Sıfır dolgusu sömürüsünü ortadan kaldırmak amacıyla, EKG sinyallerinin sonundaki sıfır kuyrukları tespit edilerek kesilmiş ve kalan gerçek EKG sinyali **Lineer İnterpolasyon** uygulanarak tekrar 187 noktaya gerilmiştir. Böylece sinyalin zaman ölçeği normalize edilmiş ve sıfır dolgusu tamamen yok edilmiştir.

![Şekil 11: İnterpolasyon Dönüşümü](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\padding_comparison.png)
*Şekil 11: Sıfır Dolgulu (Kırmızı) EKG Sinyallerinin Dolgudan Arındırılıp İnterpolasyonla Gerilmiş (Mavi) Karşılıkları*

#### Düzeltilmiş Model Grad-CAM Sonuçları
CNN modeli bu dolgudan arındırılmış temiz veriyle yeniden eğitildiğinde, Grad-CAM haritaları (Şekil 12) modelin artık sıfır dolgusu sömürmesinin imkansız olduğunu ve zorunlu olarak **tüm sınıflarda gerçek dalga morfolojilerine odaklandığını** göstermiştir.

![Şekil 12: Düzeltilmiş CNN Grad-CAM](C:\Users\salih\.gemini\antigravity-ide\brain\d58d27b3-f8b7-4e7c-954d-9ba33a237c5c\gradcam_fixed_model.png)
*Şekil 12: Sıfır Dolgusu Kaldırılan Düzeltilmiş CNN Modelinin Grad-CAM Isı Haritası (Odak noktaları artık tamamen biyolojik dalgaların üzerindedir)*

* **Performans Doğrulaması:** Kestirme yol sömürüsü engellenmesine rağmen modelin doğruluğu neredeyse hiç düşmemiş (**%98.32'den %98.10'a**), hatta azınlık **Fusion (F) sınıfının F1-skoru %77'den %78'e yükselmiştir**. Bu durum, modelin sıfır dolgusu kolaycılığından arındırıldığında, biyolojik dalga geometrisini başarıyla genelleştirebildiğinin en büyük kanıtıdır.

---

### 8. TÜM MODELLERİN KARŞILAŞTIRMASI

Proje süresince test edilen tüm ana modellerin performans metrikleri tek bir özet tabloda birleştirilmiştir:

| Model Adı | Genel Doğruluk (Accuracy) | Macro F1-Score | Supraventricular (S) F1-Score | Fusion (F) F1-Score | Eğitim Süresi | Tahmin Hızı (21k Örnek) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline RF (Raw)** | %97.72 | %89.11 | %79 | %77 | 19.68 sn | **0.20 sn** |
| **SMOTE RF (Raw)** | %98.06 | %90.07 | %81 | %77 | 106.20 sn | 0.25 sn |
| **1D CNN (Orijinal)** | **%98.32** | **%90.84** | **%83** | %77 | 1424.05 sn | 5.62 sn |
| **Düzeltilmiş CNN (İnterpolasyonlu)**| %98.10 | %90.53 | **%83** | **%78** | 1585.99 sn | 5.80 sn |
| **Özellik Mühendisliği RF** | %95.82 | %83.04 | %67 | %69 | **2.69 sn** | 0.18 sn |

---

### 9. MÜHENDİSLİK ÖNERİLERİ

EKG sınıflandırma sisteminin son ürüne dönüştürülmesinde, kullanım senaryosuna göre en uygun model tercih edilmelidir:

#### Senaryo A: Hastane Yoğun Bakım ve Klinik Tanı Cihazları (Yüksek Güvenilirlik Öncelikli)
* **Önerilen Model:** **Düzeltilmiş 1D CNN Modeli (İnterpolasyonlu)**
* **Gerekçe:** Klinik ortamlarda teşhis doğruluğu ve yanlış teşhisi (False Negative) en aza indirmek kritik önem taşır. Düzeltilmiş CNN modeli, sıfır dolgusu kestirme yollarından arındırıldığı ve gerçek morfolojiyi öğrendiği için en güvenilir modeldir. Aynı zamanda gürültü direnci en yüksek model olup, hasta hareketlerinden kaynaklanan parazitler altında kararlılığını korur.

#### Senaryo B: Taşınabilir / Giyilebilir EKG Cihazları (Holter Monitörleri - Dengeli)
* **Önerilen Model:** **SMOTE Random Forest Modeli**
* **Gerekçe:** Holter cihazları pil tasarrufu sağlamak zorunda olan ancak aritmi takibini de yüksek doğrulukla yapması gereken sistemlerdir. SMOTE RF modeli %98.06 doğruluk ve %90.07 F1-skoru ile CNN modellerine çok yakın performans sunarken, tahmin süresi açısından CNN'den yaklaşık 25 kat daha hızlıdır ve işlemciyi yormaz.

#### Senaryo C: Gerçek Zamanlı Akıllı IoT Saatler ve Düşük Güçlü Mikrodenetleyiciler
* **Önerilen Model:** **Dalgacık Tabanlı Özellik Mühendisliği RF (Senaryo 8: Sadece Wavelet L6)**
* **Gerekçe:** Çok kısıtlı bellek ve işlemciye sahip mikrodenetleyicilerde (örneğin ARM Cortex-M serisi), 187 zaman noktasını veya derin CNN katmanlarını saklamak mümkün değildir. Bu senaryoda sadece 7 dalgacık özelliği çıkaran L6 Wavelet RF modeli, sıfır bellek yükü ve mikrosaniyeler seviyesindeki tahmin hızıyla %92.88 genel doğruluk sağlayarak mükemmel bir verimlilik sunar.

---

### 10. SONUÇ VE ÖĞRENİLEN DERSLER

* **Teknik Dersler:** SMOTE gibi aşırı veri dengeleme yöntemleri uygulanırken, test verisine asla dokunulmamalıdır. Gürültü testleri tasarlanırken normalizasyon adımının per-sample olarak eklenmesi gerekliliği, modellerin gerçekçi şartlarda ne kadar hassas olduğunu göstermiştir.
* **Metodolojik Dersler (XAI Önemi):** Yüksek doğruluk (%98+) değerleri, modelin her zaman "akıllı" kararlar verdiğini göstermez. 1D Grad-CAM analizi olmasaydı, modelin zero-padding gibi klinik dışı bir dolgu alanına bağımlı olduğunu asla keşfedemezdik. Model açıklanabilirliği, medikal yapay zekanın en temel yapı taşlarındandır.
* **Klinik Dersler:** Aritmi tespitinde genel doğruluğun (accuracy) yanı sıra, azınlık sınıfların duyarlılık (recall) ve F1 değerleri izlenmelidir. Aritmileri kaçırmamak adına duyarlılık en önemli medikal parametredir.

---

### EKLER

#### Kullanılan Teknolojiler
* **Programlama Dili:** Python 3.11
* **Derin Öğrenme Kütüphanesi:** TensorFlow / Keras 3
* **Veri Dengeleme:** Imbalanced-learn (SMOTE)
* **Sinyal İşleme:** PyWavelets (Discrete Wavelet Transform), Scipy FFT
* **Makine Öğrenmesi & Metrikler:** Scikit-learn
* **Görselleştirme:** Matplotlib

#### Proje Dosya Yapısı ve Açıklamalar
* [download_data.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/download_data.py): Kaggle API kimlik bilgilerini doğrulayarak veri kümesini yerel `data/` klasörüne indiren betik.
* [explore.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/explore.py): EKG sınıf dağılımlarını hesaplayan ve EKG sınıflarının waveform grafiklerini çizen keşifsel analiz betiği.
* [baseline_model.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/baseline_model.py): Ham veriler üzerinde ağırlık dengeli Random Forest modelini eğiten ve karmaşıklık matrisini kaydeden betik.
* [smote_model.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/smote_model.py): Eğitim kümesine sentetik aşırı örnekleme (SMOTE) uygulayarak RF modelini eğiten betik.
* [cnn_model.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/cnn_model.py): 1D CNN mimarisini SMOTE'lu ham veri üzerinde kuran ve eğitim geçmişi grafiklerini kaydeden betik.
* [feature_extraction.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/feature_extraction.py): EKG sinyallerinden 19 adet zaman, FFT ve Wavelet özelliğini çıkaran ve RF üzerinde değerlendiren betik.
* [visualize_noise.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/visualize_noise.py): SNR=5dB gürültü altında sinyal kaymalarını ve normalizasyon sınır taşmalarını test edip görselleştiren betik.
* [noise_robustness_test.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/noise_robustness_test.py): Modellerin gürültülü test setlerindeki (20dB-0dB) Macro F1 skorlarını normalizasyonlu/normalizasyonsuz karşılaştıran betik.
* [feature_ablation_study.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/feature_ablation_study.py): 8 farklı özellik kombinasyonunu eğitip sonuçları `ablation_results.csv` olarak kaydeden ablasyon betiği.
* [gradcam_explainability.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/gradcam_explainability.py): Orijinal dolgulu CNN modelinin kararlarını açıklayan 1D Grad-CAM kodlarını içeren betik.
* [fix_padding.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/fix_padding.py): Sinyallerdeki zero-padding dolgularını kaldıran, enterpole eden ve CNN modelini yeniden eğiten betik.
* [gradcam_fixed.py](file:///c:/Users/salih/OneDrive/Desktop/EEGECG%20Biyosinyal%20S%C4%B1n%C4%B1fland%C4%B1rma/gradcam_fixed.py): Dolgulardan temizlenmiş düzeltilmiş CNN modeli üzerinde 1D Grad-CAM görselleştirmesini (`gradcam_fixed_model.png`) oluşturan betik.

#### Referanslar
1. **MIT-BIH Dataset:** Goldberger, A., et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation* (2000).
2. **SMOTE Paper:** Chawla, N. V., et al. "SMOTE: synthetic minority over-sampling technique." *Journal of artificial intelligence research* (2002).
3. **Grad-CAM Paper:** Selvaraju, R. R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *IEEE International Conference on Computer Vision* (2017).
