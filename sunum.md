# Trendyol Arama Alaka Düzeyi (Search Relevance) Sistemi Sunumu

Bu sunum dokümanı, geliştirdiğimiz e-ticaret arama alaka düzeyi tahminleme pipeline'ının mimarisini, veri akışını ve temel prensiplerini basit ve anlaşılır bir şekilde özetlemektedir.

---

## 🎯 1. Problem ve Amaç Nedir?

Bir kullanıcı arama çubuğuna bir sorgu (query) yazdığında (Örn: *siyah spor ayakkabı*), listelenen ürünlerin bu sorguyla ne kadar alakalı olduğunu tahmin etmek istiyoruz.

*   **Girdi:** `(Sorgu, Ürün Kartı)` çifti.
*   **Çıktı:** Ürün alakalı mı (`1`) yoksa alakasız mı (`0`)?
*   **Değerlendirme Kriteri:** Kaggle Macro F1 Skoru.

---

## 🧩 2. En Büyük Teknik Zorluk: "Sadece Pozitifler Var!"

Eğitim veri setinde (`training_pairs.csv`) sadece alaka düzeyi **kesin olarak 1 olan (başarılı arama-satış eşleşmeleri)** bulunmaktadır. Sistemde gerçek alakasız örnek etiketleri (`0`) mevcut değildir.

Bu durum literatürde **Positive-Unlabeled (PU) Learning** olarak geçer. Modeli eğitebilmek için alakasız örnekleri (negatif veri) akıllı stratejilerle bizim üretmemiz gerekir.

---

## ⚙️ 3. Sistemin Çalışma Mantığı (Adım Adım Pipeline)

Sistemimiz 6 temel aşamadan oluşan bir boru hattıdır (pipeline):

```text
  1. Veri Okuma & attribute ayrıştırma
         ↓
  2. Query Hash Split (Veri sızıntısı engelleme)
         ↓
  3. Çok Stratejili Negatif Örnekleme (Synthetic Negative Sampling)
         ↓
  4. Probable Positive Filtresi & Güven Skoru
         ↓
  5. XLM-RoBERTa Cross Encoder Eğitimi
         ↓
  6. Eşik Değeri (Threshold) Optimizasyonu & Submission
```

### A. Veri Okuma ve Attribute Ayrıştırma
*   Ürün başlıkları, kategorileri ve markaları birleştirilir.
*   Ürün detayındaki karmaşık attribute verileri (Örn: `renk: siyah, materyal: deri, beden: 42`) temizlenerek yapısal alanlara bölünür.
*   **İki Farklı Metin Alanı Oluşturulur:**
    1.  `model_text`: Türkçe karakterleri korur (Dil modelleri için).
    2.  `index_text`: ASCII karakterlerine dönüştürülür (Fuzzy matching ve kelime aramaları için).

### B. Query-Hash Split (Veri Sızıntısını Önleme)
Sorguların bir kısmı eğitimdeyken bir kısmı testtedir. Modelin daha önce görmediği sorgularda da başarılı olabilmesi için doğrulamayı (Validation) buna uygun yapmalıyız.
*   Sorgular normalize edilip hash'lenir.
*   5-Fold GroupKFold yöntemi ile veri bölünür. Bir sorgunun hiçbir varyasyonu aynı anda hem eğitimde hem de doğrulamada yer alamaz.

### C. Çok Stratejili Negatif Örnekleme
Pozitif örneklerin yanına 7 farklı strateji kullanarak akıllı negatifler yerleştirilir:
1.  **Random (Rastgele):** Rastgele başka bir kategoriden ürün seçilir.
2.  **Cross-Category (Kategori Dışı):** Farklı bir üst kategoriden ürün seçilir.
3.  **Same-Category (Aynı Kategori):** Aynı kategoride olan ama bu sorguyla eşleşmeyen ürün seçilir (Zor negatif).
4.  **Same-Brand (Aynı Marka):** Marka adı eşleşen ama alakasız ürünler seçilir.
5.  **Lexical Hard (Kelime Benzerliği Yüksek):** Ürün başlığında sorgudaki kelimeler geçen ama alakasız olan ürünler seçilir.
6.  **Attribute Conflict (Özellik Çelişkisi):** Sorguda "siyah" istenip üründe "beyaz" yazan çelişkili ürünler seçilir.
7.  **Cross-Query (Sorgular Arası):** Başka sorguların başarılı ürünleri bu sorgu için negatif kabul edilir.

### D. Probable-Positive Filtresi & Güven Skoru
*   **Filtre:** Negatif üretilirken yanlışlıkla alakalı bir ürün seçilirse (Örn: sorgu *iphone kılıf*, üretilen negatif *iphone şeffaf kılıf*), bu ürün filtre tarafından yakalanır ve elenir.
*   **Güven Skoru:** Her negatif tipe bir güven puanı verilir. Kolay elenen negatiflerin eğitimdeki ağırlığı düşürülür veya tamamen atılır.

### E. XLM-RoBERTa Cross Encoder Eğitimi
Sorgu ve ürün birleştirilerek tek bir metin halinde (`[Sorgu] | [Ürün Başlığı] | [Kategori] | [Marka]`) deep learning modeline (`xlm-roberta-base`) beslenir. Model bu iki metin arasındaki semantik alaka düzeyini 0 ile 1 arasında bir olasılık puanı olarak tahmin etmeyi öğrenir.

### F. Threshold Optimizasyonu
Model çıktıları olasılıktır (Örn: `0.64`). Bu olasılığı `0` veya `1` yapabilmek için yerel validation setinde Grid Search yapılarak en yüksek Macro F1 skorunu veren en optimum sınır değeri (Örn: `0.3540`) seçilir.

---

## 📈 4. Doğrulama Senaryoları (Local Validation Scenarios)

Modelin başarısını sadece tek bir genel skorla değil, 5 farklı senaryoda test ederiz:
*   **Easy Mix:** Kolay negatifleri eleme başarısı.
*   **Structural:** Aynı kategori ve markadaki ürünleri ayırt etme başarısı.
*   **Lexical Hard:** Kelime benzerliği tuzağına düşmeme başarısı.
*   **Semantic Hard:** Semantik olarak yakın ama alakasız ürünleri ayırt etme başarısı.
*   **Candidate Set Simulation:** Gerçek arama sonucundaki sıralama simülasyonu.

---

## 🛠️ 5. Teknolojik Altyapı

*   **Dil ve Kütüphaneler:** Python, PyTorch, Hugging Face Transformers, Pandas, Scikit-Learn
*   **Dil Modeli:** `xlm-roberta-base` (Cross-Encoder sequence classification)
*   **Verimlilik Optimizasyonları:**
    *   *Dynamic Padding & Length-Bucket Batching:* Benzer uzunluktaki girdileri gruplayarak GPU üzerindeki dolgu işlemlerini (padding) minimuma indirir, eğitimi %40 hızlandırır.
    *   *Mixed Precision (AMP):* FP16 hassasiyeti kullanarak GPU bellek tüketimini yarıya indirir.
