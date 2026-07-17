# Trendyol E-Commerce Search Relevance
# Son Teknik Revizyonlar ve Teste Geçiş Planı — v4

> [!IMPORTANT]
> Bu doküman, mevcut `Uygulama Planı v3` üzerinde yapılması gereken son teknik düzeltmeleri içerir.
>
> Bu revizyonlardan sonra mimari dondurulacak ve geliştirme/test aşamasına geçilecektir.
>
> Temel hedef:
>
> ```text
> Güvenilir validation
> → yüksek güvenli negatifler
> → erken baseline submission
> → kontrollü iterasyon
> → private leaderboard'a genelleme
> ```

---

# 1. Temel Problem Tanımı

Bu yarışma klasik bir binary classification problemi değildir.

Elimizde:

- Güvenilir pozitif çiftler vardır.
- Gerçek negatif etiketler yoktur.
- Test setinde hem pozitif hem negatif çiftler bulunur.
- Eğitim negatifleri sentetik olarak üretilecektir.
- Train ve test sorguları örtüşmemektedir.
- Ürünlerin büyük bölümü train ve test arasında örtüşmektedir.

Bu nedenle problem şu bileşenlerin birleşimidir:

```text
Search Relevance
+
Positive-Unlabeled Learning
+
Synthetic Negative Sampling
+
Hard Negative Mining
+
Cross-Encoder Classification
+
Threshold Optimization
```

---

# 2. Public Leaderboard Tek Karar Verici Olmamalıdır

Mevcut plandaki aşağıdaki yaklaşım risklidir:

```text
Her değişiklik yalnızca public leaderboard skorunu artırıyorsa korunur.
```

Public leaderboard, test setinin yalnızca belirli bir alt kümesini temsil eder.

Public skora aşırı uyum sağlamak:

- Public leaderboard overfitting
- Private leaderboard düşüşü
- Yanlış threshold seçimi
- Test dağılımına özel heuristik geliştirme
- Yarışma sonunda beklenmeyen sıralama kaybı

risklerini doğurur.

## Yeni karar kuralı

Bir değişiklik ancak aşağıdaki koşulların çoğunu sağlıyorsa korunmalıdır:

```text
1. Birden fazla validation senaryosunda iyileşme göstermesi
2. En az iki farklı query-group fold üzerinde tutarlı katkı sağlaması
3. Hard-negative validation performansını düşürmemesi
4. Public leaderboard'da güçlü şekilde ters sinyal üretmemesi
5. Teknik ve semantik olarak açıklanabilir bir gerekçeye sahip olması
6. Inference maliyetini kabul edilemez seviyede artırmaması
```

## Güncellenmiş geliştirme döngüsü

```text
Kod Değişikliği
      ↓
Çoklu Validation Senaryosu
      ↓
Fold Tutarlılık Kontrolü
      ↓
Threshold Optimizasyonu
      ↓
Submission
      ↓
Public Leaderboard Dış Sinyali
      ↓
Değişikliği Koru / Geri Al
```

Public leaderboard:

```text
önemli dış sinyal
```

olacaktır ancak:

```text
tek karar verici
```

olmayacaktır.

---

# 3. Leaderboard Sonrası Satır Bazlı Error Analysis Yapılamaz

Kaggle leaderboard yalnızca toplam skor verir.

Kaggle şu bilgileri sağlamaz:

- Hangi satırın yanlış olduğu
- Hangi sorguda hata yapıldığı
- False positive satırları
- False negative satırları
- Kategori bazlı gerçek test hataları
- Marka bazlı gerçek test hataları

Bu nedenle aşağıdaki akış teknik olarak uygulanamaz:

```text
Leaderboard sonucu
→ yanlış sorguları bul
→ false positive analizi
→ false negative analizi
```

## Doğru raporlama ayrımı

### Validation tarafı

Dosya:

```text
reports/validation_error_analysis.py
```

Bu modül şunları yapmalıdır:

- False positive satırlarını bulma
- False negative satırlarını bulma
- Query bazlı hata analizi
- Kategori bazlı hata analizi
- Marka bazlı hata analizi
- Negatif strateji bazlı hata analizi
- Attribute conflict bazlı hata analizi
- Hard-negative başarısızlık analizi

### Leaderboard tarafı

Dosya:

```text
reports/submission_comparison.py
```

Bu modül yalnızca deney seviyesinde karşılaştırma yapmalıdır.

Örnek:

```text
Submission V001

Model:
xlm-roberta-base

Negatifler:
random + cross-category + same-category

Threshold:
0.58

Public Score:
0.9412
```

```text
Submission V002

Model:
xlm-roberta-base

Negatifler:
random + cross-category + same-category + hard embedding

Threshold:
0.61

Public Score:
0.9538

Fark:
+0.0126
```

Leaderboard analizi satır seviyesinde değil, deney seviyesinde yapılmalıdır.

---

# 4. Cross Encoder İçin In-Batch Negative Tanımı Düzeltilmelidir

Cross encoder girişi şu şekildedir:

```text
(query, product) → relevance label
```

Örnek:

```text
(query_A, product_A, 1)
(query_A, product_B, 0)
```

Klasik cross encoder eğitiminde batch içerisindeki diğer ürünler otomatik olarak negatif kabul edilmez.

In-batch negatives yaklaşımı daha doğal olarak:

- Bi-encoder
- Contrastive learning
- Multiple Negatives Ranking Loss
- Dense retrieval

modellerinde kullanılır.

## Yanlış varsayım

```text
Batch içerisindeki diğer pozitif ürünler cross encoder tarafından otomatik negatif kabul edilir.
```

Bu doğru değildir.

## Doğru yaklaşım

Batch içindeki diğer ürünlerden açıkça yeni negatif çiftler üretilmelidir.

Örnek:

```text
Pozitif çift:

(query_A, product_A, 1)
```

Batch içindeki başka ürünler:

```text
product_B
product_C
product_D
```

Üretilecek yeni çiftler:

```text
(query_A, product_B, 0)
(query_A, product_C, 0)
(query_A, product_D, 0)
```

Ancak bu ürünlerin gerçekten negatif olduğuna dair güven kontrolü yapılmalıdır.

## Dosya adı değişikliği

Eski dosya:

```text
training/in_batch_negatives.py
```

Yeni dosya:

```text
training/batch_negative_augmenter.py
```

## V1 kararı

Bu özellik ilk baseline eğitiminde kullanılmayacaktır.

İlk sürümde yalnızca önceden oluşturulmuş yüksek güvenli negatif çiftler kullanılacaktır.

Batch negative augmentation:

```text
V2 veya V3
```

aşamasında ve yalnızca katkı kanıtlanırsa etkinleştirilecektir.

---

# 5. Ana Model Önceliği Düzeltilmelidir

Standart ModernBERT, Türkçe için ilk tercih olmamalıdır.

Ana veri dili Türkçedir.

Bu nedenle model sıralaması şu şekilde değiştirilmelidir:

## P0 — Smoke Test ve İlk Baseline

```text
xlm-roberta-base
```

Gerekçeler:

- Çok dilli ön eğitim
- Türkçe desteği
- Large modele göre daha düşük GPU ihtiyacı
- Daha hızlı eğitim
- Daha hızlı inference
- Pipeline hatalarını erken yakalama
- 3.36 milyon test çifti için daha uygulanabilir başlangıç

## P1 — Güçlü Ana Model

```text
xlm-roberta-large
```

Yalnızca base pipeline başarıyla çalıştıktan sonra denenmelidir.

## P2 — Türkçe Odaklı Alternatif

```text
dbmdz/bert-base-turkish-cased
```

BERTurk:

- Alternatif cross encoder
- Ensemble adayı
- Türkçe morfoloji açısından karşılaştırma modeli

olarak değerlendirilmelidir.

## P3 — Diğer Çok Dilli Modeller

Multilingual ModernBERT, mmBERT veya benzeri modeller ancak:

- Zaman varsa
- Kaynak varsa
- XLM-R sonuçları yeterli değilse
- Aynı validation protokolü altında ölçülebiliyorsa

denenmelidir.

## Güncellenmiş model sırası

```text
V0:
Heuristic baseline

V1:
XLM-R Base

V2:
XLM-R Base + hard negative

V3:
XLM-R Large

V4:
BERTurk karşılaştırması veya ensemble
```

---

# 6. Embedding ve FAISS Akışı Düzeltilmelidir

Aşağıdaki yaklaşım gerçekçi değildir:

```text
Önce tüm ürünlerle düz cosine similarity hesapla.
Yavaşsa FAISS kullan.
```

Yaklaşık karşılaştırma sayısı:

```text
966,444 ürün
×
50,153 sorgu
≈
48.5 milyar benzerlik hesabı
```

Tüm sorgu-ürün cosine matrisini üretmek:

- Bellek açısından uygun değildir.
- İşlem süresi açısından pahalıdır.
- Gereksizdir.
- Pipeline'ı kilitleyebilir.

## Doğru seçenekler

Hard-negative mining için aşağıdaki yöntemlerden biri kullanılmalıdır:

```text
FAISS
```

veya:

```text
Chunked top-k matrix multiplication
```

Tam similarity matrisi belleğe alınmamalıdır.

## Önerilen üretim akışı

```text
Ürün metinlerini normalize et
      ↓
BGE-M3 ile ürün embedding'lerini bir kez üret
      ↓
Embedding cache'e yaz
      ↓
FAISS index oluştur
      ↓
Train sorgularının embedding'lerini üret
      ↓
Her sorgu için top-k ürün adayını çek
      ↓
Bilinen pozitifleri çıkar
      ↓
Muhtemel pozitifleri filtrele
      ↓
Kalan adaylardan hard negative seç
```

## Cache kuralı

BGE-M3 modeli değişmediği sürece:

```text
ürün embedding'leri yeniden hesaplanmamalıdır
```

FAISS index de:

```text
ürün embedding cache değişmedikçe yeniden oluşturulmamalıdır
```

---

# 7. Iterative Hard Negative Mining Akışı Düzeltilmelidir

BGE-M3 embedding'lerini her cross encoder epoch'unda yeniden üretmek doğru değildir.

BGE-M3 sabitse embedding uzayı değişmez.

## Yanlış akış

```text
Epoch bitti
→ bütün ürün embedding'lerini yeniden hesapla
→ FAISS index yeniden oluştur
→ yeni negatif çıkar
```

Bu gereksiz derecede pahalıdır.

## Doğru iterative mining akışı

```text
BGE-M3 ürün embedding'lerini bir kez üret
      ↓
FAISS index'i bir kez oluştur
      ↓
Her query için top-k aday havuzu üret
      ↓
Known-positive ve probable-positive filtrele
      ↓
İlk hard negative setini oluştur
      ↓
Cross encoder eğit
      ↓
Aynı aday havuzunu cross encoder ile yeniden skorla
      ↓
Modelin yüksek skor verdiği yanlış adayları seç
      ↓
Yeni hard negative setini oluştur
      ↓
Cross encoder'ı yeniden eğit veya fine-tune et
```

Burada güncellenen bölüm:

```text
cross encoder hard-negative seçimi
```

olmalıdır.

BGE-M3 ürün embedding'leri her epoch yeniden üretilmemelidir.

## Aşamalı uygulama

### V1

```text
Static high-confidence negatives
```

### V2

```text
BGE-M3 + FAISS static hard negatives
```

### V3

```text
Cross-encoder rescored iterative hard negatives
```

---

# 8. Sentetik Negatifler Kesin Label 0 Kabul Edilmemelidir

Positive-unlabeled probleminde katalogdan seçilen her bilinmeyen ürün gerçek negatif değildir.

Örnek:

```text
Query:
iphone 15 pro kılıf
```

Bilinen pozitif:

```text
Apple iPhone 15 Pro Şeffaf Kılıf
```

Mining sonucu bulunan aday:

```text
iPhone 15 Pro Silikon Telefon Kılıfı
```

Bu ürün eğitimde pozitif görünmese bile büyük ihtimalle sorguyla alakalıdır.

Bunu `label=0` yapmak yanlış etiket üretir.

## Her negatif satırına eklenecek alanlar

```text
negative_type
negative_confidence
sampling_rank
lexical_similarity
semantic_similarity
source_query_id
source_item_id
```

Örnek kayıt:

```json
{
  "term_id": "TERM_123",
  "item_id": "ITEM_456",
  "label": 0,
  "negative_type": "same_category",
  "negative_confidence": 0.93,
  "sampling_rank": 18,
  "lexical_similarity": 0.21,
  "semantic_similarity": 0.48
}
```

## Confidence kullanımı

```text
confidence >= 0.90
→ normal eğitim ağırlığı
```

```text
0.75 <= confidence < 0.90
→ azaltılmış loss ağırlığı
```

```text
confidence < 0.75
→ eğitimden çıkar
```

Bu eşikler config üzerinden yönetilmelidir.

## Örnek sample weight

```text
confidence >= 0.95:
    sample_weight = 1.0

0.90 <= confidence < 0.95:
    sample_weight = 0.8

0.80 <= confidence < 0.90:
    sample_weight = 0.5

confidence < 0.80:
    discard
```

İlk baseline sürümünde yalnızca yüksek güvenli negatifler kullanılmalıdır.

---

# 9. Known-Positive ve Probable-Positive Filtreleri Eklenmelidir

Hard-negative veya same-category mining sırasında yalnızca bilinen pozitifleri çıkarmak yeterli değildir.

## Known-positive filtre

Bir sorgunun eğitimdeki tüm pozitif ürünleri negatif havuzundan çıkarılmalıdır.

```text
known_positive_items[term_id]
```

şeklinde bir lookup oluşturulmalıdır.

## Probable-positive filtre

Aşağıdaki koşulları sağlayan adaylar negatif olarak kullanılmamalıdır:

- Başlık sorguyu güçlü biçimde içeriyorsa
- Marka tam eşleşiyorsa
- Model numarası tam eşleşiyorsa
- Kategori tam eşleşiyorsa
- Semantik benzerlik aşırı yüksekse
- Lexical overlap aşırı yüksekse
- Sorgudaki kritik attribute'lar üründe eşleşiyorsa

Örnek:

```text
Query:
samsung galaxy s24 ultra kılıf
```

Aday:

```text
Samsung Galaxy S24 Ultra Şeffaf Kılıf
```

Bu ürün bilinen pozitif listesinde olmasa bile negatif yapılmamalıdır.

## Gri bölge

Şu adaylar:

```text
yüksek semantic similarity
+
yüksek lexical similarity
+
kategori eşleşmesi
```

gösteriyorsa:

```text
uncertain
```

olarak işaretlenmelidir.

Bu adaylar:

- Negatif eğitiminden çıkarılabilir
- Pseudo-positive havuzuna alınabilir
- Daha sonraki aşamada manuel incelenebilir

---

# 10. Attribute Conflict Negatifleri Sorguya Bağlı Olmalıdır

Bir attribute farklı olduğu için ürün otomatik olarak negatif sayılamaz.

## Yanlış yaklaşım

```text
Query:
kadın bot

Ürün:
beyaz yazlık kadın bot

Sonuç:
negatif
```

Bu yanlış olabilir çünkü sorgu renk veya sezon belirtmemiştir.

## Doğru yaklaşım

Attribute conflict yalnızca sorguda ilgili attribute açıkça belirtiliyorsa negatif sinyali oluşturmalıdır.

Örnek:

```text
Query:
siyah kışlık kadın bot
```

Ürün:

```text
beyaz yazlık kadın bot
```

Burada:

```text
color conflict
season conflict
```

vardır.

## Desteklenecek kritik attribute'lar

- Marka
- Ürün modeli
- Cihaz modeli
- Cinsiyet
- Yaş grubu
- Renk
- Materyal
- Beden
- Ölçü
- Kapasite
- Sezon
- Ürün tipi
- Uyumluluk bilgisi

## Attribute conflict kuralı

```python
if query_mentions(attribute):
    if product_attribute_conflicts(attribute):
        negative_signal = True
else:
    negative_signal = False
```

Attribute conflict sampler yalnızca sorguda tespit edilen attribute'lar üzerinden çalışmalıdır.

---

# 11. Fold İzolasyonu Doğru Tanımlanmalıdır

Split birimi:

```text
term_id
```

olmalıdır.

Aynı query hiçbir şekilde hem train hem validation tarafında bulunmamalıdır.

Ancak tüm ürün kataloğu:

```text
items.csv
```

hem train hem validation negatif mining sırasında kullanılabilir.

Çünkü gerçek yarışma dağılımında ürünler train ve test arasında büyük ölçüde örtüşmektedir.

## Doğru izolasyon

```text
Train query'leri
≠
Validation query'leri
```

## Gereksiz izolasyon

```text
Train ürün kataloğu
≠
Validation ürün kataloğu
```

zorunlu değildir.

## Leakage sayılacak durumlar

- Validation query metninin train eğitim çiftlerinde kullanılması
- Validation pozitif çiftinin train verisine eklenmesi
- Validation label sonuçlarının negatif mining kararına girmesi
- Validation threshold'una göre train sampler'ın ayarlanması
- Aynı query'nin farklı term_id ile duplicate olarak iki fold'a düşmesi

## Duplicate query kontrolü

Split yalnızca `term_id` seviyesinde yapılmamalıdır.

Normalize edilmiş query metni de kontrol edilmelidir.

Örnek:

```text
TERM_A:
iphone 15 pro kılıf
```

```text
TERM_B:
iphone 15 pro kilif
```

Bunlar farklı `term_id` olsa bile semantik olarak aynı sorgudur.

Bu nedenle group key tercihen:

```text
normalized_query_hash
```

olmalıdır.

Önerilen group alanı:

```text
group_id = hash(normalize(query))
```

Bu sayede aynı veya normalize edilmiş aynı query iki fold'a bölünmez.

---

# 12. Validation Birden Fazla Senaryoda Çalışmalıdır

Tek validation seti yeterli değildir.

## Validation senaryoları

### Scenario A — Easy Mix

```text
Random
+
Cross-category
```

Amaç:

- Pipeline'ın temel ayrım yeteneğini ölçmek

### Scenario B — Structural Mix

```text
Same-category
+
Same-brand
+
Attribute conflict
```

Amaç:

- Yapısal ürün farklarını ölçmek

### Scenario C — Lexical Hard

```text
Yüksek BM25
+
yüksek token overlap
+
farklı ürün modeli
```

Amaç:

- Kelime benzerliğine aşırı güveni ölçmek

### Scenario D — Semantic Hard

```text
BGE-M3 nearest neighbors
+
probable-positive filtresi
```

Amaç:

- Semantik yakın ama alakasız adaylarda performansı ölçmek

### Scenario E — Candidate-Set Simulation

Her validation query için:

```text
yaklaşık 50–150 aday ürün
```

oluşturulmalıdır.

Bu dağılım submission'daki query başına yaklaşık 104 aday yapısını taklit etmelidir.

## Raporlanacak metrikler

```text
overall_macro_f1
positive_f1
negative_f1
easy_negative_f1
same_category_f1
same_brand_f1
attribute_conflict_f1
lexical_hard_f1
semantic_hard_f1
candidate_set_macro_f1
```

Ayrıca:

```text
fold_mean
fold_std
worst_fold_score
```

raporlanmalıdır.

---

# 13. Threshold Optimizasyonu Daha Güvenli Yapılmalıdır

Tek validation fold üzerinde threshold seçmek overfitting yaratabilir.

## Doğru yöntem

Her fold için prediction probability üretilmelidir.

Out-of-fold tahminler birleştirilmelidir.

Threshold:

```text
OOF predictions
```

üzerinden optimize edilmelidir.

## İlk arama

```text
0.20 → 0.80
step = 0.01
```

## İnce arama

En iyi threshold çevresinde:

```text
best_threshold - 0.03
→
best_threshold + 0.03

step = 0.001
```

## Kaydedilecek sonuçlar

```text
global_threshold
fold_thresholds
threshold_mean
threshold_std
macro_f1_at_global_threshold
```

Fold threshold'ları çok farklıysa bu durum:

- Calibration sorunu
- Negatif dağılım sorunu
- Fold instability
- Overfitting

işareti olarak raporlanmalıdır.

---

# 14. Class-Specific Threshold Kullanılmamalıdır

Binary classification'da tek probability skoru üzerinden:

```text
prediction = probability >= threshold
```

kararı verilir.

Pozitif ve negatif için bağımsız iki threshold:

- Çakışan karar bölgeleri
- Kararsız alanlar
- Gereksiz karmaşıklık

oluşturabilir.

İlk aşamada tek global threshold kullanılmalıdır.

Daha ileri aşamada calibration yöntemleri değerlendirilebilir:

- Platt scaling
- Isotonic regression
- Temperature scaling

Ancak bunlar yalnızca OOF tahminleri üzerinde uygulanmalıdır.

---

# 15. İlk Baseline Model Heuristic Olmalıdır

Doğrudan büyük cross encoder eğitimine geçilmemelidir.

Önce hızlı bir V0 baseline hazırlanmalıdır.

## V0 özellikleri

- Query-title token overlap
- Brand match
- Category token overlap
- Model-number conflict
- Gender conflict
- Color conflict
- BM25 veya TF-IDF similarity
- Basit weighted score
- Threshold

## Amaç

- Submission pipeline'ını doğrulamak
- CSV formatını test etmek
- Kaggle authentication ve upload sürecini test etmek
- Label dağılımı hakkında dış sinyal almak
- Cross encoder için referans skor oluşturmak
- Pair join hatalarını tespit etmek

## V0 pipeline

```text
Load
→ Merge
→ Normalize
→ Heuristic scoring
→ Threshold
→ Validate submission
→ Upload
```

Bu baseline mümkün olan en kısa sürede hazırlanmalıdır.

---

# 16. Uygulama Sırası Dondurulmalıdır

## V0 — Heuristic Baseline

```text
Query-title overlap
+
brand/category/model conflict
+
threshold
```

Çıktı:

```text
submission_v000.csv
```

## V1 — XLM-R Base Cross Encoder

```text
Yüksek güvenli static negatives
+
XLM-R Base
+
OOF threshold
```

Çıktı:

```text
submission_v001.csv
```

## V2 — BGE-M3 Hard Negatives

```text
BGE-M3
+
FAISS
+
probable-positive filter
+
XLM-R Base retraining
```

Çıktı:

```text
submission_v002.csv
```

## V3 — Cross-Encoder Iterative Mining

```text
Cross encoder candidate rescoring
+
yüksek skorlu false-hard candidates
+
retraining
```

Çıktı:

```text
submission_v003.csv
```

## V4 — Model Upgrade

Aşağıdakilerden biri:

```text
XLM-R Large
```

veya:

```text
BERTurk
```

Aynı validation ve negatif protokolü kullanılmalıdır.

## V5 — Basit Blending

```text
cross_encoder_probability
+
embedding_similarity
+
lexical_score
+
critical_attribute_conflict
```

İlk blending manuel ve açıklanabilir ağırlıklarla yapılmalıdır.

## V6 — LightGBM Meta Model

Yalnızca V5 blending'i tutarlı biçimde geçerse uygulanmalıdır.

Input:

```text
OOF cross-encoder score
embedding similarity
lexical features
structured match features
```

LightGBM eğitiminde cross encoder skoru kesinlikle:

```text
out-of-fold prediction
```

olmalıdır.

Train setine ait in-sample cross encoder skorları kullanılmamalıdır.

---

# 17. Feature Selection İlk Aşamada Aşırı Karmaşık Olmamalıdır

İlk plandaki:

```text
Permutation Importance
→ SHAP
→ Correlation
```

zinciri başlangıç için gereğinden ağırdır.

## V1 feature selection

- Feature importance
- Ablation test
- Fold consistency
- Correlation kontrolü

yeterlidir.

SHAP yalnızca:

- LightGBM gerçekten kullanılırsa
- Açıklanabilirlik gerekiyorsa
- Zaman varsa

eklenmelidir.

## Ablation kuralı

Her feature grubu ayrı çıkarılarak test edilmelidir.

Örnek:

```text
Full features:
0.9621

Brand features removed:
0.9584

Delta:
-0.0037
```

Bu, feature katkısı için daha güvenilir bir sinyaldir.

---

# 18. Inference Maliyeti Baştan Ölçülmelidir

3,359,679 çift üzerinde cross encoder inference pahalı olabilir.

## Smoke benchmark

İlk olarak:

```text
10,000 pair
```

üzerinde inference süresi ölçülmelidir.

Sonra toplam süre tahmini yapılmalıdır:

```text
estimated_total_seconds
=
seconds_for_10k
×
335.9679
```

## Kaydedilecek performans metrikleri

```text
pairs_per_second
batch_size
max_length
gpu_memory_peak
estimated_total_runtime
actual_total_runtime
```

## Optimizasyon sırası

```text
1. FP16 veya BF16
2. Dynamic padding
3. Length-bucket batching
4. DataLoader worker optimizasyonu
5. Tokenization cache
6. Max length azaltma
7. Batch size ayarlama
8. torch.compile uygunluk testi
9. ONNX yalnızca gerçekten gerekirse
```

İlk model seçimi sadece F1'a göre değil:

```text
F1 / inference cost
```

dengesiyle değerlendirilmelidir.

---

# 19. Label Dağılımı Blind Tahmin Edilmemelidir

Test setindeki gerçek pozitif oranı bilinmemektedir.

Sentetik validation negatif oranını:

```text
1 pozitif : 3 negatif
```

seçmek pratik olabilir ancak gerçek test dağılımını garanti etmez.

## Yapılması gereken

Farklı validation prior'ları oluşturulmalıdır:

```text
1:1
1:2
1:3
1:5
1:10
```

Threshold hassasiyeti ölçülmelidir.

## Rapor

```text
threshold_by_prior
macro_f1_by_prior
predicted_positive_rate
```

Submission'ın tahmin ettiği pozitif oran da kaydedilmelidir.

Örnek:

```text
submission_v001 predicted positive rate:
18.4%
```

Versiyonlar arasında aşırı pozitif oran değişimi alarm üretmelidir.

---

# 20. Submission Comparison Sistemi Eklenmelidir

İki submission arasında yalnızca leaderboard skoru değil, tahmin farkları da analiz edilmelidir.

Dosya:

```text
reports/submission_comparison.py
```

## Hesaplanacak değerler

- Toplam farklı tahmin sayısı
- `0 → 1` değişen satır sayısı
- `1 → 0` değişen satır sayısı
- Query bazında değişim oranı
- Kategori bazında değişim oranı
- Predicted positive rate farkı
- En çok değişen query'ler
- En çok değişen kategori grupları

Örnek rapor:

```text
V001 vs V002

Changed rows:
184,392

0 → 1:
72,821

1 → 0:
111,571

Positive rate:
V001 = 22.3%
V002 = 21.1%

Public LB delta:
+0.0064
```

Bu rapor, leaderboard farkının hangi tahmin davranışıyla ilişkili olduğunu anlamaya yardımcı olur.

---

# 21. Güncellenmiş Dosya Yapısı

```text
project/
├── configs/
│   ├── config.yaml
│   ├── negative_sampling.yaml
│   └── model_configs/
│       ├── xlm_roberta_base.yaml
│       ├── xlm_roberta_large.yaml
│       └── berturk.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   │   ├── embedding/
│   │   ├── token/
│   │   ├── feature/
│   │   ├── attribute/
│   │   └── product/
│   ├── data_loader.py
│   ├── data_validator.py
│   └── split_builder.py
│
├── utils/
│   ├── text_cleaner.py
│   ├── attribute_parser.py
│   ├── cache_manager.py
│   ├── hashing.py
│   └── logging_utils.py
│
├── features/
│   ├── query_analyzer.py
│   ├── product_analyzer.py
│   ├── product_normalizer.py
│   ├── lexical_features.py
│   ├── structured_features.py
│   ├── semantic_features.py
│   ├── feature_engineer.py
│   └── feature_selector.py
│
├── baselines/
│   └── heuristic_baseline.py
│
├── negative_samples/
│   ├── negative_sampler.py
│   ├── confidence_scorer.py
│   ├── probable_positive_filter.py
│   ├── candidate_pool_builder.py
│   ├── iterative_miner.py
│   └── strategies/
│       ├── random_negative.py
│       ├── cross_category_negative.py
│       ├── same_category_negative.py
│       ├── same_brand_negative.py
│       ├── lexical_hard_negative.py
│       ├── attribute_conflict_negative.py
│       ├── cross_query_negative.py
│       └── embedding_hard_negative.py
│
├── embeddings/
│   ├── embedding_generator.py
│   ├── faiss_index.py
│   └── query_candidate_search.py
│
├── training/
│   ├── cross_encoder_trainer.py
│   ├── batch_negative_augmenter.py
│   ├── threshold_optimizer.py
│   ├── calibration.py
│   └── sample_weighting.py
│
├── validation/
│   ├── validator.py
│   ├── scenario_builder.py
│   ├── fold_metrics.py
│   └── negative_type_report.py
│
├── models/
│   ├── cross_encoder/
│   └── meta/
│       └── meta_classifier.py
│
├── inference/
│   ├── inference.py
│   ├── benchmark.py
│   └── batch_builder.py
│
├── submission/
│   ├── submission_generator.py
│   ├── submission_validator.py
│   └── upload_submission.py
│
├── experiments/
│   ├── tracker.py
│   ├── leaderboard_tracker.py
│   └── experiment_registry.json
│
├── reports/
│   ├── validation_error_analysis.py
│   ├── submission_comparison.py
│   ├── negative_type_analysis.py
│   └── inference_report.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_text_cleaner.py
│   ├── test_attribute_parser.py
│   ├── test_split_builder.py
│   ├── test_negative_sampler.py
│   ├── test_probable_positive_filter.py
│   ├── test_threshold_optimizer.py
│   └── test_submission_validator.py
│
└── notebooks/
```

---

# 22. Güncellenmiş Sprint Planı

## Sprint 0 — Ortam ve Smoke Test

Amaç:

```text
Python ortamı
+
paket kurulumu
+
dosya yolları
+
CSV okuma
+
Kaggle CLI
```

Kontroller:

```bash
python3 --version
kaggle --version
python -c "import pandas, torch, transformers; print('OK')"
```

---

## Sprint 1 — Veri Pipeline ve Heuristic Baseline

Yapılacaklar:

- CSV dosyalarını yükle
- Veri kolonlarını doğrula
- Query normalization
- Product normalization
- Attribute parser
- Term/item join
- Heuristic feature'lar
- V0 submission
- Submission format kontrolü
- Kaggle upload

Çıktı:

```text
submission_v000.csv
```

Bu sprint tamamlanmadan transformer eğitimine geçilmemelidir.

---

## Sprint 2 — Validation ve Yüksek Güvenli Negatifler

Yapılacaklar:

- Normalize-query group split
- Known-positive lookup
- Random negative
- Cross-category negative
- Same-category negative
- Same-brand negative
- Attribute conflict negative
- Negative confidence score
- Validation scenarios
- Label-prior sensitivity testi

Çıktı:

```text
processed_train_v001.parquet
validation_scenarios_v001.parquet
```

---

## Sprint 3 — XLM-R Base Cross Encoder

Yapılacaklar:

- Tokenization
- Dynamic padding
- FP16/BF16
- Sample weights
- OOF predictions
- Threshold optimization
- Inference benchmark
- Full inference
- Submission

Çıktı:

```text
submission_v001.csv
```

---

## Sprint 4 — BGE-M3 ve FAISS Hard Negatives

Yapılacaklar:

- Product embedding cache
- FAISS index
- Query top-k candidate retrieval
- Known-positive filtre
- Probable-positive filtre
- Hard-negative confidence score
- XLM-R yeniden eğitim
- Yeni OOF threshold
- Submission

Çıktı:

```text
submission_v002.csv
```

---

## Sprint 5 — Iterative Cross-Encoder Mining

Yapılacaklar:

- Candidate pool'u cross encoder ile skorla
- Yüksek skor verilen negatif adayları seç
- Belirsiz örnekleri çıkar
- Modeli yeniden fine-tune et
- Submission karşılaştırma raporu üret

Çıktı:

```text
submission_v003.csv
```

---

## Sprint 6 — Model Upgrade ve Ensemble

Koşula bağlı seçenekler:

- XLM-R Large
- BERTurk
- Manuel blending
- Calibration
- LightGBM meta model

Her değişiklik aynı validation protokolüyle test edilmelidir.

---

# 23. Zorunlu Test Planı

## 23.1 Data Loader Testleri

```bash
pytest tests/test_data_loader.py -v
```

Kontroller:

- Dosyalar okunuyor mu?
- Beklenen kolonlar var mı?
- Satır sayıları mantıklı mı?
- `term_id` join kaybı var mı?
- `item_id` join kaybı var mı?
- Duplicate primary ID var mı?
- CSV quote handling doğru mu?

---

## 23.2 Text Cleaner Testleri

```bash
pytest tests/test_text_cleaner.py -v
```

Test örnekleri:

```text
İPHONE 15 PRO KILIF
→
iphone 15 pro kilif
```

```text
  siyah   kadın   bot
→
siyah kadın bot
```

Kontroller:

- Unicode normalization
- Türkçe lowercase
- Whitespace cleanup
- HTML cleanup
- Noktalama temizliği
- Sayısal model bilgisini koruma

> [!CAUTION]
> Türkçe karakterlerin tamamı ana model metninden kaldırılmamalıdır.
>
> İki farklı text alanı tutulmalıdır:
>
> ```text
> model_text:
> Türkçe karakterleri korur
> ```
>
> ```text
> index_text:
> Arama ve fuzzy matching için ASCII-normalized olabilir
> ```

---

## 23.3 Attribute Parser Testleri

```bash
pytest tests/test_attribute_parser.py -v
```

Kontroller:

- Virgüllü CSV alanı doğru okunuyor mu?
- `key: value` çiftleri ayrılıyor mu?
- Değer içinde `:` varsa bozuluyor mu?
- Boş attribute işleniyor mu?
- Duplicate key yönetiliyor mu?
- Color/material/size/capacity çıkarılıyor mu?

---

## 23.4 Split Testleri

```bash
pytest tests/test_split_builder.py -v
```

Zorunlu kontroller:

```text
Train normalized-query groups
∩
Validation normalized-query groups
=
∅
```

Ayrıca:

- Aynı normalize query iki fold'a düşmemeli
- Fold kategori dağılımları raporlanmalı
- Fold query uzunluğu dağılımları raporlanmalı
- Duplicate query variation leakage kontrol edilmeli

---

## 23.5 Negative Sampler Testleri

```bash
pytest tests/test_negative_sampler.py -v
```

Zorunlu kontroller:

- Aynı `(term_id, item_id)` pozitif çift negatif olarak üretilmiyor
- Aynı negatif duplicate üretilmiyor
- Negatif strateji oranları config'e yakın
- `negative_type` dolu
- `negative_confidence` 0–1 arasında
- Düşük confidence örnekler filtreleniyor
- Attribute conflict yalnızca query attribute belirttiğinde oluşuyor
- Probable-positive adaylar çıkarılıyor

---

## 23.6 Threshold Optimizer Testleri

```bash
pytest tests/test_threshold_optimizer.py -v
```

Kontroller:

- Macro F1 doğru hesaplanıyor
- Coarse search çalışıyor
- Fine search çalışıyor
- Tie durumunda deterministik threshold seçiliyor
- OOF predictions kullanılıyor
- Tek sınıflı validation durumunda güvenli hata veriyor

---

## 23.7 Submission Validator Testleri

```bash
pytest tests/test_submission_validator.py -v
```

Zorunlu kontroller:

- Kolonlar tam olarak `id,prediction`
- Satır sayısı tam olarak submission pair sayısı
- Bütün ID'ler mevcut
- Fazladan ID yok
- Duplicate ID yok
- Prediction yalnızca `0` veya `1`
- Sample submission ile ID sırası aynı
- Dosya encoding'i doğru
- NaN yok

---

# 24. İlk Çalıştırılacak Komutlar

## Ortam doğrulama

```bash
python3 --version
pip3 --version
kaggle --version
```

## Sanal ortam

```bash
cd ~/Desktop/eticaret-hackaton

python3 -m venv .venv

source .venv/bin/activate
```

## Temel paketler

```bash
python -m pip install --upgrade pip

python -m pip install \
  pandas \
  polars \
  pyarrow \
  numpy \
  scikit-learn \
  pyyaml \
  pydantic \
  tqdm \
  pytest \
  rapidfuzz \
  rank-bm25
```

## Transformer paketleri

Python sürümü ve platform uyumluluğu doğrulandıktan sonra:

```bash
python -m pip install \
  torch \
  transformers \
  datasets \
  accelerate \
  sentence-transformers
```

## FAISS

Apple Silicon yerel geliştirme için:

```bash
python -m pip install faiss-cpu
```

GPU FAISS yalnızca Linux CUDA ortamında kullanılmalıdır.

---

# 25. İlk Smoke Test Komutları

## Data validation

```bash
python -m project.data.data_validator \
  --config project/configs/config.yaml
```

## Heuristic baseline

```bash
python -m project.baselines.heuristic_baseline \
  --config project/configs/config.yaml \
  --output project/submission/submission_v000.csv
```

## Submission validation

```bash
python -m project.submission.submission_validator \
  --pairs submission_pairs.csv \
  --sample sample_submission.csv \
  --submission project/submission/submission_v000.csv
```

## Kaggle upload

```bash
kaggle competitions submit \
  -c trendyol-e-ticaret-yarismasi-2026-kaggle \
  -f project/submission/submission_v000.csv \
  -m "V000 heuristic baseline"
```

---

# 26. İlk Cross Encoder Smoke Testi

Tam eğitimden önce:

```bash
python -m project.training.cross_encoder_trainer \
  --config project/configs/model_configs/xlm_roberta_base.yaml \
  --debug \
  --max-positive-samples 1000 \
  --max-negative-samples 3000 \
  --epochs 1
```

Başarı kriterleri:

- Eğitim başlıyor
- Loss düşüyor
- Validation tamamlanıyor
- OOF veya holdout probabilities oluşuyor
- Threshold optimizer çalışıyor
- Checkpoint kaydediliyor
- NaN loss oluşmuyor
- GPU/CPU belleği taşmıyor

---

# 27. Uygulamaya Geçmeden Önce Son Kontrol Listesi

## Veri

- [ ] CSV dosyaları doğru dizinde
- [ ] Dosya isimleri config ile aynı
- [ ] `items.csv` satır sayısı doğrulandı
- [ ] `terms.csv` satır sayısı doğrulandı
- [ ] `training_pairs.csv` label dağılımı doğrulandı
- [ ] `submission_pairs.csv` satır sayısı doğrulandı
- [ ] Sample submission formatı doğrulandı

## Ortam

- [ ] Sanal ortam aktif
- [ ] Python sürümü uyumlu
- [ ] Kaggle CLI çalışıyor
- [ ] Kaggle auth tamamlandı
- [ ] Disk alanı yeterli
- [ ] RAM kullanımı kontrol edildi

## Pipeline

- [ ] Data loader smoke test geçti
- [ ] Text cleaner testleri geçti
- [ ] Attribute parser testleri geçti
- [ ] Join kaybı yok
- [ ] V0 baseline üretildi
- [ ] Submission validator geçti
- [ ] İlk Kaggle submission yüklendi

## Model

- [ ] XLM-R Base tokenizer indirildi
- [ ] 1K/3K smoke training tamamlandı
- [ ] Inference benchmark ölçüldü
- [ ] Full runtime tahmini oluşturuldu
- [ ] Negatif confidence sistemi çalışıyor
- [ ] OOF threshold optimizer çalışıyor

---

# 28. Nihai Karar

Bu revizyonlardan sonra mimari yeterince sağlamdır.

Artık yeni mimari maddeler eklemek yerine uygulamaya geçilmelidir.

İlk hedef:

```text
V0 heuristic submission
```

İkinci hedef:

```text
V1 XLM-R Base submission
```

Üçüncü hedef:

```text
BGE-M3 + FAISS hard-negative pipeline
```

Başarı sırası:

```text
Önce çalışan sistem
→ sonra ölçülen baseline
→ sonra güvenilir negatifler
→ sonra güçlü cross encoder
→ sonra hard-negative mining
→ en son ensemble/meta model
```

Bundan sonraki çalışma prensibi:

```text
Test edilmemiş optimizasyon eklenmez.
Ölçülemeyen katkı korunmaz.
Public leaderboard tek başına takip edilmez.
Sentetik negatifler kesin gerçek kabul edilmez.
Validation leakage'e izin verilmez.
Submission pipeline doğrulanmadan büyük model eğitilmez.
```

Bu plan uygulama ve test aşamasına geçmek için nihai sürüm olarak kabul edilmelidir.