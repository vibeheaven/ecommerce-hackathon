# Plan v5 — Macro F1'i 0.846'dan Yukarı Taşıma Planı

> Referans: `metrics.md` (2026-07-17 koşusu, xlm-roberta-base, 3 epoch, sampled dataset)

## 1. Mevcut Durum Özeti

| Metrik | Epoch 1 | Epoch 2 | Yorum |
|---|---|---|---|
| Overall Macro F1 | 0.8353 | 0.8460 | Hâlâ yükseliyordu, erken kesildi |
| Precision / Recall | 0.830 / 0.841 | 0.839 / 0.854 | Dengeli, threshold (0.51) sağlıklı |
| Accuracy | 0.8745 | 0.8820 | — |

Negatif tipi doğruluğu (Epoch 2, True Negative Rate):

| Negatif tipi | Doğruluk | Örnek sayısı | Hata sayısı |
|---|---|---|---|
| same_category | **%71.4** | 44,117 | **~12,600** |
| lexical_hard | %89.7 | 741 | 76 |
| same_brand | %90.1 | 4,355 | 433 |
| cross_category | %99.8 | 49,988 | 109 |
| random | %99.6 | 49,987 | 209 |
| cross_query | %99.9 | 806 | 1 |
| attribute_conflict | — | **1** | — |

Senaryo F1'leri: easy_mix 0.917, structural 0.764, lexical_hard/semantic_hard 0.501, candidate_set_sim 0.846.

## 2. Kök Neden Analizi

1. **Tüm yanlış negatiflerin ~%95'i `same_category` tipinden geliyor.** Model "aynı kategoriden ama sorguya uymayan ürün" ayrımını öğrenememiş. Bu tek başına en büyük kazanç alanı: same_category %71 → %85 çıkarsa overall F1 kabaca +0.02–0.03 kazanır.
2. **Zor negatif stratejileri fiilen çalışmıyor.** `negative_sampling.yaml` oranları (lexical_hard 0.05, attribute_conflict 0.05, cross_query 0.15, embedding_hard 0.25) ile üretilen veri arasında uçurum var: lexical_hard 741, attribute_conflict **1**, cross_query 806, embedding_hard **0** örnek. Eğitim sinyalinin ~%67'si zaten %99+ doğrulukla çözülen kolay negatiflerden (random + cross_category, ~100k) oluşuyor — model kolay örneklerle vakit harcıyor.
3. **Eğitim erken kesilmiş.** Config 5 epoch derken koşu `--epochs 3` ile başlatılmış; F1 epoch 1→2 arası +0.011 artıyordu, plato görülmeden durmuş.
4. **Validasyon şüphesi:** `lexical_hard` ve `semantic_hard` senaryoları birebir aynı satır sayısı (50,741) ve birebir aynı F1 (0.5008) veriyor — `semantic_hard` (BGE-M3 tabanlı) büyük ihtimalle implement edilmemiş, lexical'a fallback yapıyor. Ayrıca 5-fold config'e rağmen sadece Fold 0 değerlendiriliyor.
5. **Senaryo F1'leri yanıltıcı okunabilir:** lexical_hard senaryosunda 50,000 pozitife karşı sadece 741 negatif var; negatif sınıfın F1'i dengesizlikten dolayı çöküyor. Senaryolar pozitif/negatif dengeli örneklenmeli, yoksa 0.50 rakamı gerçek performansı temsil etmez.

## 3. Aksiyon Planı (Öncelik Sırasıyla)

### Faz 1 — Hızlı Kazançlar (kod değişikliği minimal, beklenen etki: +0.010–0.015)

- [ ] **Eğitimi 5 epoch'a çıkar** (`--epochs 5`), `early_stopping_patience: 3` zaten var; plato görülene kadar devam et. Cosine scheduler'ın total step hesabının 5 epoch'a göre yeniden kurulduğunu doğrula.
- [ ] **5-fold CV'yi gerçekten çalıştır.** `validator.py` şu an tek fold raporluyor; OOF threshold optimizasyonu tek fold üstünden yapılınca threshold'a overfit riski var. 5 fold ortalaması hem daha güvenilir F1 verir hem ensemble için 5 checkpoint bırakır.
- [ ] **Senaryo değerlendirmesini dengele:** her senaryoda negatif sayısı kadar pozitif örnekle (subsample) F1 hesapla; mevcut 0.50'lik rakamların gerçek mi artefakt mı olduğunu netleştir.

### Faz 2 — Negatif Örnekleme Revizyonu (ana kazanç alanı, beklenen etki: +0.020–0.030)

- [ ] **Strateji üretim hattını debug et:** `attribute_conflict` 1 örnek, `embedding_hard` 0 örnek üretmiş. `negative_sampler.py` ve `strategies/` altındaki her stratejinin config oranına ulaşamama sebebini logla (aday bulunamıyor mu, probable_positive filtresi mi eliyor?).
- [ ] **Kolay negatif payını düşür:** random + cross_category toplamını ~%20'ye çek (şu an fiilen ~%67). Boşalan kotayı same_category, same_brand, lexical_hard ve attribute_conflict'e dağıt. Hedef dağılım: same_category %35, embedding_hard %20, lexical_hard %10, same_brand %10, attribute_conflict %5, cross_query %10, random+cross_category %10.
- [ ] **`embedding_hard` stratejisini devreye al** (BGE-M3 + FAISS): her sorgu için en yakın ama pozitif olmayan ürünler. `semantic_hard` validasyon senaryosunu da aynı altyapıyla gerçek verisine kavuştur.
- [ ] **Model tabanlı hard negative mining (self-mining):** mevcut en iyi checkpoint ile eğitim adaylarını skorla; yüksek skor alan (ör. >0.7) negatifleri bir sonraki tura "çok zor negatif" olarak ekle. Probable-positive filtresinden geçir ki gürültülü pozitifleri negatif diye öğretmeyelim. Bu, same_category zaafiyetine en doğrudan saldırıdır.
- [ ] **Sample weighting'i tip bazlı yap:** `sample_weighting.py` mevcut confidence katmanlarına ek olarak same_category ve lexical_hard örneklerine 1.5–2.0x ağırlık versin; random/cross_category 0.5x'e insin.

### Faz 3 — Model ve Girdi İyileştirmeleri (beklenen etki: +0.010–0.020)

- [ ] **Girdiyi zenginleştir:** cross-encoder girdisine yalnız başlık değil, `brand`, `category` ve parse edilmiş `attributes` alanlarını yapılandırılmış biçimde ekle (ör. `query [SEP] title | marka: X | kategori: Y | renk: Z`). same_category ve attribute_conflict ayrımı bu sinyaller olmadan öğrenilemez. max_length 256 yeterli mi token istatistiğiyle doğrula.
- [ ] **Daha güçlü backbone dene:** `xlm-roberta-large` ve `BAAI/bge-reranker-v2-m3` (XLM-R large tabanlı, reranking'e ön-eğitimli) — ikincisi bu görev için genelde base'e +0.01–0.02 verir. Large için lr'ı 1e-5'e düşür, LLRD (layer-wise lr decay ~0.9) ekle.
- [ ] **Eğitim stabilizasyonu:** label smoothing (0.05) ve/veya focal loss (γ=2) ile zor örneklere odaklanmayı artır; opsiyonel FGM adversarial training.

### Faz 4 — Ensemble ve Son Rötuş (beklenen etki: +0.005–0.010)

- [ ] 5 fold checkpoint'lerinin skor ortalaması (logit averaging) + farklı backbone'ların ensemble'ı.
- [ ] Threshold'u OOF üzerinden ensemble skorlarıyla yeniden optimize et.
- [ ] **Pseudo-labeling (opsiyonel):** submission_pairs üzerinde çok emin olunan tahminleri (skor <0.05 veya >0.95) eğitime geri kat, tek turla sınırla.

## 4. Başarı Kriterleri

| Metrik | Şu an | Hedef |
|---|---|---|
| Overall Macro F1 (OOF, 5-fold) | 0.846 (tek fold) | ≥ 0.88 |
| same_category TNR | %71.4 | ≥ %85 |
| structural senaryo F1 | 0.764 | ≥ 0.82 |
| lexical/semantic_hard (dengeli ölçüm) | ? (önce ölç) | baseline +5 puan |

## 5. Çalışma Sırası ve Riskler

1. Faz 1 (yarım gün) → yeniden koş, yeni baseline'ı kaydet.
2. Faz 2 (1–2 gün) → her değişikliği tek tek koşup `experiment_registry.json`'a işle; negatif dağılımı değişince threshold'un kayacağını unutma (her koşuda yeniden optimize ediliyor, sorun değil ama fold'lar arası tutarlılığı kontrol et).
3. Faz 3–4 zaman kalırsa; large model VRAM yetmezse gradient checkpointing + batch 8 / accumulation 8.

**En büyük risk:** hard negatif oranını artırırken gürültülü "aslında pozitif" örneklerin negatif olarak etiketlenmesi. Probable-positive filtresi her yeni strateji için zorunlu tutulmalı; self-mining'de discard eşiğini muhafazakâr seç.
