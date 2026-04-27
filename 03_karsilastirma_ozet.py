"""
========================================================
MODÜL 2 - İKİ CASE STUDY KARŞILAŞTIRMA ÖZETİ
Eğitim Kodu - Satır Satır Açıklamalı
========================================================

Bu dosya iki case study'yi yan yana karşılaştırır
ve temel öğrenme çıktılarını pekiştirir.
"""

# ============================================================
# KARŞILAŞTIRMA TABLOSU
# ============================================================

print("=" * 70)
print("MODÜL 2 - İKİ CASE STUDY KARŞILAŞTIRMASI")
print("=" * 70)

KARSILASTIRMA = {
    "Problem Türü": {
        "Fraud Detection"   : "İkili sınıflandırma (fraud / normal)",
        "Satış Tahmini"     : "Zaman serisi regresyonu (sayısal değer)",
    },
    "Veri Büyüklüğü": {
        "Fraud Detection"   : "1.8 milyon işlem (%1.4 fraud)",
        "Satış Tahmini"     : "95,000 kayıt (30 günlük tahmin)",
    },
    "Değerlendirme Metriği": {
        "Fraud Detection"   : "Precision & Recall (F1 Score)",
        "Satış Tahmini"     : "SMAPE (Yüzde hata)",
    },
    "LLM'in Rolü": {
        "Fraud Detection"   : "Zero-shot sınıflandırıcı (bağımsız model)",
        "Satış Tahmini"     : "Yorum katmanı (LSTM destekçisi)",
    },
    "LLM Başarısı": {
        "Fraud Detection"   : "Orta (P:0.66, R:0.58) – latency sorunu var!",
        "Satış Tahmini"     : "İyi (SMAPE: 13.2 → 12.6) – yorum rolünde başarılı",
    },
    "Kritik Kısıt": {
        "Fraud Detection"   : "< 250ms latency – LLM tek başına kullanılamaz",
        "Satış Tahmini"     : "Human approval zorunlu – LLM nihai karar veremez",
    },
    "Production Önerisi": {
        "Fraud Detection"   : "XGBoost primary + LLM offline analiz",
        "Satış Tahmini"     : "LSTM + LLM yorum + insan onayı (hibrit)",
    },
}

for kategori, degerler in KARSILASTIRMA.items():
    print(f"\n📌 {kategori}:")
    for case, deger in degerler.items():
        print(f"   {'Fraud Detection':<18}: {deger}" if case == "Fraud Detection"
              else f"   {'Satış Tahmini':<18}: {deger}")


# ============================================================
# TEMEL ÖĞRENME ÇIKTILARI
# ============================================================

print("\n\n" + "=" * 70)
print("TEMEL ÖĞRENME ÇIKTILARI")
print("=" * 70)

OGRENME_CIKTILARI = [
    {
        "no"    : 1,
        "baslik": "LLM her zaman en iyi çözüm değildir",
        "aciklama": (
            "Fraud detection'da latency kısıtı LLM'i production'dan dışlarken, "
            "satış tahmininde LLM yorum rolüyle değer katar. "
            "Doğru soruyu sorun: 'LLM burada ne rolü üstlenmeli?'"
        ),
    },
    {
        "no"    : 2,
        "baslik": "Metrikler bağlamda anlam kazanır",
        "aciklama": (
            "Recall=0.22 (Logistic Regression) istatistiksel görünür, "
            "ama iş bağlamında 780 kaçan fraud = büyük finansal kayıp. "
            "Her metriği iş maliyetiyle eşleştirin."
        ),
    },
    {
        "no"    : 3,
        "baslik": "Temperature kritik kararlarda düşük tutulmalı",
        "aciklama": (
            "Temperature=0.0 deterministik ve tutarlı cevap verir. "
            "Fraud tespiti veya satış kararları gibi kritik sistemlerde "
            "spekülatif cevaplar kabul edilemez. "
            "Önerilen aralık: 0.0–0.2"
        ),
    },
    {
        "no"    : 4,
        "baslik": "Human-in-the-Loop açıklanabilirliği zorunlu kılar",
        "aciklama": (
            "İnsan bir kararı onaylamak için 'neden?' sorusuna cevap ister. "
            "LLM bu açıklamayı üretmede güçlüdür. "
            "Hibrit mimari: model tahmin eder, LLM açıklar, insan onaylar."
        ),
    },
    {
        "no"    : 5,
        "baslik": "Dengesiz veri (imbalanced) özel dikkat gerektirir",
        "aciklama": (
            "%1.4 fraud oranı ile accuracy yanıltıcıdır. "
            "Model 'hepsini normal' dese %98.6 accuracy alır! "
            "Bu yüzden precision-recall kullanılır."
        ),
    },
]

for cikti in OGRENME_CIKTILARI:
    print(f"\n{cikti['no']}. {cikti['baslik']}")
    print(f"   → {cikti['aciklama']}")


# ============================================================
# ORTAK HATA - ACCURACY TUZAĞI
# ============================================================

print("\n\n" + "=" * 70)
print("⚠️  SERBEST ZAMAN UYARISI: ACCURACY TUZAĞI")
print("=" * 70)

toplam = 1_800_000
fraud  = 25_200
normal = toplam - fraud

# Eğer model "hepsini normal" derse...
yanlis_model_tp = 0      # Hiç fraud yakalamadı
yanlis_model_tn = normal # Tüm normalleri doğru
yanlis_model_fp = 0
yanlis_model_fn = fraud  # Tüm fraud'ları kaçırdı

accuracy = (yanlis_model_tp + yanlis_model_tn) / toplam

print(f"\nSenaryo: Model 'hepsine normal' derse...")
print(f"  Doğru normal tespiti  : {yanlis_model_tn:,}")
print(f"  Yakalanan fraud        : {yanlis_model_tp:,}")
print(f"  Kaçan fraud (FN)      : {yanlis_model_fn:,}")
print(f"  Accuracy              : %{accuracy*100:.2f}  ← ALDATICI!")
print(f"  Recall                : %{0.0:.0f}            ← FELAKET!")
print()
print("SONUÇ: Accuracy tek başına imbalanced veri için YANILTICIDIR.")
print("       Her zaman precision + recall + F1 birlikte değerlendirin.")


# ============================================================
# HIZLI REFERANS KART
# ============================================================

print("\n\n" + "=" * 70)
print("HIZLI REFERANS KARTI")
print("=" * 70)
print("""
PRECISION  = TP / (TP + FP)   → "Alarm doğruluğu"
RECALL     = TP / (TP + FN)   → "Yakalama oranı"
F1 SCORE   = 2 × P × R / (P + R)  → "Dengeli ölçüm"
SMAPE      = (100/n) × Σ |G-T| / ((|G|+|T|)/2)  → "Tahmin hatası"

TEMPERATURE:
  0.0   → Deterministik, tutarlı, dar    → Kritik kararlar için
  0.3   → Hafif çeşitlilik, açıklayıcı  → Yorum sistemleri için  
  0.7   → Yaratıcı, spekülatif          → Beyin fırtınası için
  1.0+  → Kaotik, tutarsız              → Production için UYGUN DEĞİL

LLM ROLLERI:
  ✅ Yorumlayıcı  (tahmin açıklama)
  ✅ Karar destek (öneri sunma)
  ✅ Zero-shot    (etiket yok, bağımsız sınıflandırma)
  ⚠️  Birincil tahmin → sadece latency & accuracy elverirse
  ❌  Tek başına production karar verici → destekleyici rol şart
""")

print("=" * 70)
print("TÜM EĞİTİM KODLARI TAMAMLANDI!")
print()
print("Dosyalar:")
print("  01_fraud_detection_egitim.py  → Case Study 1")
print("  02_sales_forecast_egitim.py   → Case Study 2")
print("  03_karsilastirma_ozet.py      → Bu dosya - Özet")
print("=" * 70)
