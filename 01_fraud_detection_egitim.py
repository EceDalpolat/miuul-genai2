"""
========================================================
MODÜL 2 - CASE STUDY 1: FRAUD DETECTION
Eğitim Kodu - Satır Satır Açıklamalı
========================================================

Bu kod, bir e-ticaret platformunun dolandırıcılık tespiti
problemini ele alır. Her adım detaylıca açıklanmıştır.
"""

# ============================================================
# BÖLÜM 1: VERİ HAZIRLAMA VE DENGESIZ VERİ PROBLEMİ
# ============================================================

import numpy as np  # Sayısal hesaplamalar için
import random       # Rastgele sayı üretimi için

# Gerçek veri setinin özelliklerini simüle ediyoruz
# Gerçek veri: 1.8 milyon işlem, %1.4 fraud oranı

TOPLAM_ISLEM   = 1_800_000   # Toplam işlem sayısı
FRAUD_ORANI    = 0.014        # %1.4 fraud (çok az!)
FRAUD_SAYI     = int(TOPLAM_ISLEM * FRAUD_ORANI)  # ~25,200 fraud işlem
NORMAL_SAYI    = TOPLAM_ISLEM - FRAUD_SAYI         # ~1,774,800 normal işlem

print("=" * 55)
print("VERİ SETİ ÖZETİ")
print("=" * 55)
print(f"Toplam işlem sayısı : {TOPLAM_ISLEM:,}")
print(f"Normal işlem sayısı : {NORMAL_SAYI:,}  (%{(1-FRAUD_ORANI)*100:.1f})")
print(f"Fraud işlem sayısı  : {FRAUD_SAYI:,}  (%{FRAUD_ORANI*100:.1f})")
print()
print("⚠️  DENGESİZ VERİ UYARISI:")
print(f"   Her 1 fraud için {int(NORMAL_SAYI/FRAUD_SAYI)} normal işlem var!")
print("   Bu durum modeli zorlaştırır.")
print()


# ============================================================
# BÖLÜM 2: METRİKLERİ ANLAMAK
# ============================================================

"""
TEMEL KAVRAMLAR:
─────────────────────────────────────────────────────────
TP (True Positive)  : Fraud dedi, gerçekten fraud        ✅
FP (False Positive) : Fraud dedi, aslında normal         ❌
TN (True Negative)  : Normal dedi, gerçekten normal      ✅
FN (False Negative) : Normal dedi, aslında fraud!        ❌ ← EN TEHLİKELİ

PRECISION = TP / (TP + FP)
→ "Fraud dediğimizde ne kadar haklıyız?"
→ Düşükse: Çok fazla müşteri engelliyoruz (operasyonel yük)

RECALL = TP / (TP + FN)
→ "Gerçek fraud'ların kaçını yakaladık?"
→ Düşükse: Fraud kaçırıyoruz (finansal kayıp!)
─────────────────────────────────────────────────────────
"""

def precision_hesapla(tp, fp):
    """
    Precision metriğini hesaplar.
    
    tp: True Positive - doğru yakaladığımız fraud sayısı
    fp: False Positive - yanlışlıkla fraud dediğimiz normal sayısı
    """
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)  # Basit bölme işlemi


def recall_hesapla(tp, fn):
    """
    Recall metriğini hesaplar.
    
    tp: True Positive - doğru yakaladığımız fraud sayısı
    fn: False Negative - kaçırdığımız fraud sayısı
    """
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)  # Basit bölme işlemi


def f1_score_hesapla(precision, recall):
    """
    F1 Score: Precision ve Recall'un harmonik ortalaması.
    İkisini de dengeli değerlendiren tek bir metrik.
    """
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


print("=" * 55)
print("METRİK HESAPLAMA ÖRNEĞİ")
print("=" * 55)

# Örnek senaryo: 1000 fraud işlem test ediyoruz
test_fraud_toplam = 1000

# Logistic Regression sonuçları (precision=0.81, recall=0.22)
lr_recall    = 0.22   # 1000 fraud'dan sadece 220'sini yakaladı
lr_precision = 0.81   # Yakaladıklarının %81'i gerçekten fraud

lr_tp = int(test_fraud_toplam * lr_recall)  # 220 yakalanan fraud
lr_fp = int(lr_tp * (1 - lr_precision) / lr_precision)  # Yanlış alarmlar
lr_fn = test_fraud_toplam - lr_tp  # 780 kaçan fraud!

print("\n📊 Logistic Regression:")
print(f"   Yakalanan fraud (TP) : {lr_tp}")
print(f"   Kaçan fraud (FN)     : {lr_fn}  ← Finansal kayıp!")
print(f"   Yanlış alarm (FP)    : {lr_fp}  ← Operasyonel yük")
print(f"   Precision            : {lr_precision:.2f}")
print(f"   Recall               : {lr_recall:.2f}")
print(f"   F1 Score             : {f1_score_hesapla(lr_precision, lr_recall):.2f}")

# XGBoost sonuçları (precision=0.74, recall=0.61)
xgb_recall    = 0.61
xgb_precision = 0.74

xgb_tp = int(test_fraud_toplam * xgb_recall)  # 610 yakalanan fraud
xgb_fp = int(xgb_tp * (1 - xgb_precision) / xgb_precision)
xgb_fn = test_fraud_toplam - xgb_tp  # 390 kaçan fraud

print("\n📊 XGBoost:")
print(f"   Yakalanan fraud (TP) : {xgb_tp}")
print(f"   Kaçan fraud (FN)     : {xgb_fn}  ← Finansal kayıp!")
print(f"   Yanlış alarm (FP)    : {xgb_fp}  ← Operasyonel yük")
print(f"   Precision            : {xgb_precision:.2f}")
print(f"   Recall               : {xgb_recall:.2f}")
print(f"   F1 Score             : {f1_score_hesapla(xgb_precision, xgb_recall):.2f}")

# Zero-shot LLM sonuçları (precision=0.66, recall=0.58)
llm_recall    = 0.58
llm_precision = 0.66

llm_tp = int(test_fraud_toplam * llm_recall)
llm_fp = int(llm_tp * (1 - llm_precision) / llm_precision)
llm_fn = test_fraud_toplam - llm_tp

print("\n📊 Zero-shot LLM:")
print(f"   Yakalanan fraud (TP) : {llm_tp}")
print(f"   Kaçan fraud (FN)     : {llm_fn}  ← Finansal kayıp!")
print(f"   Yanlış alarm (FP)    : {llm_fp}  ← Operasyonel yük")
print(f"   Precision            : {llm_precision:.2f}")
print(f"   Recall               : {llm_recall:.2f}")
print(f"   F1 Score             : {f1_score_hesapla(llm_precision, llm_recall):.2f}")


# ============================================================
# BÖLÜM 3: İŞ MALİYETİ ANALİZİ
# ============================================================

"""
Sadece metriğe bakmak yetmez!
İş dünyasında her hata farklı maliyete sahiptir:

  • False Negative (FN) = Kaçan fraud
    → Ortalama fraud işlem tutarı: 500 TL
    → Doğrudan finansal kayıp!

  • False Positive (FP) = Yanlış alarm
    → Manuel inceleme başına: 15 TL maliyet
    → Müşteri memnuniyeti kaybı
    → Operasyonel kapasite tükenir
"""

FRAUD_BASI_KAYIP       = 500   # TL - Kaçan her fraud için kayıp
MANUEL_INCELEME_MALIYET = 15   # TL - Her yanlış alarm için maliyet

print("\n" + "=" * 55)
print("İŞ MALİYETİ KARŞILAŞTIRMASI")
print("=" * 55)
print(f"{'Model':<22} {'FN Maliyeti':>12} {'FP Maliyeti':>12} {'Toplam':>12}")
print("-" * 60)

modeller = [
    ("Logistic Regression", lr_fn, lr_fp),
    ("XGBoost",             xgb_fn, xgb_fp),
    ("Zero-shot LLM",       llm_fn, llm_fp),
]

for ad, fn, fp in modeller:
    fn_maliyet    = fn * FRAUD_BASI_KAYIP         # Kaçan fraud maliyeti
    fp_maliyet    = fp * MANUEL_INCELEME_MALIYET  # Yanlış alarm maliyeti
    toplam        = fn_maliyet + fp_maliyet
    print(f"{ad:<22} {fn_maliyet:>10,} TL {fp_maliyet:>10,} TL {toplam:>10,} TL")

print()
print("💡 SONUÇ: En yüksek recall'a sahip model (XGBoost)")
print("   en az finansal kayba neden oluyor!")


# ============================================================
# BÖLÜM 4: LATENCY (GECİKME) KISITI
# ============================================================

"""
Sistem kısıtı: Yanıt süresi < 250 ms olmalı!
Aksi halde ödeme deneyimi bozulur.

Model karmaşıklığı arttıkça yanıt süresi uzar.
LLM'ler genellikle bu kısıtı karşılayamaz!
"""

import time  # Zaman ölçümü için

def model_hizi_simule_et(model_adi, ortalama_ms, std_ms=10):
    """
    Model yanıt süresini simüle eder.
    Gerçekte API çağrısı veya inference time ölçülür.
    """
    # Gerçekçi varyasyon ekle
    sure_ms = max(1, ortalama_ms + random.gauss(0, std_ms))
    time.sleep(sure_ms / 1000)  # Milisaniyeyi saniyeye çevir
    return sure_ms

print("\n" + "=" * 55)
print("LATENCY (GECİKME) TESTİ")
print("=" * 55)
print("Kısıt: < 250 ms olmalı\n")

# Gerçekçi tahminler (üretim ortamı değerleri)
latency_tahminleri = {
    "Logistic Regression" : 5,    # Çok hızlı - basit formül
    "XGBoost"             : 25,   # Hızlı - ağaç tabanlı
    "Zero-shot LLM (API)" : 1500, # ÇOK YAVAŞ - API çağrısı!
}

SINIR_MS = 250  # Maksimum izin verilen süre

for model, tahmin_ms in latency_tahminleri.items():
    durum = "✅ UYGUN" if tahmin_ms < SINIR_MS else "❌ UYGUN DEĞİL"
    print(f"{model:<24} ~{tahmin_ms:>5} ms  {durum}")

print()
print("⚠️  Zero-shot LLM, 250ms sınırını aşıyor!")
print("   Production'da BAĞIMSIZ KULLANIM mümkün değil.")


# ============================================================
# BÖLÜM 5: GÖREV 1 CEVABI - TEKNİK YORUM PARAGRAF
# ============================================================

print("\n" + "=" * 55)
print("GÖREV 1 - TEKNİK YORUM PARAGRAF")
print("=" * 55)
print("""
Fraud detection probleminde precision ve recall kritiktir:

• RECALL (Duyarlılık): Gerçek fraud işlemlerinin kaçının 
  tespit edilebildiğini ölçer. Düşük recall = kaçan fraud
  = doğrudan finansal kayıp. Bu örnekte Logistic Regression'ın
  recall'u yalnızca 0.22'dir; her 100 fraud'dan 78'i gözden
  kaçmaktadır.

• PRECISION (Kesinlik): Fraud olarak işaretlenen işlemlerin
  ne kadarının gerçekten fraud olduğunu ölçer. Düşük precision
  = yüksek false positive = manuel inceleme yükü artışı.
  
• İŞ BAĞLANTISI: False negative, kaçırılan her işlem başına
  ortalama 500 TL kayba yol açarken; false positive, sınırlı
  manuel inceleme kapasitesini tüketir. XGBoost (P:0.74, R:0.61),
  bu iki maliyeti en dengeli şekilde yöneten modeldir.
""")


# ============================================================
# BÖLÜM 6: GÖREV 2 - USER PROMPT TASARIMI
# ============================================================

print("=" * 55)
print("GÖREV 2 - ÖRNEK USER PROMPT")
print("=" * 55)

USER_PROMPT = """
Aşağıdaki fraud detection context'ini analiz et ve 
üretim ortamı için en uygun yaklaşımı belirle.

CONTEXT:
{context_json}

GÖREV:
Yalnızca verilen metrikler ve kısıtlara dayanarak 
(varsayım ve tahmin YAPMA) şu soruları cevapla:

1. Hangi model production için önerilir? Neden?
2. False negative ve false positive'in iş maliyetini 
   bu bağlamda değerlendir.
3. 250ms latency kısıtı göz önüne alındığında hangi 
   yaklaşım uygulanabilir?
4. Önerilen yaklaşımın riskleri nelerdir?

Cevabını şu formatta ver:
- Recommendation: [seçilen yaklaşım]
- Reasoning: [madde madde gerekçe]
- Risks: [olası riskler]
- Next actions: [sonraki adımlar]
"""

print(USER_PROMPT)


# ============================================================
# BÖLÜM 7: GÖREV 3 - TEMPERATURE DENEYİ SİMÜLASYONU
# ============================================================

"""
Temperature = 0.0 → Deterministik, kesin kararlar
Temperature = 0.7 → Daha yaratıcı, çeşitli ifadeler

Fraud detection gibi kritik sistemlerde LOW temperature
tercih edilir çünkü tutarlılık ve güvenilirlik önceliklidir.
"""

print("=" * 55)
print("GÖREV 3 - TEMPERATURE ETKİSİ KARŞILAŞTIRMASI")
print("=" * 55)

TEMPERATURE_KARSILASTIRMA = {
    "Temperature = 0.0": {
        "karar_dili"       : "Kesin ve net → 'XGBoost seçilmelidir'",
        "risk_vurgusu"     : "Yüksek → Spesifik rakamlar ve eşikler belirtilir",
        "spekulasyon"      : "Minimum → Yalnızca verilen context'e dayalı",
        "uretim_uygunlugu" : "✅ Yüksek - Kritik kararlar için ideal",
    },
    "Temperature = 0.7": {
        "karar_dili"       : "Yumuşak → 'XGBoost daha uygun görünebilir'",
        "risk_vurgusu"     : "Değişken → Bazen vurgulanır, bazen atlanır",
        "spekulasyon"      : "Yüksek → Veride olmayan yorumlar eklenebilir",
        "uretim_uygunlugu" : "⚠️  Riskli - Tutarsız cevaplar üretebilir",
    }
}

for temp, ozellikler in TEMPERATURE_KARSILASTIRMA.items():
    print(f"\n🌡️  {temp}")
    for ozellik, aciklama in ozellikler.items():
        print(f"   {ozellik:<20}: {aciklama}")

print()
print("ÖZET: Fraud detection = kritik karar")
print("      → Temperature 0.0-0.2 arası önerilir!")


# ============================================================
# BÖLÜM 8: GÖREV 4 - NİHAİ PRODUCTION KARARI
# ============================================================

print("\n" + "=" * 55)
print("GÖREV 4 - NİHAİ PRODUCTION KARARI (temp=0.2)")
print("=" * 55)

PRODUCTION_KARARI = {
    "Model": "Claude / GPT / Gemini (referans: temperature 0.1–0.3)",
    
    "Recommendation": "XGBoost — primary fraud detection engine",
    
    "Reasoning": [
        "En dengeli precision-recall profili: P=0.74, R=0.61",
        "250ms latency kısıtını karşılar (~25ms yanıt süresi)",
        "1.8M kayıtlı etiketli veriyle güçlü biçimde eğitilebilir",
        "Zero-shot LLM'in aksine, gerçek zamanlı scoring destekler",
    ],
    
    "Risks": [
        "Zaman içinde drift: Fraud pattern'ları değişirse model güncellenmeli",
        "False positive yükü: FP başına 15 TL × binlerce işlem = önemli maliyet",
        "Yeni fraud türleri: Model görmediği pattern'lara karşı kör olabilir",
    ],
    
    "Next Actions": [
        "XGBoost'u A/B test ile mevcut sistemle karşılaştır",
        "Recall eşiğini 0.70'e çıkarmak için threshold tuning yap",
        "LLM'i offline fraud pattern analizi için destekleyici kullan",
        "Model performansını haftalık izle, drift alarm sistemi kur",
    ]
}

for alan, icerik in PRODUCTION_KARARI.items():
    if isinstance(icerik, list):
        print(f"\n{alan}:")
        for madde in icerik:
            print(f"  - {madde}")
    else:
        print(f"\n{alan}: {icerik}")

print("\n" + "=" * 55)
print("EĞİTİM KODU TAMAMLANDI!")
print("Sonraki dosya: 02_sales_forecast_egitim.py")
print("=" * 55)
