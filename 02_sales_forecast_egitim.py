"""
========================================================
MODÜL 2 - CASE STUDY 2: SATIŞ TAHMİNİ DESTEK SİSTEMİ
Eğitim Kodu - Satır Satır Açıklamalı
========================================================

Bu kod, LSTM tabanlı satış tahminine LLM'in nasıl bir
YORUM KATMANI olarak entegre edileceğini gösterir.

Ana fikir:
  LSTM  → Tahmin ÜRETIR  (sayısal çıktı)
  LLM   → Tahmini YORUMLAR (açıklama üretir)
  İnsan → Nihai ONAYI verir (Human-in-the-Loop)
"""

import math    # Matematiksel hesaplamalar
import random  # Rastgele sayı üretimi

# ============================================================
# BÖLÜM 1: SMAPE METRİĞİNİ ANLAMAK
# ============================================================

"""
SMAPE = Symmetric Mean Absolute Percentage Error
      = Simetrik Ortalama Mutlak Yüzde Hata

Neden SMAPE kullanılır?
→ Satış tahmininde tahmin 0'a yaklaşırsa MAPE bozulur
→ SMAPE bu sorunu çözer

Formül:
  SMAPE = (100/n) × Σ |gerçek - tahmin| / ((|gerçek| + |tahmin|) / 2)

SMAPE = 13.2 → Ortalama %13.2 sapma var
"""

def smape_hesapla(gercek_degerler, tahmin_degerler):
    """
    SMAPE değerini hesaplar.
    
    gercek_degerler : Gerçek satış rakamları listesi
    tahmin_degerler : Model tahmini listesi
    """
    n = len(gercek_degerler)
    toplam = 0
    
    for gercek, tahmin in zip(gercek_degerler, tahmin_degerler):
        # Payda sıfır olmasın diye kontrol
        payda = (abs(gercek) + abs(tahmin)) / 2
        if payda == 0:
            continue  # Bu değeri atla
        
        # Mutlak yüzde hata hesapla
        hata = abs(gercek - tahmin) / payda
        toplam += hata
    
    # 100 ile çarp, yüzde cinsine çevir
    return (100 / n) * toplam


# Örnek satış verisi - 30 günlük tahmin
gercek_satis = [
    1200, 1350, 1100, 980, 1420, 1680, 2100,  # Hafta 1
    1300, 1250, 1050, 990, 1380, 1720, 2050,  # Hafta 2
    1280, 1370, 1120, 1010, 1500, 1900, 2200, # Hafta 3
    1350, 1400, 1150, 1050, 1600, 1800, 2100, # Hafta 4
    1380, 1420                                  # Son 2 gün
]

# LSTM tahminleri (%13.2 SMAPE ile)
def gürültülü_tahmin_uret(gercek_degerler, smape_hedef=13.2):
    """LSTM'in tahminini simüle eder - yaklaşık %13 hata ile"""
    tahminler = []
    for gercek in gercek_degerler:
        # Rastgele hata ekle (hedef SMAPE civarında)
        hata_orani = random.uniform(-smape_hedef/100, smape_hedef/100)
        tahmin = gercek * (1 + hata_orani)
        tahminler.append(round(tahmin))
    return tahminler

random.seed(42)  # Tekrarlanabilirlik için sabit seed
lstm_tahmin = gürültülü_tahmin_uret(gercek_satis, smape_hedef=13.2)

# LLM destekli düzeltilmiş tahmin (%12.6 SMAPE)
llm_duzeltilmis = gürültülü_tahmin_uret(gercek_satis, smape_hedef=12.6)

print("=" * 55)
print("SMAPE KARŞILAŞTIRMASI")
print("=" * 55)
smape_lstm = smape_hesapla(gercek_satis, lstm_tahmin)
smape_llm  = smape_hesapla(gercek_satis, llm_duzeltilmis)

print(f"LSTM Modeli SMAPE     : {smape_lstm:.1f}  (hedef: ~13.2)")
print(f"LLM Destekli SMAPE    : {smape_llm:.1f}  (hedef: ~12.6)")
print(f"İyileşme              : {smape_lstm - smape_llm:.2f} puan")
print()
print("💡 %4.5 göreli iyileşme küçük görünse de,")
print("   yüksek ciro kararlarında büyük fark yaratır!")


# ============================================================
# BÖLÜM 2: LSTM vs LLM ROL AYRIMI
# ============================================================

"""
Bu bölüm GÖREV 1'i yanıtlar:
"LSTM ve LLM rolleri neden ayrılmalıdır?"

LSTM (Zaman Serisi Modeli):
  ✅ Geçmiş trendleri öğrenir
  ✅ Mevsimselliği yakalar  
  ✅ Sayısal tahmin üretir
  ❌ Bağlamsal bilgiyi kullanamaz (rakip kampanyası, hava vs.)
  ❌ Açıklama üretemez

LLM (Dil Modeli):
  ✅ Bağlamsal yorumlama yapar
  ✅ Nedenleri açıklar
  ✅ "Neden bu tahmin?" sorusunu cevaplar
  ❌ Zaman serisi öğrenemez
  ❌ Sayısal tutarlılık garantisi yok
  ❌ Her çalıştırmada farklı sonuç verebilir (stokastik!)
"""

print("\n" + "=" * 55)
print("GÖREV 1: LSTM vs LLM ROL AYRIMI")
print("=" * 55)

LSTM_GUCLU = [
    "Sayısal pattern öğrenir (sinyal işleme)",
    "95,000 kayıtlık geçmiş veriyle eğitilmiş",
    "Deterministik: aynı giriş → aynı çıkış",
    "Hızlı inference (gerçek zamanlı çalışır)",
    "Mevsimsellik ve trend tespiti güçlü",
]

LLM_GUCLU = [
    "Bağlam anlama (kampanya, tatil, haber vs.)",
    "Tahmin gerekçesi üretir → açıklanabilirlik",
    "Yeni senaryolara sıfır eğitimle uyum",
    "İnsan analistlerle doğal dil ile konuşur",
    "Edge-case uyarıları yapabilir",
]

LLM_RISKLER_TAHMIN_YAPARSA = [
    "Eğitim verisi yok → uydurma riski (hallucination)",
    "Her çalıştırmada farklı sayı üretebilir",
    "Güven aralığı veremez",
    "İş kararları için sorumluluk atanamaz",
    "Audit trail oluşturulamaz",
]

print("\n✅ LSTM'in Güçlü Olduğu Alanlar:")
for madde in LSTM_GUCLU:
    print(f"   • {madde}")

print("\n✅ LLM'in Güçlü Olduğu Alanlar:")
for madde in LLM_GUCLU:
    print(f"   • {madde}")

print("\n❌ LLM Doğrudan Tahmin Yaparsa Riskler:")
for madde in LLM_RISKLER_TAHMIN_YAPARSA:
    print(f"   ⚠️  {madde}")


# ============================================================
# BÖLÜM 3: HUMAN-IN-THE-LOOP MİMARİSİ
# ============================================================

"""
Sistem akışı:

  [Ham Veri]
      ↓
  [LSTM Modeli] → Sayısal tahmin üretir
      ↓
  [LLM Yorum Katmanı] → "Bu tahmin mantıklı mı?" diye değerlendirir
      ↓
  [İnsan Analist] → Son onay verir veya düzeltir
      ↓
  [Karar] → Stok siparişi, lojistik, kampanya bütçesi
"""

print("\n" + "=" * 55)
print("HUMAN-IN-THE-LOOP MİMARİSİ SİMÜLASYONU")
print("=" * 55)

class LSTMTahminSistemi:
    """LSTM modelinin çıktısını simüle eder"""
    
    def tahmin_uret(self, gun_sayisi=7):
        """Önümüzdeki N gün için tahmin üretir"""
        # Gerçekte: model.predict(X_input) çağrılır
        base = 1500
        tahminler = []
        for gun in range(gun_sayisi):
            # Hafta sonu etkisi simüle ediyoruz
            hafta_sonu_carpan = 1.4 if (gun % 7) >= 5 else 1.0
            tahmin = base * hafta_sonu_carpan * random.uniform(0.9, 1.1)
            tahminler.append(round(tahmin))
        return tahminler


class LLMYorumKatmani:
    """
    LLM'in tahmin yorumlama rolünü simüle eder.
    Gerçek uygulamada: Anthropic/OpenAI API çağrısı yapılır.
    """
    
    def yorumla(self, tahminler, baglam):
        """
        LSTM tahminini bağlamsal bilgilerle değerlendirir.
        Yeni tahmin ÜRETMEZ, mevcut tahmini YORUMLAR.
        
        tahminler : LSTM'in sayısal çıktısı
        baglam    : Ek bilgiler (kampanya, tatil vs.)
        """
        ortalama = sum(tahminler) / len(tahminler)
        maksimum = max(tahminler)
        minimum  = min(tahminler)
        
        # Bağlamsal uyarılar üret
        uyarilar = []
        
        if baglam.get("kampanya_aktif") and ortalama < 2000:
            uyarilar.append(
                "⚠️  Aktif kampanyaya rağmen tahmin düşük görünüyor. "
                "Stok miktarını artırmayı değerlendirin."
            )
        
        if baglam.get("tatil_gunu_var"):
            uyarilar.append(
                "📅 Tatil günleri tahmine dahil. "
                "Teslimat lojistiği için tampon stok öneririm."
            )
        
        if maksimum / minimum > 2.5:
            uyarilar.append(
                "📊 Tahminlerde yüksek varyasyon var. "
                "Model belirsizliği yüksek - insan incelemesi önerilir."
            )
        
        return {
            "ozet"    : f"Ortalama günlük tahmin: {ortalama:.0f} birim",
            "aralik"  : f"Min: {minimum} | Max: {maksimum}",
            "uyarilar": uyarilar if uyarilar else ["Tahmin normal aralıkta görünüyor."],
            "oneri"   : "İnsan onayı önerilir" if uyarilar else "Otomatik onaya uygun",
        }


class InsanAnalist:
    """İnsan analistin karar sürecini temsil eder"""
    
    def degerlendir(self, lstm_tahmin, llm_yorum):
        """
        Hem LSTM tahminini hem LLM yorumunu görür,
        nihai kararı verir.
        """
        # Simülasyon: gerçekte analist ekranda bunları görür
        print("\n  📋 ANALİST PANELİ")
        print(f"  LSTM Tahmini: {lstm_tahmin}")
        print(f"  LLM Özeti  : {llm_yorum['ozet']}")
        print(f"  Aralık     : {llm_yorum['aralik']}")
        print("  LLM Uyarıları:")
        for uyari in llm_yorum['uyarilar']:
            print(f"    {uyari}")
        print(f"  LLM Önerisi: {llm_yorum['oneri']}")
        
        # Uyarı varsa analist manuel müdahale eder
        if "İnsan onayı" in llm_yorum['oneri']:
            print("\n  👤 ANALİST KARARI: Manuel inceleme → Onaylandı (+%10 buffer)")
            return [t * 1.1 for t in lstm_tahmin]  # %10 buffer ekle
        else:
            print("\n  👤 ANALİST KARARI: Otomatik akış → Onaylandı")
            return lstm_tahmin


# ============================================================
# Sistemi çalıştır
# ============================================================

print("\nSİSTEM ÇALIŞIYOR...")
print("-" * 40)

# Adım 1: LSTM tahmin üret
lstm = LSTMTahminSistemi()
tahmin = lstm.tahmin_uret(gun_sayisi=7)
print(f"\n1️⃣  LSTM Tahmini (7 günlük): {tahmin}")

# Adım 2: LLM yorum katmanı
baglam_bilgisi = {
    "kampanya_aktif" : True,   # Bu hafta indirim kampanyası var
    "tatil_gunu_var" : False,
    "rakip_aktif"    : False,
}

llm = LLMYorumKatmani()
yorum = llm.yorumla(tahmin, baglam_bilgisi)
print(f"\n2️⃣  LLM Yorumu:")
for k, v in yorum.items():
    if isinstance(v, list):
        print(f"   {k}: {v}")
    else:
        print(f"   {k}: {v}")

# Adım 3: İnsan kararı
analist = InsanAnalist()
print("\n3️⃣  İnsan Analisti Değerlendiriyor...")
final_tahmin = analist.degerlendir(tahmin, yorum)
print(f"\n✅ NİHAİ TAHMİN: {[round(t) for t in final_tahmin]}")


# ============================================================
# BÖLÜM 4: GÖREV 2 - USER PROMPT TASARIMI
# ============================================================

print("\n" + "=" * 55)
print("GÖREV 2 - SATIŞ TAHMİNİ USER PROMPT")
print("=" * 55)

SATIS_USER_PROMPT = """
Sen bir satış tahmin analisti asistanısın. 
Sana bir LSTM modelinin çıktısı ve bağlamsal bilgiler verilecek.

GÖREVİN:
1. LSTM tahminini verilen bağlamla YORUMLA
2. Ayarlama GEREKİP GEREKMEDİĞİNİ değerlendir
3. Gerekçeni açıkça belirt

ÖNEMLİ KISITLAR:
- Kendin YENİ bir sayısal tahmin ÜRETME
- Yalnızca verilen LSTM çıktısına yorum yap
- Veride olmayan bilgileri uydurma
- Her yorumun için somut bir neden göster

YANIT FORMATI:
  Durum    : [normal / dikkat / ayarlama gerekli]
  Gerekçe  : [kısa açıklama]
  Öneri    : [yapılması gereken iş aksiyonu]
  Belirsizlik : [düşük / orta / yüksek]

BAĞLAM:
{context_json}

LSTM TAHMİNİ:
{lstm_output}
"""

print(SATIS_USER_PROMPT)


# ============================================================
# BÖLÜM 5: GÖREV 3 - TEMPERATURE DENEYİ
# ============================================================

print("=" * 55)
print("GÖREV 3 - SATIŞ TAHMİNİ TEMPERATURE ETKİSİ")
print("=" * 55)

TEMP_ANALIZ = {
    "Temperature = 0.0": {
        "gerekce_bicimi"   : "Kısa ve kesin → 'Tahmin kabul edilebilir'",
        "belirsizlik_vurgu": "Düşük → Verideki belirsizlik yeterince belirtilmez",
        "asiri_yorum"      : "Az → Yalnızca açıkça görünen şeyleri söyler",
        "ornek_cumlesi"    : '"LSTM tahmini %0.5 sapmayla normal aralıkta."',
    },
    "Temperature = 0.7": {
        "gerekce_bicimi"   : "Detaylı ve yaratıcı → Hikaye kurar",
        "belirsizlik_vurgu": "Yüksek → Çok fazla 'olabilir', 'muhtemelen' kullanır",
        "asiri_yorum"      : "Fazla → Veride olmayan çıkarımlar yapabilir",
        "ornek_cumlesi"    : '"Kampanya sezonu yaklaşırken, geçmiş veriler..."',
    }
}

for temp, analiz in TEMP_ANALIZ.items():
    print(f"\n🌡️  {temp}")
    for kriter, aciklama in analiz.items():
        print(f"   {kriter:<22}: {aciklama}")

print()
print("SONUÇ: Business-critical tahmin sistemleri için")
print("       temperature 0.1-0.3 arası idealdir.")
print("       Hem tutarlı hem de hafif açıklayıcı olur.")


# ============================================================
# BÖLÜM 6: GÖREV 4 - NİHAİ PRODUCTION KARARI
# ============================================================

print("\n" + "=" * 55)
print("GÖREV 4 - HUMAN-IN-THE-LOOP PRODUCTION KARARI")
print("=" * 55)

PRODUCTION_KARARI = {
    "Recommendation": (
        "✅ EVET - LLM destekli ayarlama production'a alınmalı, "
        "ANCAK Human-in-the-Loop zorunlu koşuluyla"
    ),
    
    "Reasoning": [
        "SMAPE iyileşmesi kanıtlı: 13.2 → 12.6 (pilot çalışmada)",
        "LLM tahmin üretmiyor, yalnızca yorumluyor → hallucination riski sınırlı",
        "İnsan onayı gerekliliği, hatalı LLM yorumlarını filtreler",
        "Açıklanabilirlik kısıtı karşılanıyor: LLM her kararı gerekçelendiriyor",
        "Stok/lojistik kararları için mevcut süreçle entegrasyon kolay",
    ],
    
    "Risks": [
        "LLM aşırı güven üretebilir (overconfidence) → analistin dikkatini azaltır",
        "SMAPE iyileşmesi pilot veride ölçüldü, genellenebilirlik test edilmeli",
        "LLM sağlayıcı erişiminin kesilmesi → fallback mekanizması şart",
        "Analist zamanla LLM'e aşırı bağımlı hale gelebilir (automation bias)",
        "Prompt değişikliği sistem davranışını sessizce değiştirebilir",
    ],
    
    "Next Actions": [
        "6 haftalık A/B test: LLM destekli vs. salt LSTM kararları",
        "Analist feedback döngüsü kur: LLM yorumları ne sıklıkla düzeltiliyor?",
        "LLM sağlayıcı çökmesi için fallback (LSTM-only mod) hazırla",
        "Aylık SMAPE takibi: iyileşme sürdürülebilir mi?",
        "Açıklanabilirlik audit: hangi kararlar LLM yorumuna göre değiştirildi?",
    ]
}

for alan, icerik in PRODUCTION_KARARI.items():
    if isinstance(icerik, list):
        print(f"\n{alan}:")
        for madde in icerik:
            print(f"  - {madde}")
    else:
        print(f"\n{alan}:\n  {icerik}")


print("\n" + "=" * 55)
print("EĞİTİM KODU TAMAMLANDI!")
print("İki case study'nin kıyaslaması için:")
print("→ 03_karsilastirma_ozet.py dosyasını inceleyin")
print("=" * 55)
