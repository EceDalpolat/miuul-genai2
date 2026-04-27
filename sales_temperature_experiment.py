"""
========================================================
SATIŞ TAHMİNİ - TEMPERATURE DENEYİ
GenAI Modül 2 – Görev 3 (Case Study 2)

Ne yapıyor bu kod?
──────────────────
LSTM modelinin ürettiği satış tahminini, aynı LLM'lere
iki farklı temperature ile yorumlatıyoruz:
   Temperature = 0.0  →  Yapılandırılmış, ölçülü yorum
   Temperature = 0.7  →  Daha zengin ama spekülatif yorum

Case Study 1'den fark:
──────────────────────
Fraud detection'da LLM doğrudan sınıflandırma yapıyordu.
Burada LLM TAHMIN ÜRETM İYOR — sadece LSTM'in çıktısını
bağlamsal bilgilerle yorumluyor. Bu ayrım kritiktir!

LLM'in rolü: "İkinci göz" → yorumla, gerekçelendir, öner.
             Kesinlikle kendi sayısal tahmini VERME.

Kurulum & Çalıştırma:
──────────────────────
   pip install openai google-generativeai cohere
   export OPENAI_API_KEY / GOOGLE_API_KEY / COHERE_API_KEY
   python sales_temperature_experiment.py
========================================================
"""

import os
import json
import time
from datetime import datetime


# ============================================================
# ADIM 1: CONTEXT HAZIRLA
# LSTM modelinin performans bilgisi ve sistem kısıtları.
# Bu bilgiler değişmez — her temperature'da aynı.
# ============================================================

CONTEXT_JSON = json.dumps({
    "task":                   "sales forecast adjustment support",
    "dataset_size":           95000,
    "base_model":             "lstm_time_series",
    "forecast_horizon_days":  30,         # 30 günlük tahmin
    "evaluation_metric":      "smape",
    "base_model_smape":       13.2,       # LSTM tek başına %13.2 hata
    "llm_adjusted_smape":     12.6,       # LLM desteğiyle %12.6 hata
    "constraints": {
        "human_approval_required":    True,  # İnsan onayı zorunlu
        "explainability_required":    True,  # Her karar açıklanabilir olmalı
        "business_critical_decisions": True  # Stok, lojistik, tedarikçi kararları
    }
}, indent=2, ensure_ascii=False)


# ============================================================
# Bağlamsal bilgiler — LSTM'in göremediği dışsal faktörler.
# LLM bu bilgileri kullanarak LSTM tahminini yorumlayacak.
# ============================================================

CONTEXTUAL_INFO = """
Bağlamsal bilgiler (LSTM modelinin öğrenemediği dışsal faktörler):
- Önümüzdeki 30 gün içinde Anneler Günü (5 Mayıs) bulunmaktadır.
- Rakip firma geçen hafta %30 indirim kampanyası başlattı.
- Hava tahminleri önümüzdeki 2 hafta normalin üstünde sıcaklık öngörüyor.
- Şirketimiz bu hafta sonu yeni bir ürün koleksiyonu lansmanı yapacak.
"""

# Simüle edilmiş LSTM çıktısı
# (Gerçekte model.predict(X_test) sonucu burada gelir)
LSTM_OUTPUT = {
    "forecast_period": "2025-05-01 to 2025-05-30",
    "predicted_units": [
        1320, 1280, 1410, 1350, 1290, 1900, 2050,   # 1. hafta
        1310, 1270, 1390, 1340, 1280, 1870, 2020,   # 2. hafta
        1330, 1295, 1405, 1360, 1285, 1890, 2040,   # 3. hafta
        1315, 1275, 1395, 1345, 1288, 1880, 2030,   # 4. hafta
        1325, 1285                                   # 5. hafta (2 gün)
    ],
    "smape_confidence": 13.2,  # Beklenen hata oranı
    "model_version": "lstm_v2.3"
}


# ============================================================
# ADIM 2: USER PROMPT TASARIMI
#
# Bu prompt Case Study 1'dekinden önemli ölçüde farklı:
#   ✅ LLM'e açıkça "tahmin üretme" diyoruz
#   ✅ Yorumlama formatı daha yapılandırılmış
#   ✅ Belirsizlik notu ve güven seviyesi istiyoruz
#   ✅ Human approval'ı hatırlatıyoruz
# ============================================================

USER_PROMPT = f"""Sen bir satış tahmini değerlendirme uzmanısın. \
Görevin, LSTM modelinin ürettiği tahmini yorumlamak ve ayarlama \
gerekip gerekmediğini değerlendirmektir.

ÖNEMLİ: Sen tahmin ÜRETMİYORSUN — yalnızca mevcut LSTM tahminini \
bağlamsal bilgiler ışığında değerlendiriyorsun.

SISTEM BİLGİSİ:
{CONTEXT_JSON}

BAĞLAMSAL BİLGİLER:
{CONTEXTUAL_INFO}

LSTM MODEL ÇIKTISI:
{json.dumps(LSTM_OUTPUT, indent=2, ensure_ascii=False)}

TALİMATLAR:
1. LSTM modelinin tahminini yukarıdaki bağlamsal bilgilerle değerlendir.
2. Ayarlama gerekip gerekmediğine karar ver. Gerekiyorsa yönünü \
(yukarı/aşağı) ve yaklaşık yüzde aralığını belirt. \
Kesin sayısal tahmin ÜRETME.
3. Her değerlendirme için açık ve izlenebilir gerekçe sun. \
Yalnızca verilen bilgilere dayan.
4. Emin olmadığın durumlarda "bu bilgiyle belirlenemez" de.
5. Son kararı insan uzmanın vereceğini unutma.

ÇIKTI FORMATI:
- Mevcut Tahmin Değerlendirmesi: [LSTM tahmininin güçlü/zayıf yönleri]
- Ayarlama Önerisi:              [Gerekli / Gerekli Değil]
- Ayarlama Yönü ve Aralığı:      [Yukarı/Aşağı, yaklaşık %X-%Y] (varsa)
- Gerekçe:                       [Bağlamsal bilgilere dayalı açıklama]
- Belirsizlik Notu:              [Eksik bilgi veya belirsizlikler]
- Güven Seviyesi:                [Yüksek / Orta / Düşük]

KISITLAR:
- Kesin sayısal tahmin YASAK (örn. "satış 45.000 adet olacak")
- Yalnızca verilen bilgilere dayan, dış bilgi ekleme
- Her önerinin somut bir iş gerekçesi olmalı"""

TEMPERATURES = [0.0, 0.7]


# ============================================================
# ADIM 3: MODEL FONKSİYONLARI
# ============================================================

def run_gpt(temperature: float) -> str:
    """
    GPT ile satış tahmini yorumlama analizi.

    Satış tahmini senaryosunda LLM'in rolü farklı:
    → Fraud'da: "Bu işlem fraud mu?" (sınıflandırma)
    → Satışta:  "Bu tahmin bağlamla uyuşuyor mu?" (yorumlama)
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir satış tahmini değerlendirme uzmanısın. "
                    "LSTM modelinin çıktılarını yorumlarsın, "
                    "ancak kendin sayısal tahmin üretmezsin."
                )
            },
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    return response.choices[0].message.content


def run_gemini(temperature: float) -> str:
    """Gemini ile satış tahmini yorumlama analizi."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=(
            "Sen bir satış tahmini değerlendirme uzmanısın. "
            "LSTM modelinin çıktılarını yorumlarsın, "
            "kendin sayısal tahmin üretmezsin."
        )
    )
    response = model.generate_content(
        USER_PROMPT,
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    return response.text


def run_cohere(temperature: float) -> str:
    """Cohere ile satış tahmini yorumlama analizi."""
    import cohere
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    response = co.chat(
        model="command-r-plus",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir satış tahmini değerlendirme uzmanısın. "
                    "LSTM modelinin çıktılarını yorumlarsın, "
                    "kendin sayısal tahmin üretmezsin."
                )
            },
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    return response.message.content[0].text


# ============================================================
# ADIM 4: SATIŞ TAHMİNİNE ÖZEL ANALİZ
# Fraud detection'dan farklı: burada aşırı yorum (over-interpretation)
# ve hallüsinasyon riski daha kritik.
# ============================================================

def satis_analizi_yap(sonuc_t0: str, sonuc_t7: str) -> dict:
    """
    Satış tahmini bağlamında iki temperature yanıtını karşılaştırır.

    Fraud detection analizinden farkı:
    - Sayısal tahmin üretip üretmediğini kontrol ediyoruz (yasak!)
    - Context dışı bağlam kullanımını tespit ediyoruz
    - Güven seviyesi ifadelerini sayıyoruz
    """

    # ── Yasak ifadeler: LLM kesinlikle sayısal tahmin vermemeli ──
    yasak_ifadeler = [
        "adet olacak", "birim satılacak", "units will be",
        "forecast is", "prediction:", "tahminim:"
    ]

    # ── Aşırı yorum belirteçleri ──
    asiri_yorum = [
        "sosyal medya", "viral", "influencer", "global trend",
        "ekonomik kriz", "döviz", "enflasyon"  # Context'te olmayan bilgiler
    ]

    # ── Sağlıklı belirsizlik ifadeleri (iyi şey!) ──
    saglikli_belirsizlik = [
        "belirlenemez", "yetersiz veri", "bilinmiyor",
        "cannot be determined", "insufficient data"
    ]

    def say(metin: str, kelimeler: list) -> int:
        return sum(metin.lower().count(k.lower()) for k in kelimeler)

    return {
        "temp_0.0": {
            "yasak_sayi_tahmini": say(sonuc_t0, yasak_ifadeler),
            "asiri_yorum_riski":  say(sonuc_t0, asiri_yorum),
            "saglikli_belirsiz":  say(sonuc_t0, saglikli_belirsizlik),
        },
        "temp_0.7": {
            "yasak_sayi_tahmini": say(sonuc_t7, yasak_ifadeler),
            "asiri_yorum_riski":  say(sonuc_t7, asiri_yorum),
            "saglikli_belirsiz":  say(sonuc_t7, saglikli_belirsizlik),
        },
        "uyari": (
            "⚠️ T=0.7 context dışı bilgi kullanmış olabilir!"
            if say(sonuc_t7, asiri_yorum) > 0 else "✅ Her iki temp context içinde kaldı"
        )
    }


# ============================================================
# ADIM 5: ANA DENEY FONKSİYONU
# ============================================================

def run_experiment():
    """Satış tahmini temperature deneyini çalıştırır."""
    print("=" * 65)
    print("SATIŞ TAHMİNİ – TEMPERATURE DENEYİ BAŞLADI")
    print(f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\nNot: Bu deney fraud detection'dan farklı!")
    print("     LLM'in tahmin üretmeden yorum yapıp yapamadığını test ediyoruz.\n")

    providers = {
        "GPT":    run_gpt,
        "Gemini": run_gemini,
        "Cohere": run_cohere
    }

    results = {}

    for provider_name, run_fn in providers.items():
        results[provider_name] = {}
        print(f"\n{'─' * 65}")
        print(f"  SAĞLAYICI: {provider_name}")
        print(f"{'─' * 65}")

        outputs = {}

        for temp in TEMPERATURES:
            print(f"\n  🌡️  Temperature = {temp}")
            print("  " + "·" * 50)

            try:
                baslangic = time.time()
                output    = run_fn(temp)
                gecen     = time.time() - baslangic

                results[provider_name][f"temp_{temp}"] = {
                    "output":      output,
                    "elapsed_sec": round(gecen, 2),
                    "char_count":  len(output)
                }
                outputs[temp] = output

                print(f"\n{output[:600]}{'...' if len(output) > 600 else ''}")
                print(f"\n  ⏱️  {gecen:.2f}s | {len(output)} karakter")

            except Exception as e:
                hata = f"HATA [{type(e).__name__}]: {str(e)}"
                results[provider_name][f"temp_{temp}"] = {"error": hata}
                outputs[temp] = ""
                print(f"\n  ❌ {hata}")

        # Satışa özel analiz
        if outputs.get(0.0) and outputs.get(0.7):
            analiz = satis_analizi_yap(outputs[0.0], outputs[0.7])
            results[provider_name]["analiz"] = analiz

            print(f"\n  📊 SATIŞ ANALİZİ ({provider_name}):")
            print(f"     T=0.0 → yasak sayısal tahmin: {analiz['temp_0.0']['yasak_sayi_tahmini']} | "
                  f"aşırı yorum: {analiz['temp_0.0']['asiri_yorum_riski']}")
            print(f"     T=0.7 → yasak sayısal tahmin: {analiz['temp_0.7']['yasak_sayi_tahmini']} | "
                  f"aşırı yorum: {analiz['temp_0.7']['asiri_yorum_riski']}")
            print(f"     {analiz['uyari']}")

    # Kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"sales_temperature_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'=' * 65}")
    print(f"DENEY TAMAMLANDI → {filename}")
    print("=" * 65)

    print("\n💡 DEĞERLENDİRME SORULARI:")
    print("   • T=0.7 context'te olmayan bilgiler ekledi mi?")
    print("   • T=0.0 'belirlenemez' ifadelerini doğru yerde kullandı mı?")
    print("   • Hangi temperature audit trail için daha uygun?")
    print("   • LLM hiç sayısal tahmin üretti mi? (Yasak olması gerekiyor!)")

    return results


if __name__ == "__main__":
    run_experiment()
