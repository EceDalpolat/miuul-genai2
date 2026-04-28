"""
========================================================
FRAUD DETECTION - TEMPERATURE DENEYİ
GenAI Modül 2 – Görev 3 (Case Study 1)

Ne yapıyor bu kod?
──────────────────
Aynı fraud detection sorusunu, aynı modellere (GPT / Gemini / Cohere)
iki farklı temperature ile soruyoruz:
   Temperature = 0.0  →  Deterministik, kesin kararlar
   Temperature = 0.7  →  Yaratıcı, bazen spekülatif kararlar

Neden önemli?
─────────────
Production fraud detection sistemlerinde tutarlılık şarttır.
Temperature'ın karar diline, risk vurgusuna ve spekülasyon
eğilimine etkisini gözlemlemek bu deneyin asıl amacıdır.

Kurulum:
────────
   pip install openai google-generativeai cohere

   export OPENAI_API_KEY="sk-..."
   export GOOGLE_API_KEY="AI..."
   export COHERE_API_KEY="..."

Çalıştırma:
───────────
   python fraud_temperature_experiment.py
========================================================
"""

import os
import json
import time
from datetime import datetime


# ============================================================
# ADIM 1: CONTEXT HAZIRLA
# Context = Modele verdiğimiz "problem tanımı"
# Aynı context her iki temperature'da da kullanılacak
# ============================================================

CONTEXT_JSON = json.dumps({
    "task": "fraud detection (binary classification)",
    "dataset_size": 1800000,
    "fraud_rate": 0.014,
    "models": ["logistic_regression", "xgboost", "zero_shot_llm_classifier"],
    "validation": {
        "logistic_regression": {"precision": 0.81, "recall": 0.22},
        "xgboost":             {"precision": 0.74, "recall": 0.61},
        "zero_shot_llm_classifier": {"precision": 0.66, "recall": 0.58}
    },
    "constraints": {
        "false_negatives_costly":         True,   # Kaçan fraud = finansal kayıp
        "manual_review_capacity_limited": True,   # Yanlış alarm = operasyonel yük
        "latency_ms_max":                 250     # 250ms'den yavaş olursa ödeme bozulur
    }
}, indent=2, ensure_ascii=False)


# ============================================================
# ADIM 2: USER PROMPT TASARIMI
# Prompt tasarımında 3 kritik kural:
#   1. Production kararı istemek   → "production ortamı için en uygun"
#   2. İş kısıtlarını dahil etmek  → False negative, latency, kapasite
#   3. Hallüsinasyonu engellemek   → "varsayımda bulunma", "belirlenemez de"
# ============================================================

USER_PROMPT = f"""Sen bir fraud detection karar destek sistemisin. \
Aşağıdaki CONTEXT_JSON verisini analiz ederek production ortamı için \
en uygun yaklaşımı belirle.

CONTEXT:
{CONTEXT_JSON}

TALİMATLAR:
1. Yalnızca yukarıdaki verideki metrikleri ve kısıtları kullan. \
Varsayımda bulunma, veri dışı bilgi üretme.

2. Şu iş kısıtlarını değerlendir:
   - False negative'ler doğrudan finansal kayba yol açar (yüksek maliyet)
   - Manuel inceleme kapasitesi sınırlıdır (false positive'ler operasyonel yük yaratır)
   - Sistemin yanıt süresi 250 ms altında olmalıdır

3. Her üç modeli precision, recall, F1-score, latency uygunluğu ve \
operasyonel sürdürülebilirlik açısından karşılaştır.

4. Tek bir production önerisi ver ve gerekçele.

5. Önerinin risklerini ve sonraki adımları listele.

ÇIKTI FORMATI:
- Recommendation: [seçilen model]
- Reasoning:      [maddeler halinde gerekçe]
- Risks:          [potansiyel riskler]
- Next Actions:   [önerilen aksiyonlar]

ÖNEMLİ: Yalnızca verilen veriye dayanarak yanıt ver. \
Emin olmadığın noktalarda "bu veri seti ile belirlenemez" de."""


# Deneyde kullanılacak temperature değerleri
TEMPERATURES = [0.0, 0.7]


# ============================================================
# ADIM 3: MODEL FONKSİYONLARI
# Her model için ayrı fonksiyon yazıyoruz.
# Aynı prompt + aynı context → sadece temperature değişiyor.
# ============================================================

def run_gpt(temperature: float) -> str:
    """
    GPT (OpenAI) ile fraud detection analizi yapar.

    temperature: 0.0 = kesin/deterministik, 0.7 = yaratıcı/değişken
    Döndürür: Modelin metin yanıtı
    """
    from openai import OpenAI

    # API istemcisi oluştur (key environment variable'dan okunuyor)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # API çağrısı yap
    response = client.chat.completions.create(
        model="gpt-4o",             # Kullanılan model versiyonu
        temperature=temperature,    # 0.0 = deterministik, 0.7 = çeşitli
        messages=[
            # System mesajı: modele rolünü tanımlıyoruz
            {"role": "system", "content": "Sen bir fraud detection uzmanısın."},
            # User mesajı: asıl sorumuzu soruyoruz
            {"role": "user",   "content": USER_PROMPT}
        ]
    )

    # Yanıtın ilk seçeneğini (choices[0]) ve mesaj içeriğini al
    return response.choices[0].message.content


def run_gemini(temperature: float) -> str:
    """
    Gemini (Google) ile fraud detection analizi yapar.

    Not: Gemini'de system mesajı doğrudan prompt'a ekleniyor,
    çünkü Gemini API mesaj rollerini farklı işliyor.
    """
    import google.generativeai as genai

    # API yapılandırması
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Model nesnesi oluştur
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        # Sistem talimatını burada veriyoruz
        system_instruction="Sen bir fraud detection uzmanısın."
    )

    # İçerik üret - GenerationConfig ile temperature ayarla
    response = model.generate_content(
        USER_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature   # 0.0 veya 0.7
        )
    )

    return response.text


def run_cohere(temperature: float) -> str:
    """
    Cohere Command ile fraud detection analizi yapar.

    Cohere'in chat API'si de OpenAI'a benzer mesaj yapısı kullanıyor.
    """
    import cohere

    # API istemcisi (V2 = yeni sürüm API)
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    response = co.chat(
        model="command-r-plus",     # Cohere'in en gelişmiş modeli
        temperature=temperature,    # Temperature parametresi
        messages=[
            # Cohere'de system rolü ayrı verilebiliyor
            {"role": "system",  "content": "Sen bir fraud detection uzmanısın."},
            {"role": "user",    "content": USER_PROMPT}
        ]
    )

    # Cohere'in yanıt yapısı: message.content[0].text
    return response.message.content[0].text


# ============================================================
# ADIM 4: ANALİZ FONKSİYONU
# Temperature farkını sayısal olarak ölçmeye çalışıyoruz.
# ============================================================

def temperature_farkini_analiz_et(sonuc_t0: str, sonuc_t7: str) -> dict:
    """
    İki temperature sonucunu karşılaştırır.
    Basit kelime analizi ile spekülatif dili tespit eder.

    Bu otomatik analiz tamamlayıcıdır - asıl değerlendirme
    katılımcılar tarafından yapılacaktır.
    """
    # Spekülatif dil belirteçleri (yüksek temperature'da daha çok görülür)
    spekulatif_kelimeler = [
        "belki", "muhtemelen", "olabilir", "olası", "potansiyel",
        "değerlendirilebilir", "düşünülebilir", "öngörülebilir",
        "ihtimalli", "sanıyorum", "possibly", "might", "could",
        "perhaps", "potentially", "likely", "probably"
    ]

    # Kesin karar dili belirteçleri (düşük temperature'da daha çok görülür)
    kesin_kelimeler = [
        "önerilir", "seçilmeli", "tercih edilmeli", "deploy edilmeli",
        "kullanılmalıdır", "uygulanmalıdır", "recommended", "should",
        "must", "required", "optimal"
    ]

    def say(metin: str, kelimeler: list) -> int:
        """Metinde verilen kelimelerin kaç kez geçtiğini sayar."""
        metin_kucuk = metin.lower()
        return sum(metin_kucuk.count(k) for k in kelimeler)

    return {
        "temp_0.0": {
            "karakter_sayisi"  : len(sonuc_t0),
            "spekulatif_ifade" : say(sonuc_t0, spekulatif_kelimeler),
            "kesin_karar_dili" : say(sonuc_t0, kesin_kelimeler),
        },
        "temp_0.7": {
            "karakter_sayisi"  : len(sonuc_t7),
            "spekulatif_ifade" : say(sonuc_t7, spekulatif_kelimeler),
            "kesin_karar_dili" : say(sonuc_t7, kesin_kelimeler),
        },
        "yorum": (
            "Temperature=0.7 daha spekülatif görünüyor"
            if say(sonuc_t7, spekulatif_kelimeler) > say(sonuc_t0, spekulatif_kelimeler)
            else "İki temperature benzer spekülatif dil kullandı"
        )
    }


# ============================================================
# ADIM 5: ANA DENEY FONKSİYONU
# Tüm modeller × tüm temperature'lar için çalıştırır,
# sonuçları hem ekrana yazdırır hem dosyaya kaydeder.
# ============================================================

def run_experiment():
    """
    Fraud detection temperature deneyini çalıştırır.

    Akış:
      For her model (GPT, Gemini, Cohere):
        For her temperature (0.0, 0.7):
          → API çağrısı yap
          → Sonucu kaydet
      → Tüm sonuçları JSON'a yaz
      → Karşılaştırmalı analiz yap
    """
    print("=" * 65)
    print("FRAUD DETECTION – TEMPERATURE DENEYİ BAŞLADI")
    print(f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Model ismi → çalıştırma fonksiyonu eşleşmesi
    providers = {
        "GPT":    run_gpt,
        "Gemini": run_gemini,
        "Cohere": run_cohere
    }

    # Tüm sonuçları bu sözlükte tutacağız
    results = {}

    for provider_name, run_fn in providers.items():
        results[provider_name] = {}
        print(f"\n{'─' * 65}")
        print(f"  SAĞLAYICI: {provider_name}")
        print(f"{'─' * 65}")

        outputs = {}  # Bu provider'ın her temperature çıktısını tut

        for temp in TEMPERATURES:
            print(f"\n  🌡️  Temperature = {temp}")
            print("  " + "·" * 50)

            try:
                # API çağrısı - gerçek zaman ölç
                baslangic = time.time()
                output = run_fn(temp)
                gecen_sure = time.time() - baslangic

                # Sonucu kaydet
                results[provider_name][f"temp_{temp}"] = {
                    "output":        output,
                    "elapsed_sec":   round(gecen_sure, 2),
                    "char_count":    len(output)
                }
                outputs[temp] = output

                # İlk 600 karakteri önizle
                onizleme = output[:600] + ("..." if len(output) > 600 else "")
                print(f"\n{onizleme}")
                print(f"\n  ⏱️  Yanıt süresi: {gecen_sure:.2f}s | {len(output)} karakter")

            except Exception as e:
                hata = f"HATA [{type(e).__name__}]: {str(e)}"
                results[provider_name][f"temp_{temp}"] = {"error": hata}
                outputs[temp] = ""
                print(f"\n  ❌ {hata}")
                print("  → API key veya kütüphane kurulumunu kontrol edin.")

        # İki temperature arasındaki farkı analiz et (sadece her iki çıktı da varsa)
        if 0.0 in outputs and 0.7 in outputs and outputs[0.0] and outputs[0.7]:
            analiz = temperature_farkini_analiz_et(outputs[0.0], outputs[0.7])
            results[provider_name]["analiz"] = analiz

            print(f"\n  📊 OTOMATİK ANALİZ ({provider_name}):")
            print(f"     Temperature 0.0 → spekülatif ifade sayısı: "
                  f"{analiz['temp_0.0']['spekulatif_ifade']}")
            print(f"     Temperature 0.7 → spekülatif ifade sayısı: "
                  f"{analiz['temp_0.7']['spekulatif_ifade']}")
            print(f"     Yorum: {analiz['yorum']}")

    # ── Sonuçları JSON dosyasına kaydet ──
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"fraud_temperature_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'=' * 65}")
    print(f"DENEY TAMAMLANDI → Sonuçlar: {filename}")
    print("=" * 65)

    # ── Özet tablo ──
    print("\nÖZET TABLO:")
    print(f"{'Model':<10} {'T=0.0 Uzunluk':>15} {'T=0.7 Uzunluk':>15} "
          f"{'T=0.0 Spek.':>13} {'T=0.7 Spek.':>13}")
    print("─" * 70)

    for provider, data in results.items():
        t0 = data.get("temp_0.0", {})
        t7 = data.get("temp_0.7", {})
        an = data.get("analiz",   {})

        t0_len  = t0.get("char_count", "-")
        t7_len  = t7.get("char_count", "-")
        t0_spek = an.get("temp_0.0", {}).get("spekulatif_ifade", "-")
        t7_spek = an.get("temp_0.7", {}).get("spekulatif_ifade", "-")

        print(f"{provider:<10} {str(t0_len):>15} {str(t7_len):>15} "
              f"{str(t0_spek):>13} {str(t7_spek):>13}")

    print("\n💡 HATIRLATMA: Spekülatif ifade sayısı otomatik tespittir.")
    print("   Asıl değerlendirmeyi çıktıları okuyarak kendiniz yapın.")
    print("   Bakılacak sorular:")
    print("   • T=0.0 'önerilir/deploy edilmeli' mi diyor?")
    print("   • T=0.7 'belki/olabilir' gibi belirsiz dil mi kullanıyor?")
    print("   • T=0.7 context'te olmayan bilgi uyduruyor mu?")

    return results


# ── Program başlangıcı ──
if __name__ == "__main__":
    run_experiment()
