"""
========================================================
MODÜL 2 – TAM DENEY ÇALIŞTIRICI
Her iki case study'yi tek dosyadan çalıştırır.

Kullanım:
  python run_all_experiments.py --case fraud       # Sadece Fraud
  python run_all_experiments.py --case sales       # Sadece Satış
  python run_all_experiments.py --case both        # İkisi birden (varsayılan)
  python run_all_experiments.py --temp 0.2         # Sadece 1 temperature
  python run_all_experiments.py --provider gpt     # Sadece 1 sağlayıcı

Ortam değişkenleri (en az biri gerekli):
  export OPENAI_API_KEY="sk-..."
  export GOOGLE_API_KEY="AI..."
  export COHERE_API_KEY="..."
========================================================
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime


# ============================================================
# CONTEXT VERİLERİ
# ============================================================

FRAUD_CONTEXT = {
    "task": "fraud detection (binary classification)",
    "dataset_size": 1800000,
    "fraud_rate": 0.014,
    "models": ["logistic_regression", "xgboost", "zero_shot_llm_classifier"],
    "validation": {
        "logistic_regression":      {"precision": 0.81, "recall": 0.22},
        "xgboost":                  {"precision": 0.74, "recall": 0.61},
        "zero_shot_llm_classifier": {"precision": 0.66, "recall": 0.58}
    },
    "constraints": {
        "false_negatives_costly":         True,
        "manual_review_capacity_limited": True,
        "latency_ms_max":                 250
    }
}

SALES_CONTEXT = {
    "task":                   "sales forecast adjustment support",
    "dataset_size":           95000,
    "base_model":             "lstm_time_series",
    "forecast_horizon_days":  30,
    "evaluation_metric":      "smape",
    "base_model_smape":       13.2,
    "llm_adjusted_smape":     12.6,
    "constraints": {
        "human_approval_required":     True,
        "explainability_required":     True,
        "business_critical_decisions": True
    }
}

SALES_CONTEXTUAL_INFO = """
- Önümüzdeki 30 gün içinde Anneler Günü (5 Mayıs) bulunmaktadır.
- Rakip firma geçen hafta %30 indirim kampanyası başlattı.
- Hava tahminleri önümüzdeki 2 hafta normalin üstünde sıcaklık öngörüyor.
- Şirketimiz bu hafta sonu yeni bir ürün koleksiyonu lansmanı yapacak.
"""

SALES_LSTM_OUTPUT = {
    "forecast_period":  "2025-05-01 to 2025-05-30",
    "smape_confidence": 13.2,
    "predicted_units":  [
        1320, 1280, 1410, 1350, 1290, 1900, 2050,
        1310, 1270, 1390, 1340, 1280, 1870, 2020,
        1330, 1295, 1405, 1360, 1285, 1890, 2040,
        1315, 1275, 1395, 1345, 1288, 1880, 2030,
        1325, 1285
    ]
}


# ============================================================
# PROMPT ŞABLONLARI
# ============================================================

def build_fraud_prompt() -> str:
    ctx = json.dumps(FRAUD_CONTEXT, indent=2, ensure_ascii=False)
    return f"""Sen bir fraud detection karar destek sistemisin. \
Aşağıdaki CONTEXT_JSON verisini analiz ederek production ortamı için \
en uygun yaklaşımı belirle.

CONTEXT:
{ctx}

TALİMATLAR:
1. Yalnızca yukarıdaki verideki metrikleri ve kısıtları kullan. \
Varsayımda bulunma, veri dışı bilgi üretme.
2. Şu iş kısıtlarını değerlendir:
   - False negative'ler doğrudan finansal kayba yol açar
   - Manuel inceleme kapasitesi sınırlıdır
   - Sistemin yanıt süresi 250 ms altında olmalıdır
3. Her üç modeli precision, recall, F1-score, latency uygunluğu \
ve operasyonel sürdürülebilirlik açısından karşılaştır.
4. Tek bir production önerisi ver ve gerekçele.
5. Önerinin risklerini ve sonraki adımları listele.

ÇIKTI FORMATI:
- Recommendation: [seçilen model]
- Reasoning:      [maddeler halinde gerekçe]
- Risks:          [potansiyel riskler]
- Next Actions:   [önerilen aksiyonlar]

ÖNEMLİ: Emin olmadığın noktalarda "bu veri seti ile belirlenemez" de."""


def build_sales_prompt() -> str:
    ctx  = json.dumps(SALES_CONTEXT,      indent=2, ensure_ascii=False)
    lstm = json.dumps(SALES_LSTM_OUTPUT,  indent=2, ensure_ascii=False)
    return f"""Sen bir satış tahmini değerlendirme uzmanısın. \
LSTM modelinin ürettiği tahmini yorumla ve ayarlama gerekip \
gerekmediğini değerlendir. Kendin sayısal tahmin ÜRETME.

SİSTEM BİLGİSİ:
{ctx}

BAĞLAMSAL BİLGİLER:
{SALES_CONTEXTUAL_INFO}

LSTM ÇIKTISI:
{lstm}

TALİMATLAR:
1. LSTM tahminini bağlamsal bilgilerle değerlendir.
2. Ayarlama yönünü (yukarı/aşağı) ve yüzde aralığını belirt \
— kesin sayı VERME.
3. Her görüşün için açık, izlenebilir gerekçe sun.
4. Emin olmadığın durumlarda "bu bilgiyle belirlenemez" de.

ÇIKTI FORMATI:
- Mevcut Tahmin Değerlendirmesi: [güçlü/zayıf yönler]
- Ayarlama Önerisi:              [Gerekli / Gerekli Değil]
- Ayarlama Yönü ve Aralığı:      [Yukarı/Aşağı %X-%Y] (varsa)
- Gerekçe:                       [bağlamsal açıklama]
- Belirsizlik Notu:              [eksik bilgi / belirsizlikler]
- Güven Seviyesi:                [Yüksek / Orta / Düşük]"""


# ============================================================
# MODEL ÇAĞIRI FONKSİYONLARI
# ============================================================

def call_gpt(prompt: str, temperature: float, system_msg: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt}
        ]
    )
    return r.choices[0].message.content


def call_gemini(prompt: str, temperature: float, system_msg: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_msg
    )
    r = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    return r.text


def call_cohere(prompt: str, temperature: float, system_msg: str) -> str:
    import cohere
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    r  = co.chat(
        model="command-r-plus",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt}
        ]
    )
    return r.message.content[0].text


# ============================================================
# TEK ÇAĞRI SARMALAYICI
# Mevcut key'lere göre hangi provider'ın çalışacağını belirler.
# ============================================================

PROVIDER_CONFIG = {
    "gpt": {
        "fn":         call_gpt,
        "env_key":    "OPENAI_API_KEY",
        "system_fraud": "Sen bir fraud detection uzmanısın.",
        "system_sales": (
            "Sen bir satış tahmini değerlendirme uzmanısın. "
            "Sayısal tahmin üretmezsin, yalnızca yorumlarsın."
        ),
    },
    "gemini": {
        "fn":         call_gemini,
        "env_key":    "GOOGLE_API_KEY",
        "system_fraud": "Sen bir fraud detection uzmanısın.",
        "system_sales": (
            "Sen bir satış tahmini değerlendirme uzmanısın. "
            "Sayısal tahmin üretmezsin, yalnızca yorumlarsın."
        ),
    },
    "cohere": {
        "fn":         call_cohere,
        "env_key":    "COHERE_API_KEY",
        "system_fraud": "Sen bir fraud detection uzmanısın.",
        "system_sales": (
            "Sen bir satış tahmini değerlendirme uzmanısın. "
            "Sayısal tahmin üretmezsin, yalnızca yorumlarsın."
        ),
    },
}


def get_active_providers(filter_provider: str = None) -> dict:
    """
    Ortam değişkenlerinde key'i olan provider'ları döndürür.
    filter_provider belirtilirse yalnızca o provider çalışır.
    """
    active = {}
    for name, cfg in PROVIDER_CONFIG.items():
        if filter_provider and filter_provider.lower() != name:
            continue
        if os.getenv(cfg["env_key"]):
            active[name] = cfg
        else:
            print(f"  ⚠️  {name.upper()} atlandı — {cfg['env_key']} bulunamadı.")
    return active


# ============================================================
# DENEY ÇALIŞTIRICI
# ============================================================

def run_case(
    case_name: str,
    prompt: str,
    system_key: str,
    temperatures: list,
    providers: dict
) -> dict:
    """
    Tek bir case study için tüm provider × temperature kombinasyonlarını çalıştırır.

    case_name   : "fraud" veya "sales"
    prompt      : Kullanılacak user prompt
    system_key  : PROVIDER_CONFIG'daki system mesajı anahtarı
    temperatures: Denenecek temperature listesi
    providers   : Aktif provider'lar sözlüğü
    """
    results = {}

    print(f"\n{'═' * 65}")
    print(f"  CASE STUDY: {case_name.upper()}")
    print(f"{'═' * 65}")

    for pname, cfg in providers.items():
        results[pname] = {}
        print(f"\n  ── Sağlayıcı: {pname.upper()} ──")

        for temp in temperatures:
            print(f"\n  🌡️  Temperature = {temp}")

            try:
                t0     = time.time()
                output = cfg["fn"](prompt, temp, cfg[system_key])
                elapsed = round(time.time() - t0, 2)

                results[pname][f"temp_{temp}"] = {
                    "output":      output,
                    "elapsed_sec": elapsed,
                    "char_count":  len(output)
                }

                # Önizleme
                preview = output[:500] + ("…" if len(output) > 500 else "")
                print(f"\n{preview}")
                print(f"\n  ⏱️  {elapsed}s | {len(output)} karakter")

            except Exception as e:
                err = f"HATA [{type(e).__name__}]: {e}"
                results[pname][f"temp_{temp}"] = {"error": err}
                print(f"\n  ❌ {err}")

    return results


# ============================================================
# KARŞILAŞTIRMA RAPORU
# ============================================================

def print_comparison_report(all_results: dict):
    """
    Tüm sonuçları yan yana karşılaştıran özet tablo yazdırır.
    """
    print(f"\n\n{'═' * 65}")
    print("  KARŞILAŞTIRMA RAPORU")
    print(f"{'═' * 65}")

    for case_name, case_data in all_results.items():
        print(f"\n  📌 {case_name.upper()} CASE")
        print(f"  {'Sağlayıcı':<10} {'T=0.0 (kar.)':<14} {'T=0.7 (kar.)':<14} "
              f"{'T=0.0 süre':<12} {'T=0.7 süre':<12}")
        print("  " + "─" * 62)

        for provider, pdata in case_data.items():
            t0_info = pdata.get("temp_0.0", {})
            t7_info = pdata.get("temp_0.7", {})

            t0_chars = t0_info.get("char_count", "-")
            t7_chars = t7_info.get("char_count", "-")
            t0_time  = f"{t0_info.get('elapsed_sec', '-')}s"
            t7_time  = f"{t7_info.get('elapsed_sec', '-')}s"

            print(f"  {provider.upper():<10} {str(t0_chars):<14} "
                  f"{str(t7_chars):<14} {t0_time:<12} {t7_time:<12}")

    print(f"\n{'═' * 65}")
    print("  HATIRLATMA: Temperature etkisini değerlendirirken sorun:")
    print("  • T=0.7'de 'belki/muhtemelen/olabilir' arttı mı?")
    print("  • Fraud: T=0.7'de 'XGBoost önerilir' kesinliği kayboldu mu?")
    print("  • Satış: T=0.7'de context dışı bilgi eklendi mi?")
    print(f"{'═' * 65}")


# ============================================================
# ANA PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Modül 2 – Temperature Deneyi Çalıştırıcısı"
    )
    parser.add_argument(
        "--case", choices=["fraud", "sales", "both"], default="both",
        help="Hangi case study çalışsın? (varsayılan: both)"
    )
    parser.add_argument(
        "--temp", type=float, nargs="+", default=[0.0, 0.7],
        help="Temperature değerleri (varsayılan: 0.0 0.7)"
    )
    parser.add_argument(
        "--provider", choices=["gpt", "gemini", "cohere"], default=None,
        help="Belirli bir sağlayıcı seç (varsayılan: key bulunanlar)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  MODÜL 2 – TEMPERATURE DENEYİ")
    print(f"  Zaman     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Case      : {args.case}")
    print(f"  Temp. ler : {args.temp}")
    print(f"  Provider  : {args.provider or 'otomatik (key bulunanlar)'}")
    print("=" * 65)

    # Aktif provider'ları belirle
    providers = get_active_providers(args.provider)

    if not providers:
        print("\n❌ Hiç aktif provider bulunamadı!")
        print("   En az bir API key'i ortam değişkeni olarak ayarlayın:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export GOOGLE_API_KEY='AI...'")
        print("   export COHERE_API_KEY='...'")
        sys.exit(1)

    print(f"\n✅ Aktif provider'lar: {', '.join(p.upper() for p in providers)}")

    # Sonuçları topla
    all_results = {}

    if args.case in ("fraud", "both"):
        all_results["fraud"] = run_case(
            case_name   = "fraud",
            prompt      = build_fraud_prompt(),
            system_key  = "system_fraud",
            temperatures = args.temp,
            providers   = providers
        )

    if args.case in ("sales", "both"):
        all_results["sales"] = run_case(
            case_name   = "sales",
            prompt      = build_sales_prompt(),
            system_key  = "system_sales",
            temperatures = args.temp,
            providers   = providers
        )

    # Karşılaştırma raporu
    print_comparison_report(all_results)

    # JSON'a kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"modul2_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Tüm sonuçlar kaydedildi: {filename}")
    print("\nSonraki adım → Görev 4 için temperature=0.2 ile çalıştırın:")
    print(f"  python run_all_experiments.py --temp 0.2")


if __name__ == "__main__":
    main()
