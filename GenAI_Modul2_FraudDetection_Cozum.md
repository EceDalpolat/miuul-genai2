# GENAI – MODUL 2 – CASE STUDY ÇÖZÜM
## Fraud Detection Case Study: Dengesiz Verilerde Karar Verme

---

## GÖREV 1 – Metriklerin Teknik Yorumu

### Precision ve Recall Neden Kritiktir?

Fraud detection gibi dengesiz sınıflandırma problemlerinde accuracy yanıltıcı bir metriktir. Veri setinde yalnızca %1.4 fraud varken, hiçbir işlemi fraud olarak etiketlemeyen naif bir model bile %98.6 accuracy elde eder — ama bu model tamamen işe yaramazdır. Bu nedenle precision ve recall metrikleri asıl karar mekanizmasını oluşturur.

**Precision (Kesinlik) = TP / (TP + FP):** Modelin "fraud" dediği işlemlerin gerçekten kaçının fraud olduğunu gösterir. Düşük precision → yüksek false positive → meşru müşteri işlemleri gereksiz yere bloke edilir. Bu durum operasyonel maliyeti artırır: manuel inceleme ekibi aşırı yüklenir, müşteri deneyimi bozulur ve potansiyel gelir kaybı yaşanır. Örneğin Logistic Regression'ın 0.81 precision'ı, alarm verdiğinde %81 oranında haklı olduğunu gösterir — bu iyi, ama yeterli değildir çünkü recall çok düşüktür.

**Recall (Duyarlılık) = TP / (TP + FN):** Gerçek fraud işlemlerin ne kadarının yakalandığını gösterir. Düşük recall → yüksek false negative → dolandırıcılık vakaları gözden kaçar. Bu doğrudan finansal kayıptır. Logistic Regression'ın 0.22 recall'u, gerçek fraud'ların %78'inin kaçırıldığı anlamına gelir. 1.8M işlem × %1.4 fraud rate = ~25.200 fraud işlem. Bunların %78'i = ~19.656 fraud kaçırılır — bu ciddi bir finansal risk.

**İş maliyeti perspektifi:**

| Metrik | Düşük Olursa | İş Etkisi |
|--------|-------------|-----------|
| Precision ↓ | FP artar | Manuel inceleme yükü, müşteri memnuniyetsizliği, operasyonel maliyet |
| Recall ↓ | FN artar | Doğrudan finansal kayıp, regülasyon riski, itibar zararı |

Bu senaryoda **false negative'ler** false positive'lerden çok daha maliyetlidir. Kaçırılan her fraud işlemi doğrudan para kaybı demektir. Bu nedenle model seçiminde recall'a öncelik verilmeli, ancak precision de operasyonel kapasite dahilinde tutulmalıdır. F1-score veya daha iyisi F-beta (beta > 1, recall ağırlıklı) bu denge için uygun bir karar metriğidir.

**Modellerin karşılaştırması:**

| Model | Precision | Recall | F1-Score | Yorum |
|-------|-----------|--------|----------|-------|
| Logistic Regression | 0.81 | 0.22 | 0.346 | Çok muhafazakâr, fraud'ların büyük kısmını kaçırıyor |
| XGBoost | 0.74 | 0.61 | 0.669 | En dengeli precision-recall profili |
| Zero-shot LLM | 0.66 | 0.58 | 0.617 | Etiketleme gerektirmez ama doğruluk düşük |

---

## GÖREV 2 – User Prompt Tasarımı

### Tasarlanan Prompt

```
Sen bir fraud detection karar destek sistemisin. Aşağıdaki CONTEXT_JSON verisini analiz ederek production ortamı için en uygun yaklaşımı belirle.

CONTEXT:
{context_json}

TALİMATLAR:
1. Yalnızca yukarıdaki verideki metrikleri ve kısıtları kullan. Varsayımda bulunma, veri dışı bilgi üretme.
2. Şu iş kısıtlarını değerlendir:
   - False negative'ler doğrudan finansal kayba yol açar (yüksek maliyet)
   - Manuel inceleme kapasitesi sınırlıdır (false positive'ler operasyonel yük yaratır)
   - Sistemin yanıt süresi 250 ms altında olmalıdır
3. Her üç modeli (logistic_regression, xgboost, zero_shot_llm_classifier) precision, recall, F1-score, latency uygunluğu ve operasyonel sürdürülebilirlik açısından karşılaştır.
4. Tek bir production önerisi ver ve gerekçele.
5. Önerinin risklerini ve sonraki adımları listele.

ÇIKTI FORMATI:
- Recommendation: [seçilen model]
- Reasoning: [maddeler halinde gerekçe]
- Risks: [potansiyel riskler]
- Next Actions: [önerilen aksiyonlar]

ÖNEMLİ: Yalnızca verilen veriye dayanarak yanıt ver. Emin olmadığın noktalarda "bu veri seti ile belirlenemez" de.
```

### Prompt Tasarım Gerekçesi

Bu prompt aşağıdaki hedefleri karşılar:

**Production kararı istemesi:** "production ortamı için en uygun yaklaşımı belirle" ifadesiyle net bir karar beklentisi oluşturulur. Belirsiz veya genel bir analiz yerine somut bir öneri zorunlu kılınır.

**İş kısıtlarının dikkate alınması:** False negative maliyeti, manuel inceleme kapasitesi ve 250 ms latency kısıtı açıkça talimat olarak verilir. Model sadece metriklere bakarak değil, operasyonel bağlamı da göz önünde bulundurarak karar verir.

**Varsayım ve uydurma bilgiyi engellemesi:** "Yalnızca yukarıdaki verideki metrikleri kullan", "Varsayımda bulunma" ve "Emin olmadığın noktalarda belirlenemez de" ifadeleri hallüsinasyonu minimize eder. Structured output formatı da modeli disipline eder.

---

## GÖREV 3 – Temperature Deneyi

### Python Kodu

Aşağıdaki Python kodu, GPT, Gemini ve Cohere modelleri için temperature=0.0 ve temperature=0.7 ile aynı prompt'u çalıştırır:

```python
import os
import json
from datetime import datetime

# ============================================================
# CONTEXT & PROMPT
# ============================================================
CONTEXT_JSON = json.dumps({
    "task": "fraud detection (binary classification)",
    "dataset_size": 1800000,
    "fraud_rate": 0.014,
    "models": ["logistic_regression", "xgboost", "zero_shot_llm_classifier"],
    "validation": {
        "logistic_regression": {"precision": 0.81, "recall": 0.22},
        "xgboost": {"precision": 0.74, "recall": 0.61},
        "zero_shot_llm_classifier": {"precision": 0.66, "recall": 0.58}
    },
    "constraints": {
        "false_negatives_costly": True,
        "manual_review_capacity_limited": True,
        "latency_ms_max": 250
    }
}, indent=2, ensure_ascii=False)

USER_PROMPT = f"""Sen bir fraud detection karar destek sistemisin. Aşağıdaki CONTEXT_JSON verisini analiz ederek production ortamı için en uygun yaklaşımı belirle.

CONTEXT:
{CONTEXT_JSON}

TALİMATLAR:
1. Yalnızca yukarıdaki verideki metrikleri ve kısıtları kullan. Varsayımda bulunma, veri dışı bilgi üretme.
2. Şu iş kısıtlarını değerlendir:
   - False negative'ler doğrudan finansal kayba yol açar (yüksek maliyet)
   - Manuel inceleme kapasitesi sınırlıdır (false positive'ler operasyonel yük yaratır)
   - Sistemin yanıt süresi 250 ms altında olmalıdır
3. Her üç modeli precision, recall, F1-score, latency uygunluğu ve operasyonel sürdürülebilirlik açısından karşılaştır.
4. Tek bir production önerisi ver ve gerekçele.
5. Önerinin risklerini ve sonraki adımları listele.

ÇIKTI FORMATI:
- Recommendation: [seçilen model]
- Reasoning: [maddeler halinde gerekçe]
- Risks: [potansiyel riskler]
- Next Actions: [önerilen aksiyonlar]

ÖNEMLİ: Yalnızca verilen veriye dayanarak yanıt ver. Emin olmadığın noktalarda "bu veri seti ile belirlenemez" de."""

TEMPERATURES = [0.0, 0.7]

# ============================================================
# GPT (OpenAI)
# ============================================================
def run_gpt(temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {"role": "system", "content": "Sen bir fraud detection uzmanısın."},
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    return response.choices[0].message.content

# ============================================================
# Gemini (Google)
# ============================================================
def run_gemini(temperature: float) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        USER_PROMPT,
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    return response.text

# ============================================================
# Cohere Command
# ============================================================
def run_cohere(temperature: float) -> str:
    import cohere
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    response = co.chat(
        model="command-r-plus",
        temperature=temperature,
        messages=[{"role": "user", "content": USER_PROMPT}]
    )
    return response.message.content[0].text

# ============================================================
# DENEY ÇALIŞTIRICI
# ============================================================
def run_experiment():
    results = {}
    providers = {
        "GPT (OpenAI)": run_gpt,
        "Gemini (Google)": run_gemini,
        "Cohere Command": run_cohere
    }

    for provider_name, run_fn in providers.items():
        results[provider_name] = {}
        for temp in TEMPERATURES:
            print(f"\n{'='*60}")
            print(f"[{provider_name}] Temperature = {temp}")
            print('='*60)
            try:
                output = run_fn(temp)
                results[provider_name][f"temp_{temp}"] = output
                print(output)
            except Exception as e:
                error_msg = f"HATA: {str(e)}"
                results[provider_name][f"temp_{temp}"] = error_msg
                print(error_msg)

    # Sonuçları dosyaya kaydet
    with open("temperature_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n\nSonuçlar 'temperature_experiment_results.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_experiment()
```

### Temperature Deneyi Analizi

Temperature parametresi, LLM'lerin softmax çıktısındaki olasılık dağılımını kontrol eder. Düşük temperature dağılımı sivri (peaked) yapar ve en yüksek olasılıklı token seçilir; yüksek temperature dağılımı düzleştirir ve daha "yaratıcı" (rastgele) seçimler yapılır.

**Temperature = 0.0 ile beklenen davranış:**

Karar dili deterministik ve kesindir. Model "XGBoost önerilir" gibi net ifadeler kullanır. Risk vurgusu verideki somut metriklere dayanır, spekülatif ifade ("belki", "olabilir", "potansiyel olarak") minimum düzeydedir. Aynı prompt tekrar çalıştırıldığında neredeyse aynı çıktı üretilir. Production kararı için bu davranış tercih edilir çünkü tutarlılık ve tekrarlanabilirlik sağlar.

**Temperature = 0.7 ile beklenen davranış:**

Karar dili daha esnek ve nüanslıdır. Model "XGBoost tercih edilebilir, ancak şu koşullarda LLM de değerlendirilebilir" gibi alternatif senaryolar üretebilir. Risk vurgusu genişler — veri setinde olmayan potansiyel senaryolar (ör. "mevsimsel fraud pattern'leri", "model drift") gibi spekülatif riskler eklenebilir. Bu hallüsinasyon riski taşır. Aynı prompt tekrar çalıştırıldığında farklı çıktılar üretilir — bu production kararı için kabul edilemez bir tutarsızlıktır.

**Karşılaştırmalı Etki Tablosu:**

| Boyut | Temperature = 0.0 | Temperature = 0.7 |
|-------|-------------------|-------------------|
| Karar dili | Kesin, deterministik ("önerilir") | Esnek, alternatifli ("değerlendirilebilir") |
| Risk vurgusu | Veriye dayalı, somut | Genişletilmiş, spekülatif unsurlar içerebilir |
| Spekülasyon eğilimi | Düşük | Yüksek — context dışı bilgi üretme riski |
| Tekrarlanabilirlik | Yüksek (aynı çıktı) | Düşük (her seferinde farklı) |
| Production uygunluğu | Uygun | Riskli |

**Sonuç:** Fraud detection gibi yüksek riskli karar süreçlerinde temperature = 0.0–0.2 aralığı tercih edilmelidir. Yüksek temperature, brainstorming veya yaratıcı görevler için uygun olsa da, finansal kararlar için deterministik çıktı zorunludur.

---

## GÖREV 4 – Nihai Production Kararı

*Referans temperature: 0.2 (düşük varyans, yüksek tutarlılık)*

---

### Model: GPT (OpenAI)

**Recommendation:**
XGBoost – primary fraud detection modeli olarak deploy edilmeli; LLM edge-case analizi için tamamlayıcı rol üstlenmeli.

**Reasoning:**
- XGBoost, 0.74 precision ve 0.61 recall ile en dengeli F1 skorunu (0.669) sunar. False negative'lerin yüksek maliyetli olduğu bu senaryoda 0.61 recall, Logistic Regression'ın 0.22'sine göre ~2.8 kat daha fazla fraud yakalama anlamına gelir.
- XGBoost tree-based inference süresi tipik olarak 1-10 ms aralığındadır — 250 ms latency kısıtını rahatlıkla karşılar. LLM inference ise yüzlerce ms ile saniyelerce sürebilir ve bu kısıtı aşma riski taşır.
- 0.74 precision, manuel inceleme kapasitesini Logistic Regression'a yakın bir seviyede tutarken çok daha fazla fraud yakalar. Her 100 alarm'dan 74'ü gerçek fraud — operasyonel olarak yönetilebilir bir oran.
- XGBoost, SMOTE, class_weight ayarı veya threshold tuning ile recall'u daha da artırma potansiyeline sahiptir.

**Risks:**
- 0.61 recall hâlâ fraud'ların %39'unun kaçırılması demektir (~9.828 fraud işlem). Threshold optimizasyonu ile iyileştirilmezse kayıp devam eder.
- Model zamanla degrade olabilir (data drift / concept drift). Dolandırıcılar taktik değiştirdikçe modelin performansı düşer.
- XGBoost feature engineering'e bağımlıdır; yeni fraud pattern'leri için feature seti güncellenmelidir.

**Next Actions:**
- Threshold tuning ile precision-recall trade-off'unu optimize et (ör. recall > 0.75 hedefle)
- SMOTE veya cost-sensitive learning ile dengesiz veri problemini adresle
- A/B test ile XGBoost'u mevcut sisteme kademeli olarak deploy et
- Model monitoring pipeline kur: haftalık precision/recall takibi, drift detection
- Yüksek belirsizlik skorlu işlemler için LLM-based secondary review katmanı tasarla

---

### Model: Gemini (Google)

**Recommendation:**
XGBoost – ana model olarak production'a alınmalı; Logistic Regression hızlı pre-filter olarak korunabilir.

**Reasoning:**
- XGBoost'un 0.61 recall'u, Logistic Regression'ın 0.22'sine göre fraud yakalama oranını neredeyse 3 kat artırır. Verilen constraint'te false negative'ler doğrudan finansal kayıp olduğundan bu fark kritiktir.
- XGBoost'un 0.74 precision'ı kabul edilebilir düzeydedir. Manuel inceleme kapasitesi sınırlı olsa da, her 4 alarm'dan 3'ünün doğru olması operasyonel açıdan sürdürülebilirdir.
- Zero-shot LLM (precision: 0.66, recall: 0.58) her iki metrikte de XGBoost'un gerisindedir ve latency riski taşır. Etiketleme maliyetini ortadan kaldırma avantajı, bu senaryoda yeterli etiketli veri mevcut olduğundan geçerli değildir.
- Logistic Regression, düşük latency avantajıyla hızlı bir ön-eleme katmanı olarak pipeline'da tutulabilir (cascading architecture).

**Risks:**
- XGBoost single-model dependency riski taşır. Model fail ettiğinde fallback mekanizması gereklidir.
- 1.8M işlem üzerinde eğitilmiş model, yeni fraud pattern'lerine karşı kör olabilir (zero-day fraud).
- Precision-recall trade-off'u iş birimlerinin risk iştahına göre kalibre edilmezse operasyonel sürtüşme yaşanır.

**Next Actions:**
- Cascading pipeline: LR (hızlı pre-filter) → XGBoost (detaylı sınıflandırma) → Manuel review (yüksek belirsizlik)
- Recall iyileştirmesi için cost-sensitive XGBoost (scale_pos_weight parametresi) dene
- Real-time feature store entegrasyonu ile latency optimize et
- Aylık model retraining cycle'ı oluştur
- Fraud team ile işbirliği yaparak FN analiz raporu düzenle, kaçırılan fraud pattern'lerini belirle

---

### Model: Cohere Command

**Recommendation:**
XGBoost – production primary model; zero-shot LLM yalnızca etiketli veri olmayan yeni fraud tiplerinin keşfi (exploration) için kullanılmalı.

**Reasoning:**
- XGBoost verilen üç model arasında en yüksek F1 skoruna (0.669) sahiptir ve precision-recall dengesi iş kısıtlarına en uygun profildir.
- Zero-shot LLM'in 0.66 precision ve 0.58 recall değerleri, XGBoost'un altında kalmaktadır. Etiketli veri zaten mevcut olduğundan (1.8M işlem, %1.4 etiketli fraud), LLM'in "etiketleme maliyetini kaldırma" avantajı bu case'de marjinaldir.
- LLM inference latency'si genellikle 250 ms kısıtını aşar — real-time fraud detection için production-ready değildir.
- LLM'in asıl değeri, yeni ve bilinmeyen fraud pattern'lerini zero-shot olarak keşfetmektir. Bu, offline analiz ve model iyileştirme süreçlerinde kullanılabilir.

**Risks:**
- XGBoost'un non-linear karar sınırları explainability sorunları yaratabilir — regülasyon gerektiren sektörlerde bu risk önemlidir.
- Fraud rate'in %1.4 olması, model eğitiminde class imbalance'ın yeterince ele alınmadığına işaret edebilir. Mevcut recall (0.61) muhtemelen bu dengesizlikten etkilenmektedir.
- Modelin hangi feature'lara dayandığı bilinmeden production'a almak, adversarial attack'lara karşı savunmasızlık yaratır.

**Next Actions:**
- SHAP/LIME ile model explainability analizi yap
- Class imbalance için undersampling + XGBoost veya ensemble stratejisi uygula
- Zero-shot LLM'i haftalık batch job olarak çalıştır: XGBoost'un "not fraud" dediği ama LLM'in "suspicious" dediği işlemleri analiz et
- Shadow mode deployment: XGBoost'u production'da paralel çalıştır, 2 hafta metrik topla, ardından tam geçiş yap
- Alert threshold'u iş birimleriyle birlikte belirle (ör. precision ≥ 0.70, recall ≥ 0.65)

---

## Genel Değerlendirme Özeti

Her üç LLM de (GPT, Gemini, Cohere) aynı sonuca ulaşmaktadır: **XGBoost production için en uygun modeldir.** Ancak yaklaşım farklılıkları vardır:

| Karşılaştırma Boyutu | GPT | Gemini | Cohere |
|----------------------|-----|--------|--------|
| Ana öneri | XGBoost + LLM complementary | XGBoost + LR cascading | XGBoost + LLM exploration |
| Mimari önerisi | Two-tier (XGBoost + LLM review) | Three-tier cascade (LR → XGB → Manual) | Shadow mode + batch LLM |
| Risk odağı | Model drift, FN oranı | Single-model dependency | Explainability, adversarial risk |
| Güçlü yön | Pragmatik, doğrudan | Operasyonel detay, pipeline odaklı | Stratejik, uzun vadeli |

**En güvenilir karar destek aracı olarak:** Production kararı açısından GPT ve Gemini daha yapılandırılmış ve actionable çıktılar sunar. Gemini'nin cascading pipeline önerisi operasyonel olarak en olgun yaklaşımdır. Cohere ise stratejik perspektif (explainability, adversarial risk) açısından değerli tamamlayıcı içgörüler sağlar. İdeal yaklaşım, üç modelin çıktılarını birleştirerek kapsamlı bir production planı oluşturmaktır.
