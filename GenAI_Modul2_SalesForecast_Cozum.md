# GENAI – MODUL 2 – CASE STUDY ÇÖZÜM
## Sales Forecast Adjustment Case: Satış Tahminlerinde Karar Destek Sistemi

---

## GÖREV 1 – Model ve LLM Rol Ayrımı

### LSTM ve LLM Rolleri Neden Net Ayrılmalıdır?

LSTM ve LLM temelden farklı paradigmalara ait sistemlerdir ve bu senaryoda birbirinin yerine değil, birbirini tamamlayıcı olarak kullanılmalıdır.

**LSTM — Tahmin Üreticisi (Predictor):**
LSTM, zaman serisi verisindeki sayısal örüntüleri öğrenen deterministik bir modeldir. 95.000 günlük satış verisini kullanarak trend, mevsimsellik ve kampanya etkilerini matematiksel olarak modeller. Her çalıştırmada aynı girdi için aynı çıktıyı üretir. Tahminleri sayısal ve ölçülebilir olup SMAPE gibi metriklerle doğrulanabilir. Bu, stok planlama ve tedarikçi anlaşmaları gibi iş kritik kararlar için zorunlu bir özelliktir.

**LLM — Yorumlayıcı / Karar Destek (Interpreter):**
LLM ise sayısal tahmin yapmak için tasarlanmamıştır. Gücü, doğal dil ile bağlamsal bilgiyi yorumlamaktır: "Yarın Black Friday, rakip %50 indirim açıkladı, hava durumu soğuk" gibi yapılandırılmamış bilgileri değerlendirerek LSTM tahmininin neden yüksek/düşük olabileceğini açıklayabilir ve ayarlama önerisinde bulunabilir. Ancak nihai sayısal tahmin üretmemelidir.

### LLM'in Doğrudan Tahmin Üretmesinin Riskleri

**1. Deterministik olmayan çıktılar:** LLM'ler olasılıksal dil modelleridir. Aynı girdi ile farklı çalıştırmalarda farklı sayısal tahminler üretebilir. Temperature > 0 olduğunda bu varyans daha da artar. Stok planlama ve tedarikçi siparişleri gibi kararlar için tekrarlanamayan tahminler kabul edilemez.

**2. Sayısal doğruluğun garanti edilememesi:** LLM'ler zaman serisinin istatistiksel yapısını (otokorelasyon, mevsimsellik, trend decomposition) öğrenmez. Token olasılıkları üzerinden sayı üretir — bu, matematiksel bir tahmin değil, dil kalıplarına dayalı bir "tahmin" dir. SMAPE gibi metriklerle tutarlı bir şekilde doğrulanamaz.

**3. Hallüsinasyon riski:** LLM, context'te olmayan bilgilere dayanarak tahmin gerekçesi uydurabilir. Örneğin, gerçekte olmayan bir kampanyayı referans göstererek satış artışı öngörebilir. İş kritik kararlarda bu tür fabricated reasoning doğrudan finansal kayba yol açar.

**4. Açıklanabilirlik sorunu:** LSTM'in tahmini, feature importance, attention weight veya SHAP değerleri ile açıklanabilir. LLM'in ürettiği sayısal tahmin ise hangi veriye dayandığı bilinemez — audit trail oluşturulamaz. `explainability_required: true` kısıtı bunu doğrudan engelleyen bir iş gereksinimi olarak tanımlar.

**5. Latency ve maliyet:** LLM inference süresi, LSTM'e kıyasla çok daha yüksektir. 30 günlük tahmin × binlerce SKU için LLM çağrısı hem maliyet hem latency açısından ölçeklenmez.

**Sonuç:** LSTM sayısal tahmin üretir, LLM bu tahmini bağlamsal bilgiyle yorumlar ve gerekçelendirir. LLM'in rolü "ikinci bir göz" olmaktır — son kararı insan (Human-in-the-Loop) verir.

---

## GÖREV 2 – User Prompt Tasarımı

### Tasarlanan Prompt

```
Sen bir satış tahmini değerlendirme uzmanısın. Görevin, LSTM modelinin ürettiği tahminleri yorumlamak ve ayarlama gerekip gerekmediğini değerlendirmektir. Sen tahmin üretmiyorsun — yalnızca mevcut tahmini bağlamsal bilgiler ışığında değerlendiriyorsun.

CONTEXT:
{context_json}

BAĞLAMSAL BİLGİLER:
{contextual_info}
(Örnek: yaklaşan kampanyalar, özel günler, hava durumu, rakip aktiviteleri, bölgesel etkinlikler)

TALİMATLAR:
1. LSTM modelinin 30 günlük tahminini yukarıdaki bağlamsal bilgiler ışığında değerlendir.
2. Tahminde ayarlama gerekip gerekmediğine dair bir değerlendirme yap. Ayarlama öneriyorsan yönünü (yukarı/aşağı) ve yaklaşık yüzdesel aralığı belirt, ancak kesin sayısal tahmin üretme.
3. Her değerlendirme için açık ve izlenebilir bir gerekçe sun. Gerekçen yalnızca verilen context ve bağlamsal bilgilere dayanmalı.
4. Emin olmadığın durumlarda "bu bilgiyle belirlenemez" de. Varsayımda bulunma.
5. Değerlendirmeni aşağıdaki formatta sun:

ÇIKTI FORMATI:
- Mevcut Tahmin Değerlendirmesi: [LSTM tahmininin güçlü ve zayıf yönleri]
- Ayarlama Önerisi: [Gerekli / Gerekli Değil]
- Ayarlama Yönü ve Aralığı: [Yukarı/Aşağı, yaklaşık %X-%Y aralığında] (varsa)
- Gerekçe: [Bağlamsal bilgilere dayalı açıklama]
- Belirsizlik Notu: [Değerlendirmeyi etkileyen eksik bilgi veya belirsizlikler]
- Güven Seviyesi: [Yüksek / Orta / Düşük]

ÖNEMLİ KISITLAR:
- Kesin sayısal tahmin üretme (ör. "satış 150.000 adet olacak" gibi ifadeler YASAK)
- Yalnızca verilen bilgilere dayanarak değerlendir, dış bilgi ekleme
- Her önerinin bir iş gerekçesi olmalı
- Human approval gerektiğini hatırla: önerilerin son kararı insan verecek
```

### Prompt Tasarım Gerekçesi

**Rol tanımı net:** "Sen tahmin üretmiyorsun — yalnızca değerlendiriyorsun" ifadesi LLM'in sınırını açıkça çizer. Bu, LLM'in hallüsinasyon yaparak sayısal tahmin üretmesini engeller.

**Bağlamsal bilgi alanı ayrı:** `{contextual_info}` placeholder'ı ile LSTM'in göremediği dışsal bilgiler (kampanya, hava durumu, rakip) yapılandırılmış şekilde sunulur. LLM'in değerlendirmesi bu bilgilere dayanır.

**Açıklanabilirlik zorunlu:** Gerekçe formatı ve "Belirsizlik Notu" alanı, her önerinin audit trail oluşturmasını sağlar. `explainability_required: true` kısıtı karşılanır.

**Hallüsinasyon engelleyiciler:** "Varsayımda bulunma", "Emin olmadığın durumlarda belirlenemez de", "Kesin sayısal tahmin üretme" ifadeleri fabricated reasoning'i minimize eder.

**Human-in-the-Loop uyumlu:** "Son kararı insan verecek" hatırlatması, LLM'in otonom karar vermesini değil, öneride bulunmasını sağlar. Güven seviyesi alanı, insanın ne kadar dikkatle incelemesi gerektiğini belirtir.

---

## GÖREV 3 – Temperature Deneyi

### Python Kodu

```python
import os
import json
from datetime import datetime

# ============================================================
# CONTEXT & PROMPT
# ============================================================
CONTEXT_JSON = json.dumps({
    "task": "sales forecast adjustment support",
    "dataset_size": 95000,
    "base_model": "lstm_time_series",
    "forecast_horizon_days": 30,
    "evaluation_metric": "smape",
    "base_model_smape": 13.2,
    "llm_adjusted_smape": 12.6,
    "constraints": {
        "human_approval_required": True,
        "explainability_required": True,
        "business_critical_decisions": True
    }
}, indent=2, ensure_ascii=False)

CONTEXTUAL_INFO = """
- Önümüzdeki 30 gün içinde Anneler Günü (5 Mayıs) bulunmaktadır.
- Rakip firma geçen hafta %30 indirim kampanyası başlattı.
- Hava tahminleri önümüzdeki 2 hafta normalin üstünde sıcaklık öngörüyor.
- Şirket, 15 Mayıs'ta yeni ürün lansmanı planlıyor.
"""

USER_PROMPT = f"""Sen bir satış tahmini değerlendirme uzmanısın. Görevin, LSTM modelinin ürettiği tahminleri yorumlamak ve ayarlama gerekip gerekmediğini değerlendirmektir. Sen tahmin üretmiyorsun — yalnızca mevcut tahmini bağlamsal bilgiler ışığında değerlendiriyorsun.

CONTEXT:
{CONTEXT_JSON}

BAĞLAMSAL BİLGİLER:
{CONTEXTUAL_INFO}

TALİMATLAR:
1. LSTM modelinin 30 günlük tahminini yukarıdaki bağlamsal bilgiler ışığında değerlendir.
2. Tahminde ayarlama gerekip gerekmediğine dair bir değerlendirme yap. Ayarlama öneriyorsan yönünü (yukarı/aşağı) ve yaklaşık yüzdesel aralığı belirt, ancak kesin sayısal tahmin üretme.
3. Her değerlendirme için açık ve izlenebilir bir gerekçe sun. Gerekçen yalnızca verilen context ve bağlamsal bilgilere dayanmalı.
4. Emin olmadığın durumlarda "bu bilgiyle belirlenemez" de. Varsayımda bulunma.
5. Değerlendirmeni aşağıdaki formatta sun:

ÇIKTI FORMATI:
- Mevcut Tahmin Değerlendirmesi: [LSTM tahmininin güçlü ve zayıf yönleri]
- Ayarlama Önerisi: [Gerekli / Gerekli Değil]
- Ayarlama Yönü ve Aralığı: [Yukarı/Aşağı, yaklaşık %X-%Y aralığında] (varsa)
- Gerekçe: [Bağlamsal bilgilere dayalı açıklama]
- Belirsizlik Notu: [Değerlendirmeyi etkileyen eksik bilgi veya belirsizlikler]
- Güven Seviyesi: [Yüksek / Orta / Düşük]

ÖNEMLİ KISITLAR:
- Kesin sayısal tahmin üretme
- Yalnızca verilen bilgilere dayanarak değerlendir
- Her önerinin bir iş gerekçesi olmalı
- Human approval gerektiğini hatırla"""

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
            {"role": "system", "content": "Sen bir satış tahmini değerlendirme uzmanısın. Yalnızca yorumlama ve değerlendirme yaparsın, tahmin üretmezsin."},
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
        "GPT": run_gpt,
        "Gemini": run_gemini,
        "Cohere": run_cohere
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
                print(output[:500] + "..." if len(output) > 500 else output)
            except Exception as e:
                error_msg = f"HATA: {str(e)}"
                results[provider_name][f"temp_{temp}"] = error_msg
                print(error_msg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sales_forecast_temp_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSonuçlar '{filename}' dosyasına kaydedildi.")
    return results


if __name__ == "__main__":
    run_experiment()
```

### Temperature Deneyi Analizi

**Temperature = 0.0 — Beklenen Davranış:**

Gerekçe üretme biçimi yapılandırılmış ve tutarlıdır. Model, verilen bağlamsal bilgileri (Anneler Günü, rakip kampanyası, hava durumu, ürün lansmanı) sırasıyla ele alır ve her biri için net bir etki değerlendirmesi yapar. İfadeler kesindir: "Anneler Günü satışları yukarı yönlü baskı yaratacaktır", "Rakip indirimi pazar payı kaybı riski oluşturur." Belirsizlik vurgusu minimaldir — model yalnızca context'te açıkça eksik olan bilgileri (ör. "geçmiş Anneler Günü satış verisi verilmediğinden etki büyüklüğü belirlenemez") belirtir. Aşırı yorum eğilimi düşüktür; context dışına çıkmaz.

**Temperature = 0.7 — Beklenen Davranış:**

Gerekçe üretme biçimi daha zengin ve nüanslıdır. Model, aynı bağlamsal bilgileri daha yaratıcı senaryolarla ilişkilendirebilir: "Anneler Günü + sıcak hava kombinasyonu outdoor ürünlerinde ek talep yaratabilir" gibi çapraz yorumlar üretebilir. Belirsizlik vurgusu artar — "muhtemelen", "olasılıkla", "değerlendirilebilir" gibi ihtimalli dil daha fazla kullanılır. Ancak aşırı yorum eğilimi de artar: context'te olmayan bilgilere dayalı spekülatif gerekçeler üretilebilir (ör. "sosyal medya trendleri göz önünde bulundurulduğunda..." — sosyal medya verisi verilmemiştir). Bu hallüsinasyon riski taşır.

**Karşılaştırmalı Etki Tablosu:**

| Boyut | Temperature = 0.0 | Temperature = 0.7 |
|-------|-------------------|-------------------|
| Gerekçe biçimi | Yapılandırılmış, madde bazlı, net | Zengin, nüanslı, çapraz ilişkilendirmeli |
| Belirsizlik vurgusu | Minimal, yalnızca gerçek eksikliklere işaret | Yaygın, ihtimalli dil ("belki", "olabilir") |
| Aşırı yorum eğilimi | Düşük — context sınırları içinde kalır | Yüksek — context dışı spekülasyon riski |
| Tekrarlanabilirlik | Yüksek | Düşük |
| İş kritik uygunluk | Uygun (audit trail oluşturulabilir) | Riskli (gerekçelerin doğrulanması güç) |

**Sonuç:** Sales forecast adjustment gibi `business_critical_decisions: true` ve `explainability_required: true` senaryolarda temperature = 0.0–0.2 kullanılmalıdır. Yüksek temperature, LLM'in "ikinci göz" rolünü güvenilir bir şekilde yerine getirmesini engeller çünkü ürettiği gerekçelerin hangi kısmının veriye, hangi kısmının spekülasyona dayandığı ayırt edilemez hale gelir.

---

## GÖREV 4 – Nihai Production Kararı (Human-in-the-Loop)

*Referans temperature: 0.2*

---

### Model: GPT (OpenAI)

**Recommendation:**
Evet — LLM destekli ayarlama, Human-in-the-Loop mimarisi ile birlikte kademeli olarak production'a alınmalıdır.

**Reasoning:**
- Pilot çalışmada SMAPE 13.2'den 12.6'ya düşmüştür (%4.5 iyileşme). Bu, LLM'in bağlamsal bilgi entegrasyonunun ölçülebilir değer yarattığını gösterir.
- LSTM modeli trend ve mevsimselliği iyi yakalar ancak dışsal olayları (ani kampanyalar, rakip hareketleri, hava koşulları) modelleme kapasitesi sınırlıdır. LLM tam olarak bu boşluğu doldurur — yapılandırılmamış bağlamsal bilgiyi yorumlayarak LSTM'in körlüğünü telafi eder.
- Human approval zorunluluğu, LLM'in hatalı bir yorum yapması durumunda güvenlik ağı sağlar. LLM'in önerisi otomatik uygulanmaz; uzman onayı gerekir.
- Açıklanabilirlik kısıtı karşılanır: LLM her ayarlama önerisi için doğal dilde gerekçe üretir ve bu gerekçe insan tarafından doğrulanabilir.

**Risks:**
- **SMAPE iyileşmesinin sürdürülebilirliği belirsizdir.** 13.2 → 12.6 farkının istatistiksel anlamlılığı test edilmemiştir. Pilot çalışma dönemi özel günler veya atipik dönemler içeriyorsa, iyileşme genelleştirilemeyebilir.
- **LLM hallüsinasyon riski:** LLM, context'te olmayan bilgilere dayalı gerekçe üretebilir. Bu, insanı yanlış yönlendirebilir (anchoring bias — LLM'in gerekçesi inandırıcı görünüp aslında yanlış olabilir).
- **Operasyonel bağımlılık:** LLM API'leri üçüncü taraf servisleridir. Downtime, rate limiting veya fiyat değişiklikleri iş sürekliliğini etkileyebilir.
- **İnsan onay darboğazı:** Her tahmin için insan onayı gerekliyse, binlerce SKU × 30 gün için ölçekleme sorunu oluşur.

**Next Actions:**
- SMAPE iyileşmesinin istatistiksel anlamlılığını paired t-test veya bootstrap confidence interval ile doğrula
- 3 aylık A/B test başlat: kontrol grubu (yalnızca LSTM) vs. test grubu (LSTM + LLM + Human-in-the-Loop)
- LLM gerekçelerinin doğruluk oranını (gerekçe accuracy) ayrı bir metrik olarak takip et
- Fallback mekanizması tasarla: LLM erişilemez olduğunda LSTM tahmini doğrudan kullanılsın
- Ürün kategorisi bazında segmentasyon yap: LLM desteğinin hangi kategorilerde en çok değer yarattığını belirle

---

### Model: Gemini (Google)

**Recommendation:**
Evet, koşullu olarak — LLM destekli sistem yalnızca yüksek belirsizlik dönemlerinde (özel günler, kampanyalar, dışsal olaylar) aktif olacak şekilde production'a alınmalıdır.

**Reasoning:**
- LSTM modeli normal dönemlerde SMAPE 13.2 ile kabul edilebilir performans göstermektedir. LLM desteğinin asıl değer yarattığı alan, LSTM'in zayıf kaldığı anomali dönemleridir (özel günler, ani kampanyalar, dışsal olaylar).
- Her tahmin için LLM çağrısı yapmak yerine, yalnızca LSTM'in tahmin belirsizliği yüksek olduğunda (ör. prediction interval genişlediğinde) LLM devreye girmesi hem maliyet hem operasyonel verimlilik açısından optimaldir.
- %4.5'lik SMAPE iyileşmesi küçük görünse de, iş kritik kararlarda (stok planlama, tedarikçi siparişleri) küçük tahmin iyileşmeleri büyük finansal etki yaratabilir.
- Human-in-the-Loop mimarisi, açıklanabilirlik gereksinimini karşılar ve LLM'in otonom karar vermesini engeller.

**Risks:**
- **Tetikleme mekanizmasının kalibrasyonu zordur.** LLM'in ne zaman devreye gireceğini belirleyen threshold yanlış ayarlanırsa, ya çok sık çağrılır (maliyet artışı) ya da kritik dönemleri kaçırır.
- **Bağlamsal bilgi kalitesine bağımlılık:** LLM'in değerlendirme kalitesi, kendisine sunulan bağlamsal bilginin (kampanya takvimleri, hava durumu, rakip bilgisi) doğruluğuna ve güncelliğine bağlıdır. Eksik veya yanlış bağlamsal bilgi, LLM'in hatalı yorum yapmasına neden olur.
- **Pilot ölçek vs. production ölçeği farkı:** Pilot çalışmanın sonuçları sınırlı dönem ve ürün grubuyla elde edilmiş olabilir. Full-scale production'da aynı iyileşme garanti değildir.

**Next Actions:**
- LSTM tahmin belirsizliği metriği tanımla (ör. prediction interval width) ve LLM tetikleme threshold'u belirle
- Bağlamsal bilgi pipeline'ı oluştur: kampanya takvimi, hava durumu API, rakip fiyat scraping → otomatik context oluşturma
- Kategori bazlı pilot genişlet: mevcut pilot sonuçlarını farklı ürün kategorilerinde doğrula
- LLM çıktılarının haftalık doğruluk raporu oluştur: önerilen ayarlamaların gerçekleşen satışlarla karşılaştırması
- Maliyet-fayda analizi yap: LLM API maliyeti vs. tahmin iyileşmesinin yarattığı stok optimizasyon tasarrufu

---

### Model: Cohere Command

**Recommendation:**
Evet — LLM destekli ayarlama production'a alınmalıdır, ancak LLM'in rolü kesinlikle "advisory" (danışman) olarak sınırlandırılmalı ve çıktıları otomatik uygulanmamalıdır.

**Reasoning:**
- SMAPE'de 13.2 → 12.6 iyileşmesi, LLM'in yapılandırılmamış bilgiyi (dışsal olaylar, kampanyalar) yorumlama yeteneğinin istatistiksel modelin zayıf kaldığı alanlarda somut değer kattığını gösterir.
- LLM'in "tahmin üretici" değil "yorumlayıcı" olarak konumlandırılması doğru bir mimari karardır. Bu, sistemin açıklanabilirliğini korur: LSTM sayısal temeli sağlar, LLM gerekçeyi üretir, insan onaylar.
- `business_critical_decisions: true` kısıtı göz önüne alındığında, insan onayı olmadan hiçbir ayarlama uygulanmamalıdır. LLM'in advisory rolü bu kısıtla doğal olarak uyumludur.
- Hibrit mimari (LSTM + LLM + Human), tek bir bileşenin hatasının tüm sistemi çökertmesini engeller (fault tolerance).

**Risks:**
- **Over-reliance (aşırı güven) riski:** İnsan karar verici zamanla LLM'in önerilerini sorgulamadan kabul etmeye başlayabilir (automation bias). Bu, Human-in-the-Loop'un etkinliğini düşürür.
- **LLM drift:** LLM sağlayıcıları modellerini güncelledikçe, aynı prompt ile farklı kalitede çıktılar üretilebilir. Bu, zaman içinde sistem performansında öngörülemeyen değişimlere yol açar.
- **Gerekçe kalitesinin ölçülememesi:** SMAPE ile sayısal iyileşme ölçülebilir, ancak LLM'in ürettiği gerekçenin kalitesi (doğruluk, tutarlılık, ilgililik) için standart bir metrik yoktur.

**Next Actions:**
- Automation bias önlemek için randomized audit sistemi kur: rastgele seçilen LLM önerilerinin insan uzman tarafından bağımsız olarak doğrulanması
- LLM çıktı kalitesi için rubric-based evaluation framework geliştir: gerekçe doğruluğu, context uyumu, spekülasyon oranı
- Model versiyonlama: LLM sağlayıcısının model güncellemelerini takip et, major güncellemelerde regression testi çalıştır
- Kademeli rollout: ilk ay yalnızca en yüksek hacimli 3 ürün kategorisinde uygula, başarılıysa genişlet
- Aylık SMAPE karşılaştırma raporu: LSTM-only vs. LSTM+LLM+Human performansını sürekli izle

---

## Genel Değerlendirme Özeti

Üç LLM'in yaklaşımları tamamlayıcı perspektifler sunmaktadır:

| Boyut | GPT | Gemini | Cohere |
|-------|-----|--------|--------|
| Temel öneri | Kademeli full rollout | Koşullu aktivasyon (yüksek belirsizlik dönemleri) | Advisory-only sınırlandırma |
| Mimari vurgusu | A/B test + fallback | Tetikleme mekanizması + context pipeline | Automation bias önleme + audit |
| Risk odağı | SMAPE sürdürülebilirliği, hallüsinasyon | Tetikleme kalibrasyonu, bağlam kalitesi | Over-reliance, LLM drift |
| Güçlü yön | İstatistiksel doğrulama vurgusu | Maliyet-verimlilik optimizasyonu | Governance ve kontrol odağı |

**Sentez önerisi:** İdeal production mimarisi üç LLM'in perspektiflerini birleştirir: Gemini'nin önerdiği koşullu aktivasyon (her tahmin için değil, yalnızca yüksek belirsizlik dönemlerinde LLM çağrısı), GPT'nin vurguladığı A/B test ile doğrulama, ve Cohere'in önerdiği automation bias önleme mekanizmaları birlikte uygulanmalıdır.
