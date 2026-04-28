---
title: "Temel LLM ve Multimodal Üretim"
author: "Eğitim: LLM & Multimodal"
duration: "90 dk (öneri)"
---

# Temel LLM ve Multimodal Üretim

## Alt başlık

- Metin, Ses, Görsel ve Kod üretimine giriş

---

## Öğrenme Hedefleri

- LLM nedir ve temel bileşenleri açıklar
- Metin üretim stratejilerini uygular (temperature, top-k/p)
- Temel TTS ve görsel diffusion akışlarını açıklar
- LLM ile kod üretimi ve güvenli kullanım pratiklerini gösterir

---

## Zamanlama (90 dk önerisi)

- 0–10 dk: Giriş ve hedefler
- 10–30 dk: LLM temelleri (Transformer, tokenizasyon)
- 30–50 dk: Metin üretimi + kısa demo
- 50–65 dk: Ses üretimi (TTS) + demo
- 65–80 dk: Görsel üretimi (Diffusion) + demo
- 80–90 dk: Kod üretimi, kapanış ve alıştırmalar

---

## Neden LLM?

- Uygulama alanları: asistanlar, içerik üretimi, kod otomasyonu
- Güçlü ama dikkat edilmesi gereken sınırlamalar: hallucination, latency

Notes: Kısa örnek çıktı göster (metin, ses, görsel, kod) önceden renderlenmişse aç.

---

## LLM Temelleri

- Transformer mimarisi: self-attention, multi-head
- Tokenization: BPE / WordPiece / Unigram
- Öğrenme evreleri: pretraining, fine-tuning, instruction tuning, RLHF

Notes: Tek token akışını beyaz tahta/şema ile gösterin.

---

## Metin Üretimi Teknikleri

- Deterministik: Greedy, Beam Search
- Stokastik: Sampling, Temperature, Top-k, Top-p
- Prompt mühendisliği: few-shot, chain-of-thought

Notes: Temperature=0.0 vs 0.8 canlı karşılaştırma planlayın.

---

## Kod Örneği — Metin Tamamlama (örnek)

```python
from openai import OpenAI
client = OpenAI(api_key="YOUR_KEY")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Bana kısa bir Türkçe özet yaz."}],
    temperature=0.2
)
print(resp.choices[0].message["content"])
```

Notes: Katılımcıların kendi API anahtarını .env'de tutması gerektiğini vurgulayın.

---

## Ses Üretimi (TTS)

- Pipeline: Text -> Linguistic features -> Mel-spectrogram -> Vocoder -> Waveform
- Modeller: Tacotron2 + HiFi-GAN / Tacotron2 + WaveNet / Coqui TTS

Quick demo (Coqui):

```bash
pip install TTS
python -m TTS.bin.synthesize --model_name tts_models/en/ljspeech/tacotron2-DDC --text "Merhaba, bu bir demo." --out_path demo.wav
```

Notes: Aynı metni farklı TTS modellerinde dinlettirin; prosodi farklarını tartışın.

---

## Görsel Üretimi

- Modeller: DALL·E, Diffusion (Stable Diffusion), Latent Diffusion
- Prompt mühendisliği (stil, kompozisyon, negatif prompt)

Hızlı örnek (diffusers):

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
img = pipe("Sakin bir göl kenarında gün batımı, yağlı boya", num_inference_steps=30).images[0]
img.save("sunset.png")
```

Notes: Aynı prompt'u 3 farklı stil parametresiyle üretin ve karşılaştırın.

---

## Multimodal Bileşenler

- CLIP/ALIGN ile görsel-metinden eşleştirme
- Multimodal LLM örnekleri: görsel açıklama, multimodal sorgu

Notes: Basit bir görsel caption demo'su gösterin.

---

## Kod Üretimi ile İş Akışı

- Pattern: LLM -> unit test oluştur -> çalıştır -> düzeltme döngüsü
- Tool-use: LLM'in dış araçları çağırması (ör. kod çalıştırma, test komutları)

Notes: Küçük bir fonksiyon isteyin, pytest ile test yazdırıp çalıştırın.

---

## Değerlendirme ve Güvenlik

- Metrikler: BLEU/ROUGE/F1 (metin), FID/IS (görsel), MOS (ses)
- Güvenlik: hallucination, zararlı içerik, lisans/attribution

Notes: Üretim sistemi önerileri: caching, rate-limiting, drift monitoring.

---

## Uygulamalı Alıştırmalar

- Metin: temperature deneyleri ve prompt-tuning
- Ses: kısa hikâyeyi TTS ile okutma
- Görsel: inpainting ile logo varyasyonları
- Kod: küçük CLI aracı üretip test etme

---

## Kaynaklar

- Vaswani et al., "Attention is All You Need"
- Tacotron2, WaveNet, HiFi-GAN
- Stable Diffusion & Hugging Face diffusers docs
- OpenAI & safety best practices

---

## Sonraki Adımlar

- Bu Markdown slaytı PowerPoint'e çevirebilir veya Reveal.js ile sunabilirsiniz.
- İstersen: 1) PPTX oluşturayım 2) `examples/` içinde kodları çalıştırılabilir hâle getireyim

