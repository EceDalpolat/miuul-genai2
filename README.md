# GenAI Modül 2 - Vaka Çalışmaları (Case Studies)

Bu proje, Üretken Yapay Zeka (GenAI) destekli sistemlerin finans ve iş dünyası senaryolarında nasıl kullanılacağını anlatan eğitim ve deney kodlarını içerir. 

İki temel vaka çalışması üzerine kuruludur:
1. **Fraud Detection (Dolandırıcılık Tespiti):** Dengesiz veri setlerinde ve zaman kısıtlı sistemlerde LLM modellerinin nasıl davranması gerektiğini inceler.
2. **Sales Forecast (Satış Tahmini):** Mevcut bir makine öğrenmesi modelinin (LSTM) sonuçlarının, LLM tabanlı bir "Human-in-the-Loop" yorum katmanıyla nasıl entegre edileceğini (ve nelerin tehlikeli olduğunu) test eder.

---

## 📂 Dosya Yapısı ve İçerikler

### 🎓 Eğitim ve Teori Dosyaları (API İhtiyacı Yoktur)
Bu dosyalar konsepti ve mimariyi adım adım öğretmek üzere tasarlanmış simülasyonlardır. Herhangi bir API bağlantısı olmadan terminal üzerinden çalıştırarak mantığı kavrayabilirsiniz.

- `01_fraud_detection_egitim.py`: Fraud senaryosunda neden "Accuracy" yerine "Precision-Recall" bakmamız gerektiğini, iş maliyetlerini ve neden düşük temperature kullandığımızı anlatır.
- `02_sales_forecast_egitim.py`: Makine öğrenmesi modeli (LSTM) ile LLM dil modelinin iş bölümünü (Predictor vs. Interpreter) anlatır ve "Human-in-the-Loop" mimarisini simüle eder.
- `03_karsilastirma_ozet.py`: Her iki vaka çalışmasını tek tabloda özetler ve "Accuracy Tuzağı" hakkında genel bir kopya kağıdı/referans sağlar.

### 🧪 Yapay Zeka Deney Dosyaları (API Gerektirir)
Modellerin (OpenAI GPT, Google Gemini, Cohere) `0.0` ve `0.7` temperature değerlerinde iş kısıtlarına nasıl farklı ve riskli tepkiler verdiğini test eden canlı sistemlerdir.

- `fraud_temperature_experiment.py`: Dolandırıcılık tespiti için prompt tasarımını ve temperature'ın kesinlik etkisini ölçer.
- `sales_temperature_experiment.py`: Sayısal tahmin üretmesi YASAKLANMIŞ bir LLM'in, yüksek temperature ayarında bağlam dışına çıkıp çıkmayacağını analiz eder.
- `run_all_experiments.py`: **Ana Orkestratör!** Bütün modelleri ve testleri tek bir seferde koşturup yan yana kıyaslama raporu oluşturur.

---

## 🚀 Kurulum ve Çalıştırma

Kodları kendi ortamınızda çalıştırmadan önce, yapay zeka sağlayıcılarına erişim için API anahtarlarına ihtiyacınız vardır.

**1. Ortam Değişkenlerini Ayarlama (.env)**
Proje ana dizininde bulunan `.env.example` dosyasının adını `.env` olarak değiştirin ve kullanmak istediğiniz sağlayıcıların API anahtarlarını içine yapıştırın:
```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
COHERE_API_KEY=...
```
*(Sadece kullanmak istediklerinizi doldurmanız yeterlidir, sistem boş olanları atlar.)*

### Yöntem A: Docker & Streamlit ile Çalıştırma (Önerilen) 🐳
Ortam veya kütüphane çakışmaları yaşamamak ve harika bir **web arayüzü** üzerinden test yapmak için projeyi tek bir komutla Docker üzerinden ayağa kaldırabilirsiniz. Docker yüklüyse:

```bash
docker-compose up --build
```
Bu komut çalıştıktan sonra tarayıcınızı açın ve **http://localhost:8501** adresine gidin.
Karşınıza çıkan Streamlit arayüzünden modelleri, senaryoları ve Temperature ayarlarını sürükleyip bırakarak seçebilir ve sonuçları yan yana tablolar halinde görebilirsiniz. Üstelik `.env` dosyanızda API anahtarınız yoksa bile doğrudan ekrandan girebilirsiniz!

### Yöntem B: Python ile Çalıştırma (Lokal) 🐍
Docker kullanmak istemiyorsanız yerel ortamınızda da çalıştırabilirsiniz.

**1. Kütüphaneleri Yükleyin:**
```bash
pip install -r requirements.txt
```

**2. Web Arayüzünü Başlatın:**
```bash
streamlit run streamlit_app.py
```
*(Dilerseniz sadece tek bir deneyi de çalıştırabilirsiniz: `python fraud_temperature_experiment.py`)*

---

💡 **Not:** Bu projede incelenen temel konseptlerin detayı PDF dokümanlarında ve `.md` uzantılı çözüm yönergelerinde yer almaktadır. Kodlar, bu yönergelerin doğrudan teknik uygulamalarıdır.
