# Resmi Python imajını kullanıyoruz (hafif versiyon)
FROM python:3.10-slim

# Çalışma dizinini belirliyoruz
WORKDIR /app

# Önce requirements.txt'yi kopyalayıp kütüphaneleri kuruyoruz 
# (Bu adım Docker önbelleğini efektif kullanmak içindir)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projenin geri kalan dosyalarını kopyalıyoruz
COPY . .

# Streamlit portunu dışarı açıyoruz
EXPOSE 8501

# Varsayılan komut olarak Streamlit arayüzünü çalıştırıyoruz
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
