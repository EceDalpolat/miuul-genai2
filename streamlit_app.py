import streamlit as st
import os
import json
from datetime import datetime

# Import core functions from run_all_experiments
from run_all_experiments import (
    build_fraud_prompt,
    build_sales_prompt,
    get_active_providers,
    run_case,
    PROVIDER_CONFIG
)

# --- UI Yapılandırması ---
st.set_page_config(
    page_title="GenAI Modül 2 - AI Deneyleri",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 GenAI Modül 2 - Vaka Çalışması Deneyleri")
st.markdown("Bu arayüz, Fraud Detection ve Sales Forecast senaryoları için LLM modellerini farklı Temperature ayarlarında test etmenizi sağlar.")

# --- SIDEBAR (Yan Menü) ---
st.sidebar.header("⚙️ Ayarlar")

# 1. API Anahtarları (Eğer .env'de yoksa kullanıcıdan al)
with st.sidebar.expander("🔑 API Anahtarları", expanded=False):
    st.info("Eğer .env dosyasında tanımlıysa buraya girmenize gerek yoktur.")
    
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
        
    google_key = st.text_input("Google Gemini API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key
        
    cohere_key = st.text_input("Cohere API Key", value=os.getenv("COHERE_API_KEY", ""), type="password")
    if cohere_key: os.environ["COHERE_API_KEY"] = cohere_key

# Aktif sağlayıcıları kontrol et
active_providers = get_active_providers()

if not active_providers:
    st.error("⚠️ Hiçbir API anahtarı bulunamadı! Lütfen sol menüden API anahtarlarınızı girin.")
    st.stop()

# 2. Sağlayıcı Seçimi
st.sidebar.subheader("🤖 Modeller")
available_provider_names = list(active_providers.keys())
selected_providers = st.sidebar.multiselect(
    "Test edilecek modelleri seçin:",
    options=available_provider_names,
    default=available_provider_names
)

# 3. Vaka Çalışması Seçimi
st.sidebar.subheader("📂 Vaka Çalışması")
case_study = st.sidebar.radio(
    "Hangi senaryoyu test etmek istersiniz?",
    options=["Her İkisi (Fraud & Sales)", "Sadece Fraud Detection", "Sadece Sales Forecast"]
)

# 4. Temperature Seçimi
st.sidebar.subheader("🌡️ Temperature Ayarları")
st.sidebar.markdown("*0.0 deterministik, 0.7 yaratıcı cevaplar üretir.*")
temp_options = [0.0, 0.2, 0.5, 0.7, 1.0]
selected_temps = st.sidebar.multiselect(
    "Temperature Değerleri:",
    options=temp_options,
    default=[0.0, 0.7]
)

if not selected_temps:
    st.sidebar.warning("En az bir Temperature değeri seçmelisiniz.")
    
# --- ANA EKRAN ---
if st.button("🚀 Deneyi Başlat", type="primary"):
    if not selected_temps:
        st.error("Lütfen en az bir Temperature değeri seçin.")
        st.stop()
        
    if not selected_providers:
        st.error("Lütfen en az bir Model Sağlayıcı seçin.")
        st.stop()
        
    # Filtrelenmiş provider'ları hazırla
    filtered_providers = {k: v for k, v in active_providers.items() if k in selected_providers}
    
    all_results = {}
    
    # Progress bar
    progress_text = "Deneyler çalıştırılıyor. Lütfen bekleyin..."
    my_bar = st.progress(0, text=progress_text)
    
    # --- FRAUD CASE ---
    if case_study in ["Her İkisi (Fraud & Sales)", "Sadece Fraud Detection"]:
        st.subheader("🕵️ Fraud Detection Senaryosu Sonuçları")
        with st.spinner("Fraud case çalıştırılıyor..."):
            res_fraud = run_case(
                case_name="fraud",
                prompt=build_fraud_prompt(),
                system_key="system_fraud",
                temperatures=selected_temps,
                providers=filtered_providers
            )
            all_results["fraud"] = res_fraud
            
            # Sonuçları görselleştir
            for provider_name, provider_data in res_fraud.items():
                st.markdown(f"### 🔵 {provider_name.upper()}")
                cols = st.columns(len(selected_temps))
                
                for idx, temp in enumerate(selected_temps):
                    temp_key = f"temp_{temp}"
                    data = provider_data.get(temp_key, {})
                    
                    with cols[idx]:
                        st.markdown(f"**Temp = {temp}** ⏱️ `{data.get('elapsed_sec', '-')}s`")
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.info(data.get("output", ""))
                            st.caption(f"{data.get('char_count', 0)} karakter")
                            
        st.divider()
    my_bar.progress(50, text="Sales case'e geçiliyor...")

    # --- SALES CASE ---
    if case_study in ["Her İkisi (Fraud & Sales)", "Sadece Sales Forecast"]:
        st.subheader("📈 Sales Forecast Senaryosu Sonuçları")
        with st.spinner("Sales case çalıştırılıyor..."):
            res_sales = run_case(
                case_name="sales",
                prompt=build_sales_prompt(),
                system_key="system_sales",
                temperatures=selected_temps,
                providers=filtered_providers
            )
            all_results["sales"] = res_sales
            
            # Sonuçları görselleştir
            for provider_name, provider_data in res_sales.items():
                st.markdown(f"### 🟢 {provider_name.upper()}")
                cols = st.columns(len(selected_temps))
                
                for idx, temp in enumerate(selected_temps):
                    temp_key = f"temp_{temp}"
                    data = provider_data.get(temp_key, {})
                    
                    with cols[idx]:
                        st.markdown(f"**Temp = {temp}** ⏱️ `{data.get('elapsed_sec', '-')}s`")
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(data.get("output", ""))
                            st.caption(f"{data.get('char_count', 0)} karakter")
                            
    my_bar.progress(100, text="Tamamlandı!")
    
    # --- JSON İNDİRME ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_str = json.dumps(all_results, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📥 Sonuçları JSON Olarak İndir",
        data=json_str,
        file_name=f"ui_results_{timestamp}.json",
        mime="application/json"
    )
    
    st.balloons()
