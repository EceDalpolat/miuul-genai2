
import os
import google.generativeai as genai
import cohere
from dotenv import load_dotenv

load_dotenv()

print("--- GOOGLE GEMINI MODELS ---")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing Gemini models: {e}")

print("\n--- COHERE MODELS ---")
try:
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # In SDK v2, we can use list_models or similar
    # Let's try listing models
    models = co.models.list()
    for m in models.models:
        print(f"- {m.name}")
except Exception as e:
    print(f"Error listing Cohere models: {e}")
