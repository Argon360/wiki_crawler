import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print(f"Checking available models for key ending in ...{api_key[-4:]}")

try:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("\nSUCCESS! Your key is working. Available models:")
        for model in data.get('models', []):
            name = model.get('name')
            methods = model.get('supportedGenerationMethods', [])
            if 'generateContent' in methods:
                print(f" - {name} (Supports generateContent)")
            else:
                print(f" - {name} (Does NOT support generateContent)")
    else:
        print(f"\nFAILED. Status Code: {response.status_code}")
        print("Response:", response.text)
        print("\nPossible causes:")
        print("1. 'Generative Language API' is not enabled in Google Cloud Console.")
        print("2. The API Key has restrictions (IP, Referrer, or API restrictions).")
        print("3. The project linked to this key has billing issues.")

except Exception as e:
    print(f"An error occurred: {e}")
