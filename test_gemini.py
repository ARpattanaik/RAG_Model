from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words."
)

print(response.text)