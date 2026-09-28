import os

from dotenv import load_dotenv

load_dotenv()

print(os.getcwd())

print(os.getenv("GOOGLE_API_KEY") is not None)