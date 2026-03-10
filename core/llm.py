from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
AMZN_KEY = os.getenv("AMAZON_API_KEY")
if AMZN_KEY is None:
    raise ValueError("AMZN_KEY environment variable not set")

def get_llm():
     model = ChatOpenAI(
        model="nova-2-lite-v1",
        api_key=AMZN_KEY,
        base_url="https://api.nova.amazon.com/v1",
        max_tokens=2048,
        temperature=0.7,
    )
     return model
    