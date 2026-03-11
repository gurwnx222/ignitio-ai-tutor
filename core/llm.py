from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import httpx
import os

load_dotenv()

AMZN_KEY = os.getenv("AMAZON_API_KEY")
if AMZN_KEY is None:
    raise ValueError("AMZN_KEY environment variable not set")

def get_llm():
    # Force gzip/deflate only — Amazon Nova returns zstd by default,
    # which causes httpx's decompressor to fail on re-use.
    http_client = httpx.Client(
        headers={"Accept-Encoding": "gzip, deflate"}
    )

    model = ChatOpenAI(
        model="nova-2-lite-v1",
        api_key=AMZN_KEY,
        base_url="https://api.nova.amazon.com/v1",
        max_tokens=2048,
        temperature=0.7,
        http_client=http_client,
    )
    return model