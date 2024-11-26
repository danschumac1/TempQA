import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

def api_config() -> OpenAI:
    load_dotenv('./resources/.env')

    # Get the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Check if the API key is available
    if api_key is None:
        raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

    # Set the API key for the OpenAI client
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    return client