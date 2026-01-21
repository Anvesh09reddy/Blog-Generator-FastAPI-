from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

class GroqLLM:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file

    def get_llm(self):
        try:
            os.environ["GROQ_API_KEY"]=self.groq_api_key=os.getenv("GROQ_API_KEY")
            llm = ChatGroq(api_key=self.groq_api_key,model="llama-3.1-8b-instant")
            return llm
        except Exception as e:
            raise ValueError("Error initializing Groq LLM: {}".format(e))