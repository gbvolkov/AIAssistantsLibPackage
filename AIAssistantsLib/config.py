from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path

documents_path = Path.home() / ".env"

load_dotenv(os.path.join(documents_path, 'gv.env'))

LOCAL_MODEL_NAME=os.environ.get('LOCAL_MODEL_NAME') or '/models/llama3.1.8b'
LOCALGGUF_MODEL_NAME=os.environ.get('LOCALGGUF_MODEL_NAME') or '/models/mistral-large-instruct-2411-Q4_K_M'
EMBEDDING_MODEL=os.environ.get('EMBEDDING_MODEL') or '/models/multilingual-e5-large'
RERANKING_MODEL=os.environ.get('RERANKING_MODEL') or '/models/bge-reranker-large'

GIGA_CHAT_USER_ID=os.environ.get('GIGA_CHAT_USER_ID')
GIGA_CHAT_SECRET = os.environ.get('GIGA_CHAT_SECRET')
GIGA_CHAT_AUTH = os.environ.get('GIGA_CHAT_AUTH')
GIGA_CHAT_SCOPE = "GIGACHAT_API_PERS"

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

YA_API_KEY = os.environ.get('YA_API_KEY')
YA_FOLDER_ID = os.environ.get('YA_FOLDER_ID')
YA_AUTH_TOKEN = os.environ.get('YA_AUTH_TOKEN')

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

JINA_API_KEY=os.environ.get('JINA_API_KEY')
