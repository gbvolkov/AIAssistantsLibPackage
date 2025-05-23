##TODO: Complete this classes to keep chat history!!!

import os

# Add the parent directory to sys.path
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AIAssistantsLib.config as config
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
#from yandex_chain import YandexLLM
from langchain_community.llms import YandexGPT
#from langchain_community.llms import YandexGPT


from abc import abstractmethod
from typing import List, Any, Optional, Dict, Tuple

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAssistant:
    def __init__(self, system_prompt: Optional[str] = None, model_name=None, temperature=0.4):
        self.model_name=model_name
        self.temperature=temperature
        logger.info("Initializing chat model")
        self.llm = self.initialize()
        self.conversation_history = []
        if system_prompt:
            self.set_system_prompt(system_prompt)
        else:
            logger.warning("No system prompt provided. The assistant may not behave as expected.")

        logger.info("Initialized")

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt


    @abstractmethod
    def initialize(self):
        """
            Initialize model here.
        """

    def ask_question(self, query: str) -> str:
        if self.llm is None:
            logger.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            result = self.llm.invoke(query)
            return result.content
        except AttributeError as e:
            logger.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in ask_question: {str(e)}")
            raise


class ChatAssistantGPT(ChatAssistant):
    def __init__(self, system_prompt, model_name="gpt-4.1-mini", temperature=0.4):
        super().__init__(system_prompt, model_name=model_name, temperature=temperature)
    def initialize(self):
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature)


class ChatAssistantMistralAI(ChatAssistant):
    def __init__(self, system_prompt, model_name="mistral-large-latest", temperature=0.4):
        super().__init__(system_prompt, model_name=model_name, temperature=temperature)
    def initialize(self):
        return ChatMistralAI(
            model=self.model_name,
            temperature=self.temperature)


class ChatAssistantYA(ChatAssistant):
    def __init__(self, system_prompt, model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/latest', temperature=0.4):
        super().__init__(system_prompt, model_name=model_name, temperature=temperature)
    def initialize(self):
        return YandexGPT(
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model=self.model_name,
            temperature=self.temperature)

class ChatAssistantSber(ChatAssistant):
    def __init__(self, system_prompt, model_name="GigaChat-Pro", temperature=0.4):
        super().__init__(system_prompt, model_name=model_name, temperature=temperature)
    def generate_auth_data(self, user_id, secret):
        return {"user_id": user_id, "secret": secret}
    def initialize(self):
        return GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model=self.model_name,
            temperature=self.temperature,
            verify_ssl_certs=False,
            scope = config.GIGA_CHAT_SCOPE)
    
class ChatAssistantGemini(ChatAssistant):
    def __init__(self, system_prompt, model_name="gemini-1.5-pro", temperature=0.4):
        super().__init__(system_prompt, model_name=model_name, temperature=temperature)

    def initialize(self):
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=config.GEMINI_API_KEY,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )


