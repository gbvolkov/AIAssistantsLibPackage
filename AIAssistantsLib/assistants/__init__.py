# assistants/__init__.py

# Import classes from simple_assistants
from .simple_assistants import (
    SimpleAssistant,
    SimpleAssistantGPT,
    SimpleAssistantMistralAI,
    SimpleAssistantYA,
    SimpleAssistantSber,
    SimpleAssistantGemini,
    SimpleAssistantLocal,
)

# Import classes from rag_assistants
from .rag_assistants import (
    RAGAssistant,
    RAGAssistantGPT,
    RAGAssistantMistralAI,
    RAGAssistantYA,
    RAGAssistantSber,
    RAGAssistantGemini,
    RAGAssistantLocal,
    RAGAssistantGGUF
)

# Import classes from json_assistants
from .json_assistants import (
    JSONAssistant,
    JSONAssistantGPT,
    JSONAssistantMistralAI,
    JSONAssistantYA,
    JSONAssistantSber,
    JSONAssistantGemini
)

# Import classes from chat_assistants
from .chat_assistants import (
    ChatAssistant,
    ChatAssistantGPT,
    ChatAssistantMistralAI,
    ChatAssistantYA,
    ChatAssistantSber,
    ChatAssistantGemini
)

# Optionally, import utilities from rag_utils if you want them easily accessible
# If only certain utils are needed, import them individually
from .rag_utils.rag_utils import load_vectorstore, show_retrieved_documents

# Define __all__ for clarity and to limit what gets imported on wildcard imports
__all__ = [
    "SimpleAssistant",
    "SimpleAssistantGPT",
    "SimpleAssistantMistralAI",
    "SimpleAssistantYA",
    "SimpleAssistantSber",
    "SimpleAssistantGemini",
    "SimpleAssistantLocal",
    "RAGAssistant",
    "RAGAssistantGPT",
    "RAGAssistantMistralAI",
    "RAGAssistantYA",
    "RAGAssistantSber",
    "RAGAssistantGemini",
    "RAGAssistantLocal",
    "RAGAssistantGGUF",
    "JSONAssistant",
    "JSONAssistantGPT",
    "JSONAssistantMistralAI",
    "JSONAssistantYA",
    "JSONAssistantSber",
    "JSONAssistantGemini",
    "ChatAssistant",
    "ChatAssistantGPT",
    "ChatAssistantMistralAI",
    "ChatAssistantYA",
    "ChatAssistantSber",
    "ChatAssistantGemini",
    "load_vectorstore",
    "show_retrieved_documents",
    # add others as needed
]


import logging
import os

logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

