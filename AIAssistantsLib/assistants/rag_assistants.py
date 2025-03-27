import AIAssistantsLib.config as config
#from AIAssistantsLib import config
from .rag_utils.rag_utils import load_vectorstore

import torch

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.storage import InMemoryByteStore
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import LlamaCpp
from langchain_gigachat import GigaChat
from langchain_community.llms import YandexGPT

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from abc import abstractmethod
from typing import List, Any, Optional, Dict, Tuple

import logging

DISTANCE_TRESHOLD = 0.7
MAX_RETRIEVALS = 20

reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL)
RERANKER = CrossEncoderReranker(model=reranker_model, top_n=3)

class KBRetrieverManager:
    """
    Manager for handling multiple vector store retrievers.
    Ensures that each vector store is loaded only once and provides access to their retrievers.
    """

    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize the KBRetrieverManager.  

        :param embedding_model: The embedding model to use. If None, uses config.EMBEDDING_MODEL.
        """
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.retrievers: Dict[(str, int), Any] = {}  # Maps vector_store_path and max_max_context_length to retriever
        self.vectorstores: Dict[str, Any] = {}  # Maps vector_store_path to vectorstore
        #self.llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini") #os.getenv("OPENAI_API_KEY)
        logging.info(f"KBRetrieverManager initialized with embedding model: {self.embedding_model}")

    def build_retriever(self, vector_store_path, max_context_length, vectorstore, documents):
        #Load document store from persisted storage
        #loading list of problem numbers as ids
        doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
        store = InMemoryByteStore()
        id_key = "problem_number"
        multi_retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key=id_key,
                search_kwargs={"k": MAX_RETRIEVALS, "score_threshold": DISTANCE_TRESHOLD},
            )
        multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
        retriever = ContextualCompressionRetriever(
                base_compressor=RERANKER, base_retriever=multi_retriever
                )

        self.retrievers[(vector_store_path, max_context_length)] = retriever
        self.vectorstores[vector_store_path] = (vectorstore, documents)
        logging.info(f"Vector store '{vector_store_path}' loaded and retriever initialized.")
        return retriever

    def get_retriever(self, vector_store_path: str, max_context_length: int = -1, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Retrieve the retriever for the specified vector store path.
        Loads the vector store if it hasn't been loaded yet.

        :param vector_store_path: Path to the vector store.
        :param search_kwargs: Keyword arguments for the retriever's search method.
        :return: The retriever object.
        """
        if (vector_store_path, max_context_length) in self.retrievers:
            logging.info(f"Retriever for '{vector_store_path}' already loaded. Returning existing retriever.")
            return self.retrievers[(vector_store_path, max_context_length)]

        logging.info(f"Loading vector store from '{vector_store_path}' with embedding model '{self.embedding_model}'.")
        try:
            if vector_store_path not in self.vectorstores:
                 (vectorstore, documents) = load_vectorstore(vector_store_path, self.embedding_model)
            else:
                (vectorstore, documents) = self.vectorstores[vector_store_path]
            retriever = self.build_retriever(vector_store_path, max_context_length, vectorstore, documents)
            return retriever
        except Exception as e:
            logging.error(f"Failed to load vector store from '{vector_store_path}': {e}")
            raise e


    def reload_vector_store(self, vector_store_path: str, max_context_length: int = -1) -> Any:
        """
        Reload a specific vector store and update its retriever.

        :param vector_store_path: Path to the vector store.
        :return: The updated retriever object.
        """
        logging.info(f"Reloading vector store from '{vector_store_path}'.")
        try:
            (vectorstore, documents) = load_vectorstore(vector_store_path, self.embedding_model)
            retriever = self.build_retriever(vector_store_path, max_context_length, vectorstore, documents)
            return retriever
        except Exception as e:
            logging.error(f"Failed to reload vector store from '{vector_store_path}': {e}")
            raise e

    def unload_vector_store(self, vector_store_path: str, max_context_length: int = -1) -> None:
        """
        Unload a specific vector store and remove its retriever from the manager.

        :param vector_store_path: Path to the vector store.
        """
        if vector_store_path in self.retrievers:
            del self.vectorstores[vector_store_path]
            del self.retrievers[(vector_store_path, max_context_length)]
            logging.info(f"Vector store '{vector_store_path}' unloaded and retriever removed.")
        else:
            logging.warning(f"Attempted to unload non-loaded vector store '{vector_store_path}'.")

    def unload_all(self) -> None:
        """
        Unload all vector stores and clear all retrievers.
        """
        self.retrievers.clear()
        self.vectorstores.clear()
        logging.info("All vector stores have been unloaded and retrievers cleared.")

retriever_manager = KBRetrieverManager()

def get_retriever(kkb_path, max_context_window = -1):
    return retriever_manager.get_retriever(kkb_path, max_context_window)

def reload_vectore_store(kkb_path, max_context_window = -1):
    return retriever_manager.reload_vector_store(kkb_path, max_context_window)

class KBDocumentPromptTemplate(StringPromptTemplate):
    max_length : int = 0
    def __init__(self, max_length: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_length = max_length

    def format(self, **kwargs: Any) -> str:
        page_content = kwargs.pop("page_content")
        problem_number = kwargs.pop("problem_number")
        chunk_size = kwargs.pop("actual_chunk_size")
        #here additional data could be retrieved based on problem_number
        result = page_content
        if self.max_length > 0:
            result = result[:self.max_length]
        return result

    @property
    def _prompt_type(self) -> str:
        return "kb_document"

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGAssistant:
    def __init__(self, system_prompt, kkb_path, max_context_window = -1, output_parser = BaseOutputParser, model_name=None, temperature=0.4):
        logging.info(f"Initializing model with: {kkb_path}")
        self.model_name=model_name
        self.temperature=temperature
        self.system_prompt = system_prompt
        self.max_context_window = max_context_window
        self.output_parser = output_parser
        self.kkb_path = kkb_path
        self.retriever = get_retriever(self.kkb_path, self.max_context_window)
        logging.info(f"Dataretrieved built: {kkb_path}")
        self.llm = self.initialize()
        self.set_system_prompt(self.system_prompt)
        logging.info("Initialized")

    def truncate_context(self, context, question, max_tokens):
        # Default implementation: no truncation
        return context

    def reload_vectore_store(self):
        self.retriever = reload_vectore_store(self.kkb_path, self.max_context_window)
        self.set_system_prompt(self.system_prompt)
        return 

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.prompt = self.get_prompt(self.system_prompt)

        def get_chat_prompt_template_length(chat_prompt: ChatPromptTemplate) -> int:
            total_length = 0
            for message in chat_prompt.messages:
                if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                    total_length += len(message.prompt.template)
                else:
                    # For custom message types, fallback to string representation
                    total_length += len(str(message))
            return total_length

        max_length = self.max_context_window - get_chat_prompt_template_length(self.prompt)
        my_prompt = KBDocumentPromptTemplate(max_length, input_variables=["page_content", "problem_number", "actual_chunk_size"])

        docs_chain = create_stuff_documents_chain(self.llm, self.prompt, output_parser=self.output_parser(), document_prompt=my_prompt, document_separator='\n#EOD\n\n')
        self.rag_chain = create_retrieval_chain(self.retriever, docs_chain)

    def get_prompt(self, system_prompt):
        #return ChatPromptTemplate.from_template(system_prompt)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "User request: \n{input}\n\nContext: \n{context}"),
            ]
        )
    @abstractmethod
    def initialize(self):
        """
            Initialize model here.
        """

    def ask_question(self, query: str) -> str:
        if self.rag_chain is None:
            logging.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            result = self.rag_chain.invoke({"input": query})
            return result #result['answer']
        except AttributeError as e:
            logging.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in ask_question: {str(e)}")
            raise


class RAGAssistantGPT(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name="gpt-4o-mini", temperature=0.4):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser, model_name=model_name, temperature=temperature)
    def initialize(self):
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature)

class RAGAssistantMistralAI(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name="mistral-large-latest", temperature=0.4):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser, model_name=model_name, temperature=temperature)
    def initialize(self):
        return ChatMistralAI(
            model=self.model_name,
            temperature=self.temperature)


class RAGAssistantYA(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/rc', temperature=0.4):
        super().__init__(system_prompt, kkb_path, 7200, output_parser = output_parser, model_name=model_name, temperature=temperature)
    def initialize(self):
        return YandexGPT(
            #iam_token = None,
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=self.model_name,
            temperature=self.temperature
            )

class RAGAssistantSber(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name="GigaChat-Pro", temperature=0.4):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser, model_name=model_name, temperature=temperature)
    def generate_auth_data(self, user_id, secret):
        return {"user_id": user_id, "secret": secret}
    def initialize(self):
        return GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model=self.model_name,
            verify_ssl_certs=False,
            temperature=self.temperature,
            scope = config.GIGA_CHAT_SCOPE)
    
class RAGAssistantGemini(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name="gemini-1.5-pro", temperature=0.4):
        super().__init__(system_prompt, kkb_path, output_parser = output_parser, model_name=model_name, temperature=temperature)

    def initialize(self):
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=config.GEMINI_API_KEY,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )


class RAGAssistantLocal(RAGAssistant):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name=config.LOCAL_MODEL_NAME, temperature=0.4):
        self.max_new_tokens = 2000
        super().__init__(system_prompt, kkb_path, 4096, output_parser = output_parser, model_name=model_name, temperature=temperature)

    def initialize(self):
        # Load the text generation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")

        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = self.temperature
        generation_config.top_p = 0.9
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.2
        generation_config.eos_token_id=self.tokenizer.eos_token_id,
        generation_config.pad_token_id=self.tokenizer.eos_token_id

        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, generation_config=generation_config,)
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.4})
        return llm

    def get_prompt(self, system_prompt):
        #return ChatPromptTemplate.from_template(system_prompt)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Context: {context}"),
                ("human", "{input}"),
                ("ai", ""),
            ]
        )

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.prompt = self.get_prompt(self.system_prompt)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        def truncate_chain(inputs):
            context = format_docs(inputs["context"])
            truncated_context = self.truncate_context(context, inputs["question"], self.max_context_window)
            return {
                "context": truncated_context,
                "question": inputs["question"]
            }
        
        def get_chat_prompt_template_length(chat_prompt: ChatPromptTemplate) -> int:
            total_length = 0
            for message in chat_prompt.messages:
                if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                    total_length += len(message.prompt.template)
                else:
                    # For custom message types, fallback to string representation
                    total_length += len(str(message))
            return total_length

        max_length = self.max_context_window - get_chat_prompt_template_length(self.prompt)
        my_prompt = KBDocumentPromptTemplate(max_length, input_variables=["page_content", "problem_number", "actual_chunk_size"])
        docs_chain = create_stuff_documents_chain(self.llm, self.prompt, output_parser=self.output_parser(), document_prompt=my_prompt, document_separator='\n#EOD\n\n')
        self.rag_chain = create_retrieval_chain(self.retriever, docs_chain)

    def ask_question(self, query: str) -> str:
        if self.rag_chain is None:
            logging.error("RAG chain not initialized")
            raise ValueError("Model or RAG chain not initialized.")
        try:
            result = self.rag_chain.invoke({"input": query})
            return result
        except AttributeError as e:
            logging.error(f"AttributeError in ask_question: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in ask_question: {str(e)}")
            raise

    def count_tokens(self, text):
        if isinstance(self.llm, LlamaCpp):
            return self.llm.get_num_tokens(text)
        else:
            return len(self.tokenizer.encode(text))

    def truncate_input(self, text, max_tokens):
        if isinstance(self.llm, LlamaCpp):
            while self.count_tokens(text) > max_tokens:
                text = text[:int(len(text) * 0.9)]  # Reduce by 10% and retry
        else:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                text = self.tokenizer.decode(tokens[:max_tokens])
        return text

    def truncate_context(self, context, question, max_tokens):
        question_tokens = self.count_tokens(question)
        system_tokens = self.count_tokens(self.system_prompt)
        available_tokens = max_tokens - question_tokens - system_tokens - self.max_new_tokens
        return self.truncate_input(context, available_tokens)

class RAGAssistantGGUF(RAGAssistantLocal):
    def __init__(self, system_prompt, kkb_path, output_parser = BaseOutputParser, model_name=config.LOCALGGUF_MODEL_NAME, temperature=0.4):
        super().__init__(system_prompt, kkb_path, output_parser, model_name, temperature)

    def initialize(self):
        # Load the text generation model and tokenizer
        logging.info(f"loading {self.model_name}...")
        try:
            llm = LlamaCpp(
                model_path=self.model_name,
                temperature=self.temperature,
                top_p=0.9,
                max_tokens=self.max_new_tokens,
                n_ctx=self.max_context_window,
                echo=False
            )
        except Exception as e:
            logging.error(f"Error loading {self.model_name}: {str(e)}")
        logging.info(f"...{self.model_name} loaded")
        self.tokenizer = None
        return llm

if __name__ == '__main__':
    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
        BooleanOptionalAction,
    )
    from langchain_core.output_parsers import StrOutputParser

    vectorestore_path = 'data/vectorstore_e5'

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='query', 
        choices = ['query'],
        help='query - query vectorestore\n'
    )
    args = vars(parser.parse_args())
    mode = args['mode']

    with open('prompts/system_prompt_markdown_3.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    if mode == 'query':
        assistants = []
        vectorstore = load_vectorstore(vectorestore_path, config.EMBEDDING_MODEL)
        retriever = get_retriever(vectorestore_path)
        #assistants.append(RAGAssistantGPT(system_prompt, vectorestore_path, output_parser=StrOutputParser))
        assistants.append(RAGAssistantLocal(system_prompt, vectorestore_path, output_parser=StrOutputParser, model_name='HuggingFaceTB/SmolLM2-1.7B-Instruct'))

        query = ''

        while query != 'stop':
            print('=========================================================================')
            query = input("Enter your query: ")
            if query != 'stop':
                for assistant in assistants:
                    try:
                        reply = assistant.ask_question(query)
                    except Exception as e:
                        logging.error(f'Error: {str(e)}')
                        continue
                    print(f'{reply['answer']}')
                    print('=========================================================================')
