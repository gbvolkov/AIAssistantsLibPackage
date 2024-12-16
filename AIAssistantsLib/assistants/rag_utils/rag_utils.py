import os

import pickle
from tiktoken import get_encoding, Encoding

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever 
from langchain.schema import Document

from typing import List, Any, Optional, Dict, Tuple

import logging

def load_vectorstore(file_path: str, embedding_model_name: str) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")
    
    logging.info(f"Loading vectorstore  from {file_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    logging.info(f"Vectorstore loaded from {file_path}")

    # Load docstore
    with open(f'{file_path}/docstore.pkl', 'rb') as file:
        documents = pickle.load(file)
    logging.info(f"Documentstore loaded from {file_path}/docstore.pkl")

    return (vectorstore, documents)


def show_retrieved_documents(vectorstore, retriever, query):
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    scores = [score for _, score in results_with_scores]

    logging.info(f"\n>>{query}==========================\n")
    # Check if documents have scores in metadata
    index = vectorstore.index  # FAISS index object

    # Check the metric
    metric = index.metric_type
    logging.info(f"FAISS Metric Type: {metric}\n")

    for doc, score in results_with_scores:
        doc_id = doc.metadata.get('problem_number')
        description = doc.page_content
        idx_start = description.find('Problem Description')
        inx_end = description.find('Systems')
        logging.info(f"Document: {doc_id}: {description[idx_start:inx_end-1]}; ===>Similarity Score: {score}")
    logging.info(f"\n=========================={query}<<\n\n")


class ThresholdBasedRetriever(BaseRetriever):
    """
    A custom retriever that filters documents based on a distance threshold.
    """
    vectorstore: FAISS  # Type-annotated attribute
    distance_threshold: float
    k: int = 5  # Default value for top-k documents
    tokenizer: Encoding = get_encoding("o200k_base")
    max_tokens: int = -1

    def _count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text using the tokenizer.

        Parameters:
        - text (str): The text to tokenize.

        Returns:
        - int: Number of tokens.
        """
        return len(self.tokenizer.encode(text))
    
    def get_tokens_allocation(self, sorted_docs_and_scores: List[Tuple[Document, float]]) -> List[int]:
        # Calculate importance for each document (inverse of score)
        epsilon = 1e-6  # To prevent division by zero
        importances = [1 / (score + epsilon) for _, score in sorted_docs_and_scores]
        total_importance = sum(importances)

        # Number of documents
        num_docs = len(sorted_docs_and_scores)

        # Allocate tokens proportionally based on importance
        raw_allocations = [(imp / total_importance) * self.max_tokens for imp in importances]
        # Convert to integers
        allocated_tokens = [int(allocation) for allocation in raw_allocations]

        # Handle any remaining tokens due to integer rounding
        allocated_sum = sum(allocated_tokens)
        remaining_tokens = self.max_tokens - allocated_sum

        if remaining_tokens > 0:
            # Distribute the remaining tokens starting from the most important document
            for i in range(remaining_tokens):
                allocated_tokens[i % num_docs] += 1
        elif remaining_tokens < 0:
            # If over-allocated, remove tokens starting from the least important document
            for i in range(-remaining_tokens):
                idx = -(i % num_docs) - 1
                if allocated_tokens[idx] > 1:
                    allocated_tokens[idx] -= 1
        return allocated_tokens

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the query based on the distance threshold
        and ensure the total tokens do not exceed max_tokens.

        Parameters:
        - query (str): The user query.

        Returns:
        - List[Document]: Filtered list of relevant documents within token limit.
        """
        # Perform similarity search with distance scores
        docs_and_scores: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        filtered_docs = []
        total_tokens = 0

        for doc, score in docs_and_scores:
            doc_id = doc.metadata.get('problem_number', 'N/A')
            
            # Since METRIC_L2 is used, lower scores are better
            if score <= self.distance_threshold:
                if self.max_tokens > 0:
                    doc_tokens = self._count_tokens(doc.page_content)
                    logging.info(f"Document ID: {doc_id}, Distance Score: {score}, Tokens: {doc_tokens}")
                    
                    if total_tokens + doc_tokens > self.max_tokens:
                        logging.info(f"Skipping Document ID: {doc_id} to maintain max_tokens limit.")
                        continue  # Skip this document as it would exceed the token limit
                    total_tokens += doc_tokens
                
                logging.info(f"Added Document ID: {doc_id}, Distance Score: {score} below threshold {self.distance_threshold}")
                filtered_docs.append(doc)
            else:
                logging.info(f"Rejected Document ID: {doc_id}, Distance Score: {score} above threshold {self.distance_threshold}")
        
        logging.info(f"Total tokens in returned documents: {total_tokens} (max allowed: {self.max_tokens})")
        return filtered_docs
