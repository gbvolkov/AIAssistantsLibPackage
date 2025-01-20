import os

import pickle
from tiktoken import get_encoding, Encoding

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
