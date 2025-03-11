
from AIAssistantsLib.assistants.rag_utils.rag_utils import load_vectorstore
from AIAssistantsLib.assistants.rag_assistants import get_retriever, RAGAssistantLocal, RAGAssistantMistralAI
import AIAssistantsLib.config as config
import os

import logging

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

    with open('prompts/system_prompt_short.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if mode == 'query':
        assistants = []
        vectorstore = load_vectorstore(vectorestore_path, config.EMBEDDING_MODEL)
        retriever = get_retriever(vectorestore_path)
        #assistants.append(RAGAssistantGPT(system_prompt, vectorestore_path, output_parser=StrOutputParser))
        #model_name = "mistralai/Ministral-8B-Instruct-2410"
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        assistants.append(RAGAssistantLocal(system_prompt, vectorestore_path, output_parser=StrOutputParser, model_name=model_name))
        #assistants.append(RAGAssistantMistralAI(system_prompt, vectorestore_path, output_parser=StrOutputParser))

        query = ''
        while query != 'stop':
            print('=========================================================================')
            #query = input("Enter your query: ")
            query = "Кто такие key users?"
            if query != 'stop':
                for assistant in assistants:
                    try:
                        reply = assistant.ask_question(query)
                    except Exception as e:
                        logging.error(f'Error: {str(e)}')
                        continue
                    print(f'{type(assistant).__name__} answers:\n{reply['answer']}')
                    print('=========================================================================')
