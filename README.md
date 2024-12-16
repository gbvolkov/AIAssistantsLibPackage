```markdown
# AIAssistantsLib

AIAssistantsLib is a Python library designed to simplify the creation and integration of various AI-powered assistants. It provides classes and utilities for both simple language model assistants and Retrieval-Augmented Generation (RAG) assistants, along with utilities for managing vector stores and retrieving context.

## Features

- **Simple Assistants:** Easily create assistants powered by different LLMs, such as GPT, MistralAI, YandexGPT, GigaChat, and more.
- **RAG Assistants:** Implement Retrieval-Augmented Generation to ground model responses with context from vector stores.
- **Modular Structure:** Each assistant type is neatly organized, making it easy to integrate with other projects.
- **Logging & Configuration:** Centralized logging and configuration for consistent behavior across the entire library.

## Installation

You can install AIAssistantsLib directly from PyPI once it is published:

```bash
pip install AIAssistantsLib
```

If you are installing from source (assuming you have the `pyproject.toml` file in the root directory):

```bash
cd AIAssistantsLibPackage
pip install .
```

## Usage

After installation, you can import and use the available assistants. For example, to use a simple GPT-based assistant:

```python
from AIAssistantsLib.assistants import SimpleAssistantGPT

assistant = SimpleAssistantGPT(system_prompt="You are a helpful assistant.", model_name="gpt-4o-mini")
response = assistant.ask_question({"query": "What is the capital of France?"})
print(response)
```

Similarly, for RAG-based assistants:

```python
from AIAssistantsLib.assistants import RAGAssistantGPT

system_prompt = "You are an assistant that uses retrieved documents to answer questions."
assistant = RAGAssistantGPT(system_prompt, kkb_path="path/to/vectorstore")
response = assistant.ask_question("What is the main topic of the retrieved documents?")
print(response)
```

## Project Structure

```
AIAssistantsLibPackage/
    pyproject.toml
    README.md
    LICENSE
    requirements.txt
    tests/
        test_something.py
    AIAssistantsLib/
        __init__.py
        assistants/
            __init__.py
            simple_assistants.py
            rag_assistants.py
            json_assistants.py
            rag_utils/
                __init__.py
                rag_utils.py
```

- **AIAssistantsLib/**: The Python package containing all source code.
- **assistants/**: Subpackage holding various assistant classes and utilities.
- **rag_utils/**: Helper functions for working with vector stores and retrieval.

## Configuration & Logging

The library sets some environment variables and a global logger in `assistants/__init__.py`. You can use the provided logger in your code:

```python
from AIAssistantsLib.assistants import logger

logger.info("Logging an informational message")
```

## Contributing

Contributions are welcome! If you find bugs, have ideas, or would like to add new features:

1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature`.
3. Make your changes.
4. Commit your changes: `git commit -m 'Add my feature'`.
5. Push to your branch: `git push origin my-feature`.
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
```