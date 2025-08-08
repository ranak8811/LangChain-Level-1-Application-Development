# LangChain Level 1 Application Development

This repository contains a series of beginner-friendly Python applications demonstrating fundamental concepts and common use cases of the LangChain framework. Each script focuses on a specific aspect of building LLM-powered applications, from simple chatbots to more advanced data extraction and retrieval-augmented generation (RAG).

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Prerequisites](#prerequisites)
3.  [Setup Instructions](#setup-instructions)
4.  [Running the Applications](#running-the-applications)
5.  [Application Descriptions](#application-descriptions)
    - [001-simple-chatbot.py](#001-simple-chatbotpy)
    - [002-advanced-chatbot.py](#002-advanced-chatbotpy)
    - [003-key-data-extraction.py](#003-key-data-extractionpy)
    - [004-sentiment-analysis.py](#004-sentiment-analysispy)
    - [005-qa-from-sql.py](#005-qa-from-sqlpy)
    - [006-qa-from-pdf.py](#006-qa-from-pdfpy)
    - [007-retriever-app.py](#007-retriever-apppy)
    - [008-simple-rag.py](#008-simple-ragpy)
6.  [Dependencies and Versions](#dependencies-and-versions)
7.  [Troubleshooting](#troubleshooting)

## Project Overview

This project aims to provide hands-on examples for learning LangChain. Each Python script is a self-contained example that you can run to understand how different LangChain components work together.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Python 3.11 or higher**: It's recommended to use Python 3.11.4 or newer for full compatibility with all libraries.
- **Google API Key**: You will need a Google API Key to interact with Google's Generative AI models (e.g., Gemini). You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup Instructions

Follow these steps to set up your local development environment:

1.  **Clone the repository (if you haven't already):**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

    (Note: If you are already in the project directory, you can skip this step.)

2.  **Create a Python Virtual Environment:**
    It's good practice to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv level1Env
    ```

3.  **Activate the Virtual Environment:**

    - **On macOS/Linux:**
      ```bash
      source level1Env/bin/activate
      ```
    - **On Windows (Command Prompt):**
      ```bash
      level1Env\Scripts\activate.bat
      ```
    - **On Windows (PowerShell):**
      ```bash
      level1Env\Scripts\Activate.ps1
      ```

4.  **Install Dependencies:**
    Install all required Python packages using `pip` and the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your Google API Key:**
    Create a `.env` file in the root of your project directory and add your Google API Key to it:
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    Replace `"your_google_api_key_here"` with your actual API key.

## Running the Applications

To run any of the applications, first ensure your virtual environment is activated (as shown in step 3 of Setup Instructions). Then, execute the Python script using `python3`.

For example, to run the simple chatbot:

```bash
source level1Env/bin/activate # if not already active
python3 001-simple-chatbot.py
```

To save the output of a script to a text file:

```bash
source level1Env/bin/activate # if not already active
python3 001-simple-chatbot.py > 001-simple-chatbot-output.txt
```

## Application Descriptions

Here's a brief overview of each application:

### 001-simple-chatbot.py

This script demonstrates a basic chatbot using LangChain. It covers:

- Initializing a `ChatGoogleGenerativeAI` model.
- Sending simple messages to the chatbot.
- Implementing conversation memory using `ConversationBufferMemory` and `FileChatMessageHistory` to maintain context across turns.
- Using `RunnableSequence` for chaining components.

### 002-advanced-chatbot.py

Builds upon the simple chatbot by introducing:

- Managing chat history for multiple sessions using `RunnableWithMessageHistory`.
- Implementing a custom function to limit the memory of messages, showcasing more control over conversation context.

### 003-key-data-extraction.py

Focuses on extracting structured data from unstructured text using LangChain's output parsers and Pydantic models. It shows how to:

- Define a Pydantic model for the desired output schema.
- Use `JsonOutputParser` to parse LLM responses into structured JSON.
- Create prompts for data extraction, including few-shot examples.

### 004-sentiment-analysis.py

Demonstrates how to perform sentiment analysis and other text classifications using LLMs. Key features include:

- Defining Pydantic models with `Enum` types for constrained output.
- Using `JsonOutputParser` for structured classification results.
- Applying few-shot prompting techniques to guide the LLM's classification.

### 005-qa-from-sql.py

Illustrates how to build a question-answering system over a SQL database. It covers:

- Connecting to a SQLite database using `SQLDatabase`.
- Generating SQL queries from natural language questions using `create_sql_query_chain`.
- Executing SQL queries and retrieving results.
- Constructing a full QA chain that generates a query, executes it, and then uses the LLM to answer the original question based on the SQL result.

### 006-qa-from-pdf.py

Shows how to build a RAG (Retrieval-Augmented Generation) system for querying information from PDF documents. It includes:

- Loading PDF documents using `PyPDFLoader`.
- Splitting documents into smaller chunks with `RecursiveCharacterTextSplitter`.
- Creating a vector store (`Chroma`) from document chunks and embeddings (`GoogleGenerativeAIEmbeddings`).
- Setting up a retriever to fetch relevant document snippets.
- Building a retrieval chain to answer questions based on the retrieved context.

### 007-retriever-app.py

Explores different ways to use retrievers in LangChain. It demonstrates:

- Creating a vector store from in-memory documents.
- Performing similarity searches with and without scores.
- Configuring retrievers with different search types and parameters (e.g., `k` for number of results).
- Using `RunnableLambda` and `RunnablePassthrough` to integrate retrievers into a chain for question answering.

### 008-simple-rag.py

Provides another example of a RAG application, focusing on text files. It covers:

- Loading text documents using `TextLoader`.
- Text splitting and vector store creation similar to the PDF example.
- Using a pre-defined RAG prompt (similar to `rlm/rag-prompt` from LangChain Hub).
- Constructing a RAG chain that retrieves context, formats it, and passes it to the LLM for answering.

## Dependencies and Versions

The following are the key Python libraries and their versions used in this project, as specified in `requirements.txt`:

- `python-dotenv==1.0.1`
- `langchain-google-genai>=1.0.5`
- `langchain-community>=0.2.10`
- `langchain-chroma>=0.1.2`
- `langchain==0.2.12`
- `langchain-core>=0.2.12`
- `pypdf==4.3.1`
- `bs4==0.0.2`

These versions ensure compatibility and leverage the latest features and fixes from the LangChain ecosystem.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'langchain'`**: Ensure your virtual environment is activated and you have run `pip install -r requirements.txt`.
- **Deprecation Warnings**: LangChain is under active development. While efforts have been made to update the code to newer patterns, you might still encounter deprecation warnings. These typically indicate that a function or class will be removed in a future version. The code should still run, but it's good practice to keep an eye on LangChain's official documentation for the latest recommended approaches.
- **Missing Database/PDF Files**: If you encounter errors related to missing files (e.g., `street_tree_db.sqlite`, `Be_Good.pdf`), ensure these files are present in the `data/` directory. They should be provided alongside the code.
- **Telemetry Errors (e.g., `capture() takes 1 positional argument but 3 were given`)**: This error often indicates a version incompatibility with `chromadb` or its dependencies. Try updating `chromadb` specifically:
  ```bash
  source level1Env/bin/activate
  pip install --upgrade chromadb
  ```
  If the issue persists, you might need to try a specific older version of `chromadb` that is known to be stable with the other `langchain` components.
