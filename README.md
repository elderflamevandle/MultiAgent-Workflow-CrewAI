# Retrieval-Augmented Generation (RAG) Agent
This project implements a Retrieval-Augmented Generation (RAG) agent using the langchain-ollama, crewai, and other supporting libraries.

# Overview
The RAG agent is designed to answer questions by leveraging both retrieval-based and generation-based approaches. The system is composed of several specialized agents, each responsible for a specific task in the overall question-answering process.

# Agent Flow

1. **Router Agent:** The Router_Agent is responsible for analyzing the keywords in the input question and deciding whether it should be routed to the vectorstore or the web search.

2. **Retriever Agent:** The Retriever_Agent is responsible for retrieving the relevant information from either the vectorstore or the web, based on the decision made by the Router_Agent.

3. **Grader Agent:** The Grader_agent is responsible for evaluating the relevance of the retrieved information to the input question.

4. **Hallucination Grader:** The hallucination_grader is responsible for checking whether the answer generated by the system is grounded in the retrieved facts.

5. **Answer Grader:** The answer_grader is responsible for evaluating the usefulness of the generated answer and, if necessary, performing a web search to generate a better response.

# The flow of the system is as follows:

- The user provides an input question.
- The Router_Agent analyzes the question and decides whether to route it to the vectorstore or the web search.
- The Retriever_Agent retrieves the relevant information based on the decision made by the Router_Agent.
- The Grader_agent evaluates the relevance of the retrieved information.
- The hallucination_grader checks whether the generated answer is grounded in the retrieved facts.
- The answer_grader evaluates the usefulness of the generated answer and, if necessary, performs a web search to generate a better response.
- The final result is returned to the user.

This modular design makes the system more extensible and easier to maintain, as new agents and tasks can be added or existing ones can be modified without affecting the overall structure.

# Installation
To install the required dependencies, run the following command:
``` 
pip install -r requirements.txt 
```
# Usage 

* Run the Flask app:
   ``` 
   python app.py 
   ```
* The RAGAgent class is the main entry point for using the RAG agent. Here's an example of how to use it:

```
from langchain_ollama import ChatOllama
from crewai_tools import PDFSearchTool
from crewai_tools import tool
from crewai import Crew
from crewai import Task
from crewai import Agent

llm = ChatOllama(model="mistral:instruct")
pdf_path = 'doc.pdf'
config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama3-8b-8192",
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5",
        }
    }
}
rag_agent = RAGAgent(llm, pdf_path, config)

question = "User Question will go here"
result = rag_agent.run(question)
print(result)
```

This will create an instance of the ```RAGAgent class```, which will then be used to answer the given question.

# Contributing
If you'd like to contribute to this project, please feel free to submit a pull request or open an issue on the project's GitHub repository.

# License
This project is licensed under the MIT Licence.


