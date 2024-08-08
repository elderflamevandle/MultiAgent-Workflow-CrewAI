import os
from typing import Tuple, Dict, Any
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from crewai_tools import PDFSearchTool
from crewai_tools import tool
from crewai import Crew
from crewai import Task
from crewai import Agent

class RAGAgent:
    def __init__(self, llm: ChatOllama, pdf_path: str, config: Dict[str, Any]):
        self.llm = llm
        self.rag_tool = PDFSearchTool(pdf=pdf_path, config=config)
        self.router_agent = self._create_router_agent()
        self.retriever_agent = self._create_retriever_agent()
        self.grader_agent = self._create_grader_agent()
        self.hallucination_grader = self._create_hallucination_grader()
        self.answer_grader = self._create_answer_grader()

        self.rag_crew = Crew(
            agents=[self.router_agent, self.retriever_agent, self.grader_agent, self.hallucination_grader, self.answer_grader],
            tasks=[self.router_task, self.retriever_task, self.grader_task, self.hallucination_task, self.answer_task],
            verbose=True,
        )

    def _create_router_agent(self) -> Agent:
        @tool
        def router_tool(question: str) -> str:
            """Router Function"""
            if 'Sporo Health' in question:
                return 'vectorstore'
            else:
                return 'web_search'

        Router_Agent = Agent(
            role='Router',
            goal='Route user question to a vectorstore or web search',
            backstory=(
                "You are an expert at routing a user question to a vectorstore or web search."
                "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
                "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.router_task = Task(
            description=(
                "Analyse the keywords in the question {question}"
                "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
                "Return a single word 'vectorstore' if it is eligible for vectorstore search."
                "Return a single word 'websearch' if it is eligible for web search."
                "Do not provide any other premable or explaination."
            ),
            expected_output=(
                "Give a binary choice 'websearch' or 'vectorstore' based on the question"
                "Do not provide any other premable or explaination."
            ),
            agent=Router_Agent,
            tools=[router_tool],
        )
        return Router_Agent

    def _create_retriever_agent(self) -> Agent:
        Retriever_Agent = Agent(
            role="Retriever",
            goal="Use the information retrieved from the vectorstore to answer the question",
            backstory=(
                "You are an assistant for question-answering tasks."
                "Use the information present in the retrieved context to answer the question."
                "You have to provide a clear concise answer."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.retriever_task = Task(
            description=(
                "Based on the response from the router task extract information for the question {question} with the help of the respective tool."
                "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
                "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
            ),
            expected_output=(
                "You should analyse the output of the 'router_task'"
                "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
                "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
                "Return a claer and consise text as response."
            ),
            agent=Retriever_Agent,
            context=[self.router_task],
        )
        return Retriever_Agent

    def _create_grader_agent(self) -> Agent:
        Grader_agent = Agent(
            role='Answer Grader',
            goal='Filter out erroneous retrievals',
            backstory=(
                "You are a grader assessing relevance of a retrieved document to a user question."
                "If the document contains keywords related to the user question, grade it as relevant."
                "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.grader_task = Task(
            description=(
                "Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
            ),
            expected_output=(
                "Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
                "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
                "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
                "Do not provide any preamble or explanations except for 'yes' or 'no'."
            ),
            agent=Grader_agent,
            context=[self.retriever_task],
        )
        return Grader_agent

    def _create_hallucination_grader(self) -> Agent:
        hallucination_grader = Agent(
            role="Hallucination Grader",
            goal="Filter out hallucination",
            backstory=(
                "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
                "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.hallucination_task = Task(
            description=(
                "Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."
            ),
            expected_output=(
                "Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
                "Respond 'yes' if the answer is in useful and contains fact about the question asked."
                "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
                "Do not provide any preamble or explanations except for 'yes' or 'no'."
            ),
            agent=hallucination_grader,
            context=[self.grader_task],
        )
        return hallucination_grader

    def _create_answer_grader(self) -> Agent:
        answer_grader = Agent(
            role="Answer Grader",
            goal="Filter out hallucination from the answer.",
            backstory=(
                "You are a grader assessing whether an answer is useful to resolve a question."
                "Make sure you meticulously review the answer and check if it makes sense for the question asked"
                "If the answer is relevant generate a clear and concise response."
                "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.answer_task = Task(
            description=(
                "Based on the response from the hallucination task for the quetion {question} evaluate whether the answer is useful to resolve the question."
                "If the answer is 'yes' return a clear and concise answer."
                "If the answer is 'no' then perform a 'websearch' and return the response"
            ),
            expected_output=(
                "Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
                "Perform a web search using 'web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
                "Otherwise respond as 'Sorry! unable to find a valid response'."
            ),
            context=[self.hallucination_task],
            agent=answer_grader,
        )
        return answer_grader

    def run(self, question: str) -> str:
        inputs = {"question": question}
        result = self.rag_crew.kickoff(inputs=inputs)
        return result

if __name__ == "__main__":
    llm = ChatOllama(model= "mistral:instruct")
    pdf_path = r'D:\Company_Assignments\Attack Capital\NLP\Multi-Agent-Application\transformer.pdf'
    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "mistral:instruct",
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
    question = "what does transformer do?"
    result = rag_agent.run(question)
    print(result)