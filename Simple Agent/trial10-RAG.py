from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from math import radians, sin
from langchain_groq import ChatGroq
from typing import Optional
from langchain import hub
from langchain_core.prompts import PromptTemplate
import os
import requests

# Load API Key securely
GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    model_name="deepseek-r1-distill-qwen-32b",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Question Generator Tool
class QuestionGenerator(BaseTool):
    name: str = "Question Generator"
    description: str = (
        "Generates structured questions based on the original user-provided text and question requirements. "
        "When using this tool, the action input must be in the following format: \n\n"
        "    <original_text> ### <question_requirements>\n\n"
        "For example, if the user provides a detailed educational text and specific requirements, "
        "the input should include the full original text, followed by '###', followed by the requirements. \n\n"
        "This tool will then generate:\n"
        "- 10 Y/N questions (each including an answer),\n"
        "- 100 True/False questions (without answers),\n"
        "- 60 WH questions (without answers), and\n"
        "- 4 MCQs (each with 4 options and the correct answer).\n\n"
        "DO NOT use any summarized or modified version of the input. Use the original text exactly as provided."
        "generate the user query only one time and then stop"
    )

    def _run(self, input: str) -> str:
        # Print the received input for debugging.
        print(f"Received input: {input}")

        # Check if the delimiter is present
        if "###" in input:
            paragraph_text, user_request = input.split("###", 1)
        else:
            # If no delimiter is found, assume the entire input is the original text
            # and use a default question requirement.
            paragraph_text = input
            user_request = ("Generate 10 y/n questions with answers and 100 t/f without answers "
                            "and 60 wh questions without answers and 4 mcq with answers")
        
        # Create a prompt to generate questions
        prompt = PromptTemplate(
            input_variables=["paragraph_text", "user_request"],
            template=(
                "Based on the following educational text:\n\n"
                "{paragraph_text}\n\n"
                "Strictly generate questions as per the following user request:\n"
                "{user_request}\n\n"
                "Ensure the following formatting:\n"
                "- **Y/N Questions (10 total)** → MUST include the correct answer (Yes or No)\n"
                "- **T/F Questions (100 total)** → MUST NOT include answers\n"
                "- **WH Questions (60 total)** → MUST NOT include answers\n"
                "- **MCQs (4 total)** → MUST include 4 answer choices, with one correct answer indicated.\n"
                "Return the questions in a structured JSON format like this:\n"
                "{{\n"
                "  'yes_no': [{{'question': '...', 'answer': 'Yes/No'}}, ...],\n"
                "  'true_false': [{{'question': '...'}}, ...],\n"
                "  'wh_questions': [{{'question': '...'}}, ...],\n"
                "  'mcq': [{{'question': '...', 'options': ['A) ...', 'B) ...', 'C) ...', 'D) ...'], 'answer': '...'}}]\n"
                "}}\n"
                "Generate exactly the required number of questions. No extra questions."
            )
        )

        final_prompt = prompt.format(paragraph_text=paragraph_text.strip(), user_request=user_request.strip())
        response = llm.invoke(final_prompt)
        return response.content  # Return the generated questions

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

question_tool = QuestionGenerator()

# RAG Tool
class RAGTool(BaseTool):
    name: str = "RAG Tool"
    description: str = (
        "Use this tool to retrieve information from the RAG system. "
        "This tool is useful when you need to answer questions based on a specific knowledge base. "
        "The input should be a clear and concise question."
    )

    def _run(self, input: str) -> str:
        """
        Sends a request to the RAG Flask app and retrieves the response.
        """
        try:
            # Replace with the URL of your Flask app
            url = "http://localhost:5000/ask"
            payload = {"prompt": input}
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            result = response.json()
            return result.get("answer", "No answer found.")
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

rag_tool = RAGTool()

# Pull the ReAct prompt from the hub
#prompt = hub.pull("hwchase17/react")

prompt = PromptTemplate.from_template(
    """
    You are an intelligent AI assistant that **must always use the RAG tool first** to retrieve relevant context.
    Never answer directly unless the tool fails.

    TOOLS:
    {tools}

    TOOL NAMES:
    {tool_names}

    User question: {input}

    THOUGHT: Let's use the RAG tool to find relevant information.
    ACTION: RAG_Tool
    ACTION_INPUT: {input}

    OBSERVATION: {rag_output}

    THOUGHT: Based on the retrieved context:
    - If the user asked for questions, I will generate questions using the Question Generator tool.
    - If the user asked about something else, I will provide the answer directly from the RAG tool's response.

    DECISION:
    {decision}

    THOUGHT: The task is complete. Stopping now.
    {agent_scratchpad}
    """
)

# Add tools to the list
tools = [question_tool, rag_tool]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

# Example query
query = "history of pasta"

# Invoke the agent
response = agent_executor.invoke({"input": query})
print(response)