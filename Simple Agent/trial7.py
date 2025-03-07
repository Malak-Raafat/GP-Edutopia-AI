import os
import requests
from typing import Optional
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, Tool
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.prompts import PromptTemplate

# Load API Key securely from environment variables
GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Define the RAG tool function
def query_rag_tool(user_query: str) -> str:
    """
    Queries the RAG Flask API with a user question and returns the response.
    """
    url = "http://127.0.0.1:5000/ask"  # Ensure this is the correct Flask API URL
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": user_query}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("answer", "No relevant answer found.")
    except requests.exceptions.RequestException as e:
        return f"Error querying RAG API: {str(e)}"

# Wrap the function as a LangChain Tool
rag_tool = Tool(
    name="RAG_Tool",
    func=query_rag_tool,
    description="Use this tool to retrieve relevant information from the RAG system."
)

class QuestionGenerator(BaseTool):
    name: str = "Question Generator"
    description: str = (
        "Generates structured questions based on a retrieved context."
    )

    def _run(self, input: str) -> str:
        if "###" in input:
            retrieved_context, question_type = input.split("###", 1)
        else:
            return "Invalid format. Expected: <retrieved_context> ### <question_type>"

        prompt = PromptTemplate(
            input_variables=["retrieved_context", "question_type"],
            template=(
                "Based on the following context:\n\n"
                "{retrieved_context}\n\n"
                "Generate questions of type {question_type} and return them with their correct answers."
            )
        )

        final_prompt = prompt.format(
            retrieved_context=retrieved_context.strip(), 
            question_type=question_type.strip()
        )
        
        response = llm.invoke(final_prompt)

        if response:
            return f"Generated Questions:\n{response.content.strip()}"
        else:
            return "Error: No questions generated."

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

question_tool = QuestionGenerator()

# List of tools
tools = [rag_tool, question_tool]
tools_names=["rag_tool","question_tool"]

# Modified prompt to ensure the query runs only once
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

Once RAG retrieves the information:
THOUGHT: Now, let's generate questions using the retrieved context.
ACTION: Question Generator
ACTION_INPUT: <retrieved_context> ### <question_type>

Stop after generating the requested questions. Do not require any further user input.
{agent_scratchpad}
"""
)

# Construct the agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

# Read the file content
file_path = "/Users/habibaalaa/Downloads/Simple Agent/quantum.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file does not exist: {file_path}")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Example query
user_query = "Generate 10 T/F questions about trump"

# Invoke the agent
response = agent_executor.invoke({"input": user_query})

# Print the response
print(response)
