import os
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from pydantic import BaseModel

# Load API Key
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.2-1b-preview",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Pull the ReAct prompt from the hub
prompt = hub.pull("hwchase17/react")

# Define Pydantic model for structured input
class QuestionGeneratorInput(BaseModel):
    paragraph: str
    user_request: str

# Define the question generation function
def generate_questions(paragraph: str, user_request: str) -> str:
    """Generate multiple-choice questions from a given paragraph."""
    
    print(f"Generating questions for:\nParagraph: {paragraph}\nUser Request: {user_request}")  # Debug print

    # Create input text
    input_text = f"Paragraph: {paragraph}\nUser Request: {user_request}"

    # Debug before invoking the chain
    print(f"[DEBUG] Input Text to LLM:\n{input_text}")

    # Use the chain
    chain = PromptTemplate.from_template(
        "Generate multiple-choice questions from the following text:\n{input}"
    ) | llm
    
    # Fetch response
    result = chain.invoke({"input": input_text})
    
    print(f"[DEBUG] Raw LLM Response: {result}")  # Debug print

    return result.content if hasattr(result, "content") else str(result)

# Define the tool for the agent
question_generator_tool = Tool(
    name="question_generator",
    func=generate_questions,  # Direct function call
    description="Generate multiple-choice questions based on a given text input.",
    args_schema=QuestionGeneratorInput  # Ensure structured input
)

# Define tools list
tools = [question_generator_tool]

# Construct the ReAct agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Use ReAct framework
    verbose=True
)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Example Input (Fixed Formatting)
formatted_input = {"paragraph": """Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy.""",
                   "user_request": "Generate three multiple-choice questions with four answer options each."}

# Debugging Agent Execution
print("\n[DEBUG] Running Agent Executor...")
response = agent_executor.invoke(formatted_input)

# Print Output
print("\n[DEBUG] Generated Questions:\n", response)
