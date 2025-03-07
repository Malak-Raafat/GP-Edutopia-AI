import os
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain import hub
from langchain_core.prompts import PromptTemplate

# Load API Key securely
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.2-1b-preview",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Define Pydantic model for input validation
class QuestionGeneratorInput(BaseModel):
    paragraph: str
    user_request: str

# Define the question generation function
def generate_questions(paragraph: str, user_request: str) -> str:
    """Generate multiple-choice questions from the given paragraph."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You generate multiple-choice questions based on a given paragraph."),
        ("user", "Paragraph: {paragraph}\nUser Request: {user_request}")
    ])
    
    chain = prompt_template | llm
    result = chain.invoke({"paragraph": paragraph, "user_request": user_request})

    # Debugging output
    print("\n🔍 DEBUG: LLM Output:", result)  

    if isinstance(result, str):
        return result
    elif hasattr(result, "content"):
        return result.content
    else:
        return str(result)

import json

# Modify function to accept a single string input
def generate_questions_single_input(query: str) -> str:
    """Wrapper function to handle single input as JSON string."""
    try:
        query_dict = json.loads(query)  # Parse JSON input
        paragraph = query_dict.get("paragraph", "")
        user_request = query_dict.get("user_request", "")
        return generate_questions(paragraph, user_request)
    except json.JSONDecodeError:
        return "Invalid input format. Expected a JSON string with 'paragraph' and 'user_request'."

# Update the tool definition
question_generator_tool = Tool(
    name="question_generator",
    func=generate_questions_single_input,  # Accepts single string input
    description="Generate multiple-choice questions based on a given text input. Input should be a JSON string with 'paragraph' and 'user_request'."
)

# Initialize Agent with Structured Tool Use
agent = initialize_agent(
    tools=[question_generator_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Alternative to OPENAI_FUNCTIONS
    verbose=True
)

# Example Input
paragraph = """Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy."""
user_request = "Generate three multiple-choice questions with four answer options each."

formatted_input = json.dumps({"paragraph": paragraph, "user_request": user_request})
response = agent.run(formatted_input)

# Print Output
print("\nGenerated Questions:\n", response)
