from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Load API Key
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.2-1b-preview",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Wikipedia Search Tool
wikipedia_tool = Tool(
    name="wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
    description="Search for information on Wikipedia."
)

# Define the question generation function
def generate_questions(paragraph: str, user_request: str) -> str:
    """Generate structured questions based on the given paragraph and request."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an AI that generates structured questions based on a given paragraph."),
        ("user", "Paragraph: {paragraph}\nUser Request: {user_request}")
    ])
    chain = prompt_template | llm
    result = chain.invoke({"paragraph": paragraph, "user_request": user_request})
    return result.content

# Question Generator Tool
question_generator_tool = Tool(
    name="question_generator",
    func=lambda query: generate_questions(query["paragraph"], query["user_request"]),
    description="Generate questions based on the provided text."
)

# Define Tools
tools = [wikipedia_tool, question_generator_tool]

# Create ReAct Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use ReAct framework
    verbose=True,
    handle_parsing_errors=True) # Set to True to see the agent's thought process


# Example Input
paragraph = """Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy."""
user_request = "Generate three multiple-choice questions with four answer options each."

formatted_input = {
    "paragraph": paragraph,
    "user_request": user_request
}

# Run the Agent
response = agent.invoke({"input": formatted_input})

# Print Output
print("\nGenerated Questions:\n", response)