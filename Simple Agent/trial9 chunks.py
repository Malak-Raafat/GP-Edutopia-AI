from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.agents import AgentExecutor, create_react_agent

# Load API Key securely
GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    model_name="deepseek-r1-distill-qwen-32b",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Updated Question Generator Tool
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

# Pull the ReAct prompt from the hub
prompt = hub.pull("hwchase17/react")

# Add tools to the list
tools = [question_tool]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

# Read the file content
file_path = "/Users/habibaalaa/Downloads/Simple Agent/quantum.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file does not exist: {file_path}")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Split the content into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunk size to avoid API limits
    chunk_overlap=100  # Add overlap to maintain context
)
chunks = text_splitter.split_text(content)

# Print chunks for debugging
print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")

# User request
user_request = "Generate 5 y/n questions with answers and 5 t/f without answers and 5 wh questions without answers and 5 mcq with answers"

# Process each chunk and collect results
all_questions = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}...")
    input_text = chunk + "###" + user_request
    try:
        response = agent_executor.invoke({"input": input_text})
        all_questions.append(response["output"])
        print(f"Chunk {i + 1} processed successfully.")
    except Exception as e:
        print(f"Error processing chunk {i + 1}: {e}")
        continue

# Combine the results
final_output = "\n\n".join(all_questions)
print("Final Output:\n", final_output)