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

# Load API Key securely
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize LLM
llm = ChatGroq(
    model_name="deepseek-r1-distill-qwen-32b",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Wikipedia Tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Parallelogram Area Calculator
class AreaCalculator(BaseTool):
    name: str = "Parallelogram Area Calculator"
    description: str = '''Use this tool when you need to calculate the area of a parallelogram.
    If the parallelogram is a rectangle then you will be given just width and height.
    If it is not a rectangle, then you will need the angle between the width and the height. Angle should be in degrees.
    The input to the tool must be provided as width|height|angle or width|height.'''

    def _run(self, input: str) -> float:
        extract = input.split("|")
        width = float(extract[0])
        height = float(extract[1])
        angle: Optional[float] = float(extract[2]) if len(extract) == 3 else None
        return width * height * sin(radians(angle)) if angle else width * height

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

area_tool = AreaCalculator()

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

# Pull the ReAct prompt from the hub
prompt = hub.pull("hwchase17/react")

# Add tools to the list
tools = [wikipedia_tool, area_tool, question_tool]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

# Original text and explicit question requirements
paragraph_text = """First: Plant Classification
Data Preparation
1. Setting Constants
image_size: Defines the target size to which images will be resized (224x224 pixels).
batch_size: The number of images to process at once during training. (64 batches)
2. Loading Data
This function loads images and their corresponding labels from the specified directory.
3. Normalizing Images
The images are normalized by dividing each pixel's value by 255.0, converting the pixel values from
the range [0, 255] to [0, 1]. This step helps improve the convergence during model training.
4. Splitting the Data into Train and Validation Sets
• This splits the loaded data into training and validation sets, with 80% of the data used for
training and 20% for validation
7. Label Encoding
• The LabelEncoder is used to convert text labels into integer labels
8. Class Weight Calculation
• The class weights are computed to handle class imbalance. The compute_class_weight
function calculates weights inversely proportional to class frequencies.
10. Data Augmentation
We apply augmentation only in MobileNet, VGG, AlexNet, as we don't apply augmentation in
ViT. The train_datagen applies several data augmentation techniques to the training images,
including: Random rotation (rotation_range), width/height shifting, shearing, zooming, and flipping
the images horizontally. val_datagen doesn't apply any augmentation
These augmentations help improve model generalization by providing more varied input data.
11. Data Generators
train_generator and validation_generator and testing_generator are instances of
ImageDataGenerator that yield batches of images and labels during training and validation,
respectively
Vision Transformer (ViT)
Architecture
• Divides input images into fixed-size non-overlapping patches.
• Converts patches into 1D vector embeddings via a linear projection.
• Adds positional embeddings to preserve spatial relationships.
• Processes embeddings using Transformer encoder layers (self-attention + feedforward networks).
• Uses a learnable [CLS] token for classification.
• Final output is passed to a classification head for tasks like image classification.
MobileNet
MobileNet is a lightweight deep learning model designed for mobile and embedded devices,
using depthwise separable convolutions to reduce the number of parameters.
VGG16
VGG is a deep convolutional neural network known for its simplicity and uniform architecture.
"""

user_request = "Generate 5 y/n questions with answers and 5 t/f without answers and 5 wh questions without answers and 5 mcq with answers"

# Read the file content
file_path = "/Users/habibaalaa/Downloads/Simple Agent/quantum.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file does not exist: {file_path}")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# IMPORTANT: Make sure to pass the original input in the expected format:
input_text = content + "###" + user_request

# Now, when you invoke the agent, the Question Generator tool will receive the original text as intended.
response = agent_executor.invoke({"input": input_text})
print(response)