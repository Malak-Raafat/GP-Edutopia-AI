from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated, Sequence
import requests
import json

GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Define the state
class AgentState(TypedDict):
    input: str
    chat_history: Sequence[str]
    current_step: str
    final_answer: str
    questions: str
    tool_used: str
    thoughts: str

# Define the RAG Tool
class RAGTool(BaseTool):
    name: str = "RAG Tool"
    description: str = "Use this tool to retrieve information from the RAG system or generate an answer using built-in knowledge."

    def _run(self, input: str) -> str:
        try:
            print(f"\nAttempting to connect to RAG server at http://localhost:5000/ask")
            url = "http://localhost:5000/ask"
            payload = {"prompt": input}
            print(f"Sending request with payload: {payload}")
            response = requests.post(url, json=payload)
            print(f"Response status code: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print(f"Received response: {result}")
            return result.get("answer", "No answer found.")
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            print("\nCould not access RAG server. Falling back to LLM knowledge...")
            # Create a prompt for the LLM to answer using its knowledge
            prompt = PromptTemplate(
                input_variables=["query"],
                template=(
                    "You are a knowledgeable assistant. Please provide a detailed and accurate answer to this query:\n\n"
                    "{query}\n\n"
                    "Requirements:\n"
                    "1. Provide accurate historical information\n"
                    "2. Include dates and facts where relevant\n"
                    "3. Structure the response in clear paragraphs\n"
                    "4. Focus on providing factual information\n"
                    "Note: This answer is being generated from built-in knowledge as the RAG system is unavailable."
                )
            )
            final_prompt = prompt.format(query=input)
            response = llm.invoke(final_prompt)
            return f"[Using built-in knowledge] {response.content}"
        except Exception as e:
            error_msg = f"Error in both RAG and fallback: {str(e)}"
            print(f"\nError: {error_msg}")
            return error_msg

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

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
        "- 5 Y/N questions (each including an answer),\n"
        "- 5 True/False questions (without answers),\n"
        "- 5 WH questions (without answers), and\n"
        "- 5 MCQs (each with 4 options and the correct answer).\n\n"
        "DO NOT use any summarized or modified version of the input. Use the original text exactly as provided."
    )

    def _run(self, input: str) -> str:
        try:
            print("\nStarting question generation...")
            if "###" in input:
                paragraph_text, user_request = input.split("###", 1)
                print(f"Split input into text and request")
            else:
                paragraph_text = input
                user_request = ("Generate 5 y/n questions with answers and 5 t/f without answers "
                                "and 5 wh questions without answers and 5 mcq with answers")
                print("Using default question request")
            
            print("Creating prompt template...")
            prompt = PromptTemplate(
                input_variables=["paragraph_text", "user_request"],
                template=(
                    "You are a question generator that creates structured questions based on educational text.\n\n"
                    "Based on this text:\n\n"
                    "{paragraph_text}\n\n"
                    "Generate questions as per this request:\n"
                    "{user_request}\n\n"
                    "IMPORTANT: Return ONLY a valid JSON object with this EXACT structure:\n"
                    "{{\n"
                    '  "yes_no": [\n'
                    '    {{"question": "Is pasta originally from Sicily?", "answer": "Yes"}},\n'
                    '    // 4 more yes/no questions\n'
                    "  ],\n"
                    '  "true_false": [\n'
                    '    {{"question": "Pasta was first recorded in the 15th century."}},\n'
                    '    // 4 more true/false questions\n'
                    "  ],\n"
                    '  "wh_questions": [\n'
                    '    {{"question": "When was pasta first recorded in Sicily?"}},\n'
                    '    // 4 more wh questions\n'
                    "  ],\n"
                    '  "mcq": [\n'
                    '    {{"question": "What is the origin of the word pasta?",\n'
                    '      "options": [\n'
                    '        "A) From Greek pastros",\n'
                    '        "B) From Italian for dough",\n'
                    '        "C) From Latin pasta",\n'
                    '        "D) From Arabic pastah"\n'
                    '      ],\n'
                    '      "answer": "B) From Italian for dough"}},\n'
                    '    // 4 more mcq questions\n'
                    "  ]\n"
                    "}}\n\n"
                    "REQUIREMENTS:\n"
                    "1. Return ONLY the JSON object, no other text\n"
                    "2. Ensure all JSON keys and values are in double quotes\n"
                    "3. Generate EXACTLY 5 questions of each type\n"
                    "4. Follow the example format exactly\n"
                    "5. Base all questions on the provided text only\n"
                )
            )

            print("Formatting prompt...")
            final_prompt = prompt.format(paragraph_text=paragraph_text.strip(), user_request=user_request.strip())
            print("Sending prompt to LLM...")
            response = llm.invoke(final_prompt)
            print("Received response from LLM")
            
            # Validate JSON
            try:
                json.loads(response.content)
                return response.content
            except json.JSONDecodeError:
                print("Invalid JSON received. Attempting to fix...")
                # Try to extract JSON from the response if it contains other text
                import re
                json_match = re.search(r'({[\s\S]*})', response.content)
                if json_match:
                    json_str = json_match.group(1)
                    # Validate the extracted JSON
                    json.loads(json_str)
                    return json_str
                raise ValueError("Could not extract valid JSON from response")
                
        except Exception as e:
            error_msg = f"Error in question generation: {str(e)}"
            print(f"\nError: {error_msg}")
            return error_msg

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

# Initialize tools
rag_tool = RAGTool()
question_tool = QuestionGenerator()
tools = [rag_tool, question_tool]

def plan_action(state: AgentState) -> AgentState:
    """Plan the next action based on the input"""
    query = state["input"].lower()
    if "generate" in query and "questions" in query:
        state["thoughts"] = "I need to generate questions about the history of pasta. I'll use the Question Generator tool to create structured questions."
        state["current_step"] = "planned_questions"
    else:
        state["thoughts"] = "This is a general query about pasta history. I'll use the RAG tool to get a comprehensive answer."
        state["current_step"] = "planned_rag"
    return state

def router(state: AgentState) -> Dict[str, str]:
    """Route to the appropriate node based on the query"""
    query = state["input"].lower()
    if "generate" in query and "questions" in query:
        return {"next": "generate_questions"}
    else:
        return {"next": "rag_query"}

def rag_query(state: AgentState) -> AgentState:
    """Use the RAG tool to answer the query"""
    try:
        state["thoughts"] += "\n\nExecuting RAG query to get information about pasta history..."
        answer = rag_tool.run(state["input"])
        state["thoughts"] += f"\n\nReceived answer from RAG system. Processing response..."
        state["final_answer"] = answer
        state["current_step"] = "completed"
        state["tool_used"] = "rag"
        return state
    except Exception as e:
        state["thoughts"] += f"\n\nError occurred while querying RAG system: {str(e)}"
        state["final_answer"] = f"Error getting answer: {str(e)}"
        state["current_step"] = "error"
        state["tool_used"] = "rag"
        return state

def generate_questions(state: AgentState) -> AgentState:
    """Generate questions using the Question Generator tool"""
    try:
        state["thoughts"] += "\n\nPreparing to generate questions..."
        # Extract the topic from the query
        query = state["input"].lower()
        topic = query.replace("generate", "").replace("questions", "").replace("about", "").replace("the history of", "").strip()
        
        # Create a prompt for the LLM to get information about the topic
        info_prompt = PromptTemplate(
            input_variables=["topic"],
            template=(
                "Provide a detailed historical overview of {topic}. Include key dates, facts, and developments. "
                "Focus on major milestones and important events. Make it comprehensive but concise."
            )
        )
        
        state["thoughts"] += f"\n\nGetting information about {topic}..."
        topic_info = llm.invoke(info_prompt.format(topic=topic))
        
        state["thoughts"] += "\n\nUsing Question Generator tool to create structured questions..."
        questions = question_tool.run(f"{topic_info.content} ### Generate 5 questions about the history of {topic}")
        state["thoughts"] += "\n\nQuestions generated successfully. Moving to formatting step..."
        state["questions"] = questions
        state["current_step"] = "questions_generated"
        state["tool_used"] = "questions"
        return state
    except Exception as e:
        state["thoughts"] += f"\n\nError occurred while generating questions: {str(e)}"
        state["final_answer"] = f"Error generating questions: {str(e)}"
        state["current_step"] = "error"
        state["tool_used"] = "questions"
        return state

def format_final_answer(state: AgentState) -> AgentState:
    """Format the final answer with the generated questions"""
    if state["tool_used"] == "questions":
        try:
            state["thoughts"] += "\n\nFormatting the generated questions into a readable structure..."
            questions_data = json.loads(state["questions"])
            formatted_questions = "Here are the generated questions:\n\n"
            
            # Format Yes/No questions
            formatted_questions += "Yes/No Questions:\n"
            for q in questions_data.get("yes_no", []):
                formatted_questions += f"- {q['question']} (Answer: {q['answer']})\n"
            
            # Format True/False questions
            formatted_questions += "\nTrue/False Questions:\n"
            for q in questions_data.get("true_false", []):
                formatted_questions += f"- {q['question']}\n"
            
            # Format WH questions
            formatted_questions += "\nWH Questions:\n"
            for q in questions_data.get("wh_questions", []):
                formatted_questions += f"- {q['question']}\n"
            
            # Format MCQs
            formatted_questions += "\nMultiple Choice Questions:\n"
            for q in questions_data.get("mcq", []):
                formatted_questions += f"- {q['question']}\n"
                for option in q["options"]:
                    formatted_questions += f"  {option}\n"
                formatted_questions += f"  Answer: {q['answer']}\n"
            
            state["thoughts"] += "\n\nQuestions formatted successfully. Preparing final output..."
            state["final_answer"] = formatted_questions
            state["current_step"] = "completed"
            return state
        except Exception as e:
            state["thoughts"] += f"\n\nError occurred while formatting questions: {str(e)}"
            state["final_answer"] = f"Error formatting questions: {str(e)}"
            state["current_step"] = "error"
            return state
    else:
        return state

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("plan", plan_action)  # Add planning node first
workflow.add_node("router", router)
workflow.add_node("rag_query", rag_query)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("format_answer", format_final_answer)

# Add edges
workflow.add_edge("plan", "router")  # Connect plan to router
workflow.add_edge("rag_query", END)
workflow.add_edge("generate_questions", "format_answer")
workflow.add_edge("format_answer", END)

# Add conditional edges
def should_generate_questions(state: AgentState) -> bool:
    """Determine if we should generate questions"""
    query = state["input"].lower()
    return "generate" in query and "questions" in query

workflow.add_conditional_edges(
    "router",
    should_generate_questions,
    {
        True: "generate_questions",
        False: "rag_query"
    }
)

# Set entry point
workflow.set_entry_point("plan")

# Compile the graph
app = workflow.compile()

def process_query(query: str):
    """Process a user query and return the response"""
    try:
        print(f"\nProcessing query: '{query}'")
        print("Initializing agent state...")
        
        # Initialize the state
        initial_state = {
            "input": query,
            "chat_history": [],
            "current_step": "start",
            "final_answer": "",
            "questions": "",
            "tool_used": "",
            "thoughts": "Starting to process the query..."
        }
        
        print("Running workflow...")
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        print("Workflow completed successfully!")
        
        # Return both thoughts and final answer
        return f"Agent's Thoughts:\n{final_state['thoughts']}\n\nFinal Answer:\n{final_state['final_answer']}"
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        return f"Error processing query: {str(e)}"

# Example usage
try:
    print("\nStarting agent execution...")
    
    # Test with a question generation query
    query = "generate 5 questions about the history of game design"  # Simplified query
    print("\nTesting question generation:")
    response = process_query(query)
    if response:
        print("\nAgent's Process:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    else:
        print("No response received due to error")

    # Test with a RAG query
    query = "explain for me the concept of quantum computing"
    print("\nTesting RAG query:")
    response = process_query(query)
    if response:
        print("\nAgent's Process:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    else:
        print("No response received due to error")
        
    print("\nAll tests completed successfully!")
except Exception as e:
    print(f"\nMain execution error: {str(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")