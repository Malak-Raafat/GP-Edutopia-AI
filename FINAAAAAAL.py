from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated, Sequence
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
import os
import json

GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Initialize embeddings and load documents
embeddings = HuggingFaceEmbeddings()
file_path = "/Users/habibaalaa/Downloads/Simple Agent/quantum.txt"

# Load and process documents
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.create_documents([content])

# Initialize Chroma DB
vectors = Chroma.from_documents(final_documents, embeddings)

# Create RAG chain
rag_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the question **ONLY** using the provided context. 
    If the context does not contain relevant information, respond with:  
    "I don't know based on the provided context."  
      
    Context:  
    {context}  
    
    Question: {input}  
    """
)

document_chain = create_stuff_documents_chain(llm, rag_prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

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
    description: str = "Use this tool to retrieve information from the knowledge base or generate an answer using built-in knowledge."

    def _run(self, input: str) -> str:
        try:
            print("\nQuerying knowledge base...")
            # Get relevant documents with scores
            docs_with_scores = vectors.similarity_search_with_score(input, k=5)
            
            # Filter documents based on similarity threshold
            relevant_docs = [doc for doc, score in docs_with_scores if score > 0.7]
            
            print(f"Found {len(relevant_docs)} relevant documents")
            
            # First try RAG response
            print("Generating RAG-based response...")
            response = retrieval_chain.invoke({"input": input})
            rag_answer = response['answer']
            
            # If RAG doesn't know, fall back to LLM
            if "I don't know based on the provided context" in rag_answer:
                print("No relevant information in RAG. Falling back to LLM knowledge...")
                # Create a prompt for the LLM to answer using its knowledge
                prompt = PromptTemplate(
                    input_variables=["query"],
                    template=(
                        "You are a knowledgeable assistant. Please provide a detailed and accurate answer to this query:\n\n"
                        "{query}\n\n"
                        "Requirements:\n"
                        "1. Provide accurate information\n"
                        "2. Include technical details where relevant\n"
                        "3. Structure the response in clear paragraphs\n"
                        "4. Focus on providing factual information"
                    )
                )
                final_prompt = prompt.format(query=input)
                llm_response = llm.invoke(final_prompt)
                return (
                    "RAG System Response: I don't know based on the provided context.\n\n"
                    "Falling back to built-in knowledge:\n"
                    "----------------------------------------\n"
                    f"{llm_response.content}"
                )
            
            return rag_answer
            
        except Exception as e:
            error_msg = f"Error in RAG processing: {str(e)}"
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
        # Extract topic for question generation
        topic = query.replace("generate", "").replace("questions", "").replace("about", "").replace("the history of", "").strip()
        state["thoughts"] = f"I need to generate questions about {topic}. I'll use the Question Generator tool to create structured questions."
        state["current_step"] = "planned_questions"
    else:
        # For RAG queries, use the original query
        state["thoughts"] = f"This is a query about '{state['input']}'. I'll use the RAG tool to get a comprehensive answer."
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
        state["thoughts"] += "\n\nExecuting RAG query to find relevant information..."
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
        
        # First try to get information from RAG
        state["thoughts"] += f"\n\nChecking RAG system for information about {topic}..."
        rag_query = f"provide detailed information about {topic}"
        rag_response = rag_tool.run(rag_query)
        
        # Check if RAG had relevant information
        if "I don't know based on the provided context" in rag_response:
            state["thoughts"] += f"\n\nNo relevant information found in RAG. Using LLM knowledge..."
            # Create a prompt for the LLM to get information about the topic
            info_prompt = PromptTemplate(
                input_variables=["topic"],
                template=(
                    "Provide a detailed overview of {topic}. Include key facts, concepts, and important information. "
                    "Focus on accuracy and comprehensiveness while being concise. "
                    "Include both historical and current information where relevant."
                )
            )
            topic_info = llm.invoke(info_prompt.format(topic=topic))
            base_text = topic_info.content
            state["thoughts"] += "\n\nGenerated base information using LLM knowledge."
        else:
            state["thoughts"] += "\n\nUsing information from RAG system."
            base_text = rag_response
        
        state["thoughts"] += "\n\nUsing Question Generator tool to create structured questions..."
        questions = question_tool.run(f"{base_text} ### Generate 5 questions about {topic}")
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
    query = "explain for me the rules of chess"
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