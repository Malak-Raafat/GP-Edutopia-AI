from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated, Sequence, List, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
import os
import json
import sqlite3
from datetime import datetime

# Suppress HuggingFace tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

GROQ_API_KEY = "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"

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

def parse_conversation_data(user_str: str, ai_str: str) -> ConversationSummaryBufferMemory:
    try:
        # Initialize memory
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            return_messages=True
        )
        
        # Split the strings into lists
        user_messages = [msg.strip() for msg in user_str.split(',')]
        ai_messages = [msg.strip() for msg in ai_str.split(',')]
        
        # Ensure both lists have the same length
        if len(user_messages) != len(ai_messages):
            print("Warning: Number of user messages does not match number of AI messages")
            min_length = min(len(user_messages), len(ai_messages))
            user_messages = user_messages[:min_length]
            ai_messages = ai_messages[:min_length]
        
        # Save each conversation pair to memory
        for user_msg, ai_msg in zip(user_messages, ai_messages):
            if user_msg and ai_msg:  # Only save non-empty messages
                memory.save_context(
                    {"input": user_msg},
                    {"output": ai_msg}
                )
        
        return memory
        
    except Exception as e:
        print(f"Error parsing conversation data: {str(e)}")
        # Return fresh memory if parsing fails
        return ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            return_messages=True
        )

def load_context(file_path: str, user_str: str = None, ai_str: str = None) -> tuple:
    try:
        print(f"\nLoading context from: {file_path}")
        
        # Load and process documents
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.create_documents([content])
        
        print(f"Created {len(final_documents)} document chunks")
        
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
        
        print("Context loaded and processed successfully!")
        
        # Initialize memory with conversation history if provided
        if user_str and ai_str:
            print("Loading conversation history into memory...")
            memory = parse_conversation_data(user_str, ai_str)
        else:
            print("Initializing fresh memory...")
            memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=2000,
                return_messages=True
            )
        
        return vectors, retrieval_chain, memory
        
    except Exception as e:
        print(f"Error loading context: {str(e)}")
        raise

# Load initial context and memory
file_path = "/Users/habibaalaa/Downloads/🎓🫣/Simple Agent/text.txt"
vectors, retrieval_chain, memory = load_context(file_path)

# Define the state with memory
class AgentState(TypedDict):
    input: str
    current_step: str
    final_answer: str
    questions: str
    tool_used: str
    thoughts: str
    memory: ConversationSummaryBufferMemory

# Define the RAG Tool
class RAGTool(BaseTool):
    name: str = "RAG Tool"
    description: str = "Use this tool to retrieve information from the knowledge base or generate an answer using built-in knowledge."
    memory: ConversationSummaryBufferMemory = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = None

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
            
            # If RAG doesn't know, fall back to LLM silently
            if "I don't know based on the provided context" in rag_answer:
                # Get conversation history from memory if available
                conversation_history = []
                if self.memory:
                    memory_variables = self.memory.load_memory_variables({})
                    conversation_history = memory_variables.get("history", [])
                
                # Create a prompt for the LLM that includes conversation history
                prompt = PromptTemplate(
                    input_variables=["conversation_history", "query"],
                    template=(
                        "Previous conversation:\n{conversation_history}\n\n"
                        "Current question: {query}\n\n"
                        "Please provide a brief and concise answer that takes into account the previous conversation. "
                        "Requirements:\n"
                        "1. Keep the answer short and to the point\n"
                        "2. Focus on the most important information\n"
                        "3. Use bullet points if appropriate\n"
                        "4. Maximum 3-4 sentences\n"
                        "5. Reference previous conversation when relevant"
                    )
                )
                final_prompt = prompt.format(
                    conversation_history="\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history]) if conversation_history else "No previous conversation",
                    query=input
                )
                llm_response = llm.invoke(final_prompt)
                return llm_response.content
            
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
        "Generates 5 structured questions based on the original user-provided text and question requirements. "
        "When using this tool, the action input must be in the following format: \n\n"
        "    <original_text> ### <question_requirements>\n\n"
        "The tool can generate different types of questions (MCQ, Y/N, T/F, WH) based on the requirements."
    )

    def _run(self, input: str) -> str:
        try:
            print("\nStarting question generation...")
            if "###" in input:
                paragraph_text, user_request = input.split("###", 1)
                print(f"Split input into text and request")
            else:
                paragraph_text = input
                user_request = "Generate 5 questions about the topic"
                print("Using default question request")
            
            # Determine question type from user request
            question_type = "mcq"  # default
            if "mcq" in user_request.lower():
                question_type = "mcq"
            elif "y/n" in user_request.lower() or "yes/no" in user_request.lower():
                question_type = "yes_no"
            elif "t/f" in user_request.lower() or "true/false" in user_request.lower():
                question_type = "true_false"
            elif "wh" in user_request.lower():
                question_type = "wh"
            
            print(f"Generating {question_type} questions...")
            
            # Create appropriate template based on question type
            if question_type == "mcq":
                template = (
                    "{{\n"
                    '  "questions": [\n'
                    '    {{"question": "Question 1?",\n'
                    '      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],\n'
                    '      "answer": "B) Option 2"}},\n'
                    '    // 4 more similar MCQ questions\n'
                    "  ]\n"
                    "}}"
                )
            else:
                template = (
                    "{{\n"
                    '  "questions": [\n'
                    '    {{"question": "Question 1?", "answer": "Answer 1"}},\n'
                    '    // 4 more similar questions\n'
                    "  ]\n"
                    "}}"
                )
            
            prompt = PromptTemplate(
                input_variables=["paragraph_text", "user_request", "template"],
                template=(
                    "You are a question generator that creates structured questions based on educational text.\n\n"
                    "Based on this text:\n\n"
                    "{paragraph_text}\n\n"
                    "Generate questions as per this request:\n"
                    "{user_request}\n\n"
                    "IMPORTANT: Return ONLY a valid JSON object with this EXACT structure:\n"
                    "{template}\n\n"
                    "REQUIREMENTS:\n"
                    "1. Return ONLY the JSON object, no other text\n"
                    "2. Ensure all JSON keys and values are in double quotes\n"
                    "3. Generate EXACTLY 5 questions\n"
                    "4. Follow the example format exactly\n"
                    "5. Base all questions on the provided text only\n"
                    "6. For MCQ questions, provide 4 options (A, B, C, D) and mark the correct answer\n"
                )
            )

            print("Formatting prompt...")
            final_prompt = prompt.format(
                paragraph_text=paragraph_text.strip(),
                user_request=user_request.strip(),
                template=template
            )
            print("Sending prompt to LLM...")
            response = llm.invoke(final_prompt)
            print("Received response from LLM")
            
            # Validate JSON
            try:
                questions_data = json.loads(response.content)
                # Format the output nicely
                formatted_output = f"Here are the generated {question_type.upper()} questions:\n\n"
                
                if question_type == "mcq":
                    for i, q in enumerate(questions_data["questions"], 1):
                        formatted_output += f"{i}. {q['question']}\n"
                        for option in q["options"]:
                            formatted_output += f"   {option}\n"
                        formatted_output += f"   Answer: {q['answer']}\n\n"
                else:
                    for i, q in enumerate(questions_data["questions"], 1):
                        formatted_output += f"{i}. {q['question']}\n"
                        formatted_output += f"   Answer: {q['answer']}\n\n"
                
                return formatted_output
                
            except json.JSONDecodeError:
                print("Invalid JSON received. Attempting to fix...")
                import re
                json_match = re.search(r'({[\s\S]*})', response.content)
                if json_match:
                    json_str = json_match.group(1)
                    json.loads(json_str)  # Validate
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
            # The questions are already formatted by the QuestionGenerator
            state["final_answer"] = state["questions"]
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
        
        # Initialize the state with existing memory
        initial_state = {
            "input": query,
            "current_step": "start",
            "final_answer": "",
            "questions": "",
            "tool_used": "",
            "thoughts": "Starting to process the query...",
            "memory": memory
        }
        
        # Set memory in RAG tool
        rag_tool.memory = memory
        
        print("Running workflow...")
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Save to memory
        final_state["memory"].save_context(
            {"input": query},
            {"output": final_state["final_answer"]}
        )
        
        print("Workflow completed successfully!")
        
        # Return both thoughts and final answer
        return f"Agent's Thoughts:\n{final_state['thoughts']}\n\nFinal Answer:\n{final_state['final_answer']}"
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        return f"Error processing query: {str(e)}"

# Example usage
try:
    print("\nStarting agent execution...")
    
    # Test Case 1: Creating new memory
    print("\n=== Test Case 1: Creating New Memory ===")
    print("Initializing a new conversation...")
    
    # Test conversation with new memory
    initial_queries = [
        "what is seaborn? give me a short answer",
        "how to make bar plot? keep it brief",
        "how to change colors? short answer please"
    ]
    
    # Initialize conversation strings
    user_str = ""
    ai_str = ""
    
    print("\nProcessing initial queries to create memory:")
    for i, query in enumerate(initial_queries, 1):
        print(f"\nQuery {i}: {query}")
        response = process_query(query)
        if response:
            print("\nAgent's Process:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Update conversation strings with the new interaction
            user_str += query + ","
            ai_str += response.split("Final Answer:\n")[1].strip() + ","
            
            # Print current memory state
            print("\nCurrent Memory State:")
            print("-" * 50)
            print(memory.chat_memory.messages)
            print("-" * 50)
        else:
            print("No response received due to error")
    
    # Test Case 2: Loading existing memory and adding new conversations
    print("\n=== Test Case 2: Loading Existing Memory and Adding New Conversations ===")
    print("Loading previous conversation data...")
    
    # Reload context with conversation history
    vectors, retrieval_chain, memory = load_context(file_path, user_str, ai_str)
    
    # Test conversation with loaded memory
    follow_up_queries = [
        "can you explain more about bar plots?",
        "generate 5 y/n questions about seaborn"
    ]
    
    print("\nProcessing follow-up queries with loaded memory:")
    for i, query in enumerate(follow_up_queries, 1):
        print(f"\nQuery {i}: {query}")
        response = process_query(query)
        if response:
            print("\nAgent's Process:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Update conversation strings with the new interaction
            user_str += query + ","
            ai_str += response.split("Final Answer:\n")[1].strip() + ","
            
            # Print current memory state
            print("\nCurrent Memory State:")
            print("-" * 50)
            print(memory.chat_memory.messages)
            print("-" * 50)
        else:
            print("No response received due to error")
    
    # Test Case 3: Verify memory persistence
    print("\n=== Test Case 3: Verifying Memory Persistence ===")
    print("Testing memory with a new query that references previous conversations...")
    
    # Reload context with updated conversation history
    vectors, retrieval_chain, memory = load_context(file_path, user_str, ai_str)
    
    final_query = "based on our previous discussion, what's the best way to customize a seaborn plot?"
    print(f"\nFinal Query: {final_query}")
    response = process_query(final_query)
    
    if response:
        print("\nAgent's Process:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Print final memory state
        print("\nFinal Memory State:")
        print("-" * 50)
        print(memory.chat_memory.messages)
        print("-" * 50)
    else:
        print("No response received due to error")
    
    print("\nAll tests completed successfully!")
except Exception as e:
    print(f"\nMain execution error: {str(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")