from flask import Flask, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents.tools import tool
from langchain.memory import ConversationBufferMemory
#from dotenv import load_dotenv

# Load environment variables
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize Flask app
app = Flask(__name__)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
file_path = "/Users/habibaalaa/Downloads/Simple Agent/text.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file does not exist: {file_path}")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.create_documents([content])

# Initialize ChromaDB
vectorstore = Chroma.from_documents(final_documents, embeddings)
retriever = vectorstore.as_retriever()

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.2-90b-vision-preview", temperature=0)

# Define tools

def rag_tool(query: str):
    """Retrieve relevant documents based on query."""
    response = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in response])

def question_generator_tool(context: str):
    """Generate a question based on provided context."""
    prompt = ChatPromptTemplate.from_template("Generate a question based on the following context:\n{context}")
    return llm.invoke(prompt.format(context=context))

rag_tool_obj = tool(name="RAG Tool", func=rag_tool, description="Retrieves relevant information from documents.")
question_tool_obj = tool(name="Question Generator", func=question_generator_tool, description="Generates a question based on retrieved text.")

# Initialize Agent with tools
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools=[rag_tool_obj, question_tool_obj], llm=llm, agent="zero-shot-react-description", memory=memory)
agent_executor = AgentExecutor(agent=agent, tools=[rag_tool_obj, question_tool_obj], verbose=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Invalid request, "query" field is required'}), 400

    query = data['query']
    response = agent_executor.invoke({"input": query})
    return jsonify({'response': response["output"]})

if __name__ == '__main__':
    app.run(debug=True)
