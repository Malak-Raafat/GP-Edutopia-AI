from flask import Flask, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma  # Changed to Chroma
import time

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Flask app initialization
app = Flask(__name__)

# Initialize resources (global state)
embeddings = HuggingFaceEmbeddings()

# Load documents from a text file
file_path = "D:\RAG GP\Model and Animate a Synthesizer in Blender – Full Tutorial_youtube_transcription.txt"  # Replace with the actual path to your text file
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file does not exist: {file_path}")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.create_documents([content])

# Initialize Chroma DB
vectors = Chroma.from_documents(final_documents, embeddings)  # Changed to Chroma

# Initialize LLM and chains
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-90b-vision-preview", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()  # Using Chroma retriever
retrieval_chain = create_retrieval_chain(retriever, document_chain)


@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle user prompts.
    Expects a JSON payload with the 'prompt' field.
    """
    # Parse the JSON input
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Invalid request, "prompt" field is required'}), 400

    user_prompt = data['prompt']
    start = time.process_time()

    try:
        # Use the Chroma vector store directly to get documents and scores
        docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)  # Retrieve top 5 documents
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed_time = time.process_time() - start

        # Format the response
        result = {
            'answer': response['answer'],
            'response_time': elapsed_time,
            'similar_documents': [
                {
                    'content': doc.page_content,
                    'similarity_score': float(score)  
                } for doc, score in docs_with_scores
            ]
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
