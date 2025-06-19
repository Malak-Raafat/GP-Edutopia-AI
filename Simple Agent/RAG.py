from flask import Flask, request, jsonify
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma

# Load the Groq API key
GROQ_API_KEY = "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"

# Flask app initialization
app = Flask(__name__)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Load documents from text file
file_path = "/Users/habibaalaa/Downloads/Simple Agent/quantum.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file does not exist: {file_path}")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

#print(content)
# Split content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.create_documents([content])

# Initialize Chroma DB
vectors = Chroma.from_documents(final_documents, embeddings)

# Initialize LLM and prompt
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.2-90b-vision-preview", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the question **ONLY** using the provided context. 
    If the context does not contain relevant information, respond with:  
    "I don't know based on the provided context."  
      
    Context:  
    {context}  
    
    Question: {input}  
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle user prompts.
    Expects a JSON payload with the 'prompt' field.
    """
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Invalid request, "prompt" field is required'}), 400

    user_prompt = data['prompt']
    start = time.process_time()

    try:
        # Retrieve documents with similarity scores
        docs_with_scores = vectors.similarity_search_with_score(user_prompt, k=5)

        # Filter documents based on similarity threshold (0.7)
        relevant_docs = [
            doc for doc, score in docs_with_scores if score > 0.7
        ]

        # Debugging: Print retrieved documents
        print("\nRetrieved Documents:")
        for doc, score in docs_with_scores:
            print(f"Score: {score:.2f} - Content: {doc.page_content[:200]}...")

        # If no relevant documents, return default response
        if not relevant_docs:
            elapsed_time = time.process_time() - start
            return jsonify({
                'answer': "I don't know based on the provided context.",
                'response_time': elapsed_time,
                'similar_documents': []
            }), 200

        # Invoke the retrieval chain
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed_time = time.process_time() - start

        # Format the response
        result = {
            'answer': response['answer'],
            'response_time': elapsed_time,
            'similar_documents': [
                {'content': doc.page_content, 'similarity_score': float(score)}
                for doc, score in docs_with_scores if score > 0.7
            ]
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)