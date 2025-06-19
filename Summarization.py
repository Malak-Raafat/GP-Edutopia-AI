from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma  
import time

load_dotenv()

os.environ["HF_HOME"] = "C:/Users/Ahmed Raafat/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


groq_api_key = os.getenv("GROQ_API_KEY", "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")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")

app = Flask(__name__)


file_path = r"C:\Users\Malak Raafat\Downloads\SECURITY.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.create_documents([content])

# Vector DB
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)



summary_prompt = ChatPromptTemplate.from_template("""
You are an expert educational content summarizer with expertise across ALL academic disciplines. Your task is to create a comprehensive yet concise summary that captures ALL important information while removing redundancy.

Follow these guidelines:
1. Capture EVERY new term, concept, or definition introduced
2. Preserve ALL explanations of new concepts or ideas
3. Maintain ALL key relationships between concepts
4. Include ALL practical applications and examples
5. Remove ONLY redundant explanations and repeated information
6. Use clear, direct language without unnecessary elaboration
7. Adapt your summary format to the specific field of study

For ANY scientific content:
    - ALWAYS include ALL mathematical equations, formulas, and rules with their explanations
    - ALWAYS preserve the exact notation and symbols used in the original content
    - ALWAYS explain what each variable and symbol represents
    - ALWAYS include any important constants or parameters
    - ALWAYS explain the context and application of each equation

    For ANY humanities content:
    - Include literary analysis, themes, and symbolism
    - Capture historical context, timelines, and cause-and-effect relationships
    - Preserve philosophical arguments and ethical considerations
    - Note cultural significance and societal impact

    For ANY social sciences:
    - Include theoretical frameworks and methodologies
    - Capture statistical data and research findings
    - Highlight policy implications and real-world applications
    - Note cultural and societal contexts

    For ANY arts and languages:
    - Include techniques, styles, and movements
    - Capture grammar rules, vocabulary, and cultural context
    - Highlight artistic elements and creative processes
    - Note historical development and influence

    Format your response with these sections (include only those relevant to the content):
    - New Terms and Definitions (list ALL new terms with their explanations)
    - Core Concepts (include ALL key concepts with their explanations)
    - [Field-Specific Content] (e.g., Mathematical Content, Literary Analysis, Historical Context)
    - Relationships and Connections (how concepts relate to each other)
    - Applications and Examples (ALL practical uses and examples)

    Important: Do not omit any new concept, definition, or explanation. Only remove truly redundant content. Adapt your format to best represent the specific field of study. For ANY scientific content, ALWAYS include ALL equations and formulas with their explanations.
    <document>
    {context}
    </document>

""")

title_prompt = ChatPromptTemplate.from_template("""
Generate a clear, informative academic title using a maximum of 3 words:
<document>
{context}
</document>
""")

# Chains
summary_chain = create_stuff_documents_chain(llm, summary_prompt)
title_chain = create_stuff_documents_chain(llm, title_prompt)
retrieval_chain = create_retrieval_chain(retriever, summary_chain)


@app.route('/summarize', methods=['POST'])
def summarize():
    start = time.process_time()
    try:
        title_response = title_chain.invoke({"context": documents})
        summary_response = summary_chain.invoke({"context": documents})

        title = title_response.strip()
        summary = summary_response.strip()

        elapsed = time.process_time() - start

        formatted_response = f"""**Title:** {title}

        **Summary:**
        {summary}

        **Response Time:** {elapsed:.2f} seconds
        """
        return formatted_response, 200, {'Content-Type': 'text/plain'}


    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(debug=True)