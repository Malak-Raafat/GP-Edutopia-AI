from langchain_groq import ChatGroq

# Load API Key from environment variable
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set! Please check your environment variables.")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.2-1b-preview",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

