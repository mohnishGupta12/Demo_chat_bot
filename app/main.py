import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .chatBot import Chatbot
from .vectorstore import StoreEmbeddings
from .config import Config
from .model import QueryRequest, EmbedRequest
from functools import wraps
from .utils import  clean_markdown , logger , handle_exceptions

# Initialize FastAPI
app = FastAPI()

#Enable CORS (Modify allowed origins for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query")
async def query_chatbot(request: QueryRequest):
    """Handles chatbot queries and allows optional collection selection."""
    
    # Get collection name (default to config if not provided)
    collection_name = request.collection_name or Config.CHROMA_COLLECTION_NAME

    logger.info(f"Received query: {request.query} | Collection: {collection_name}")

    # Initialize chatbot with OpenAI API Key
    chatbot = Chatbot(openai_api_key=Config.OPENAI_API_KEY)
    print(collection_name)
    # Dynamically set collection name for retrieval
    chatbot.collection_name = collection_name
    print(chatbot.collection_name)
    # Get response from chatbot
    response = chatbot.handle_conversation(request.query)

    logger.info(f"Chatbot response: {response}")

    return {"collection": collection_name, "response": response}

@app.post("/embed")
async def create_embeddings(request: EmbedRequest):
    """Creates vector embeddings and updates config dynamically."""

    logger.info(f"Embedding request received for Collection: {request.collection_name}")

    # Update Config with new collection name and directory
    Config.update_config(collection_name=request.collection_name, data_directory=request.directory)

    # Initialize vector store and create embeddings
    embedding_store = StoreEmbeddings()
    embedding_store.create_vector_store(collection_name=request.collection_name, directory=request.directory)

    logger.info(f"Embeddings created successfully for {request.collection_name}")

    return {
        "message": "Embeddings created successfully."
    }

# Health Check Route
@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "API is running"}

# Global Exception Handler for Uncaught Errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handles any uncaught exceptions in the API."""
    logger.exception(f"Unhandled error for request {request.url}")
    return HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
   uvicorn.run(app , port = 1025, host = "127.0.0.1")