class Config:  
    """
    Configuration settings for the chatbot application.

    Attributes:
        OPENAI_API_KEY (str): API key for OpenAI GPT models.
        CHROMA_COLLECTION_NAME (str): Name of the ChromaDB collection for vector storage.
        EMBEDDING_MODEL_NAME (str): Name of the embedding model used for text vectorization.
        DOCUMENTS_PATH (str): Path to the directory containing documents to be processed.
    """
    OPENAI_API_KEY= "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"
    CHROMA_COLLECTION_NAME = "abinbev_docs_v2"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    DOCUMENTS_PATH = "app/ubuntu-docs/"
