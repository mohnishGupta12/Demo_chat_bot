""""""
import chromadb
from sklearn.cluster import KMeans ,DBSCAN
from functools import wraps
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import  MarkdownTextSplitter
from langchain.indexes import index, SQLRecordManager
from langchain_community.document_loaders import DirectoryLoader
from .utils import  clean_markdown , logger , handle_exceptions
from .config import Config
from langchain.docstore.document import Document
from typing import List



class StoreEmbeddings:
    """
    A class to manage document embedding storage using ChromaDB.
    """

    def __init__(self, chromadb_collection_name: str = None):
        """
        Initializes the ChromaDB client and collection name.

        Args:
            chromadb_collection_name (str, optional): The name of the Chroma collection. Defaults to None.
        """
        try:
            self.client = chromadb.Client()
            self.chromadb_collection_name = chromadb_collection_name or Config.CHROMA_COLLECTION_NAME
            logger.info("StoreEmbeddings initialized successfully.")
        except Exception as err:
            logger.exception("Failed to initialize StoreEmbeddings.")

    @handle_exceptions
    def load_documents(self):
        """
        Loads markdown documents from the specified directory.

        Returns:
            list: A list of loaded and cleaned documents.
        """
        logger.info("Loading documents from directory...")
        loader = DirectoryLoader(Config.DOCUMENTS_PATH, glob="**/*.md", show_progress=True)
        documents = loader.load()

        if not documents:
            logger.warning("No documents found in the specified directory.")

        # Apply cleaning to each document
        for doc in documents:
            doc.page_content = clean_markdown(doc.page_content)

        logger.info(f"Loaded {len(documents)} documents successfully.")
        return documents

    @handle_exceptions
    def get_embeddings(self):
        """
        Returns the embedding model used for vector storage.

        Returns:
            HuggingFaceEmbeddings: The embedding function.
        """
        logger.info("Loading embedding model...")
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)

    @handle_exceptions
    def create_vector_store(self):
        """
        Creates a Chroma vector store from the loaded documents.

        Returns:
            tuple: (bool, list) - Status of operation and indexed vector details.
        """
        logger.info("Creating vector store...")
        documents = self.load_documents()
        if not documents:
            logger.error("No documents found. Aborting vector store creation.")
            return False, []

        text_chunks = self._chunk_text(documents)

        # Setup SQL Record Manager for persistence
        record_manager = SQLRecordManager(namespace="abinbev", db_url="sqlite:///record_manager_cache.sql")
        record_manager.create_schema()

        # Create Chroma vector store and persist it
        collection = Chroma.from_documents(
            documents=text_chunks,
            embedding=self.get_embeddings(),
            collection_name=self.chromadb_collection_name
        )

        # Index and clean-up
        indexing_vector = self._clean_and_update(text_chunks, collection, record_manager)

        logger.info("Vector store created successfully.")
        return True, indexing_vector

    @handle_exceptions
    def _chunk_text(self, documents, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Splits documents into smaller chunks for better embedding and retrieval.

        Args:
            documents (list): List of documents to be split.
            chunk_size (int, optional): Size of each text chunk. Defaults to 200.
            chunk_overlap (int, optional): Overlapping words between chunks. Defaults to 50.

        Returns:
            list: List of document chunks.
        """
        logger.info("Splitting documents into smaller chunks...")
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_chunks = splitter.split_documents(documents)

        logger.info(f"Created {len(docs_chunks)} document chunks.")
        return docs_chunks
    
    @handle_exceptions
    def similarity_chunking_dbscan(self, documents: List[Document], eps: float = 0.5, min_samples: int = 2) -> List[Document]:
        """
        Clusters text documents into semantically similar chunks using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
        
        Args:
            documents (List[Document]): List of Document objects to be clustered.
            eps (float, optional): The maximum distance between two samples for them to be considered as in the same cluster. Defaults to 0.5.
            min_samples (int, optional): The number of samples required to form a dense region. Defaults to 2.
        
        Returns:
            List[Document]: A list of Document objects representing clustered text chunks.
        """
        # Get the embedding model
        model = self.get_embeddings()

        # Extract text content from Document objects
        text_list = [doc.page_content for doc in documents]

        # Generate embeddings for extracted text
        embeddings = model.embed_documents(text_list)

        # Apply DBSCAN clustering using cosine distance
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        # Group text chunks by cluster, ignoring noise (-1 label)
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                # Ignore noise points
                continue
            clusters.setdefault(label, []).append(text_list[i])

        # Combine grouped texts into chunks
        text_chunks = [" ".join(clusters[i]) for i in sorted(clusters.keys())]

        # Convert each text chunk back into a Document object
        return [Document(page_content=chunk) for chunk in text_chunks]


    @handle_exceptions
    def _clean_and_update(self, docs, vectorstore, record_manager):
        """
        Cleans up and updates the vector store.

        Args:
            docs (list): List of document chunks.
            vectorstore (Chroma): Chroma vector store instance.
            record_manager (SQLRecordManager): SQL record manager instance for indexing.

        Returns:
            list: Indexed vector details.
        """
        logger.info("Cleaning up and updating the vector store...")
        return index(
            docs_source=docs,
            vector_store=vectorstore,
            record_manager=record_manager,
            cleanup="full",
            source_id_key="source"
        )
