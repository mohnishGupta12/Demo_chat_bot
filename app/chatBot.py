from functools import wraps
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor, create_openai_tools_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
import chromadb
from .config import Config
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.retrievers import BM25Retriever
from langchain import hub
from .utils import clean_markdown , logger , handle_exceptions
from .vectorstore import StoreEmbeddings


class Chatbot:
    """
    Chatbot class that interacts with OpenAI LLM, retrieves relevant documents, and provides responses.
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initializes the chatbot with OpenAI API key, memory, and vector storage.

        Args:
            openai_api_key (str, optional): API key for OpenAI. Defaults to Config.OPENAI_API_KEY.

        Raises:
            Exception: If initialization fails.
        """
        try:
            self.llm = ChatOpenAI(api_key=openai_api_key or Config.OPENAI_API_KEY, model="gpt-3.5-turbo")
            self.client = chromadb.Client()
            self.embedding_function = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
            self.memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True,output_key="output")
            logger.info("Chatbot initialized successfully.")
        except Exception as err:
            logger.exception(f"Error initializing Chatbot: {err}")

    @handle_exceptions
    def query_vector_store(self, query: str) -> str:
        """
        Retrieves relevant documents from the vector store using similarity search.

        Args:
            query (str): The user's input query.

        Returns:
            str: Concatenated relevant document content or a message if no documents are found.
        """
        collection_name = Config.CHROMA_COLLECTION_NAME
        print("list collection in chroma db")
        print(self.client.list_collections())
        collection = self.client.list_collections()
        if not collection:
            obj = StoreEmbeddings()
            _,index= obj.create_vector_store()
        vector_store = Chroma(client=self.client, collection_name=collection_name, embedding_function=self.embedding_function)
        collection = vector_store._collection  # Access the collection
        print(f"Number of stored documents: {collection.count()}")
        vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        relevant_docs = vector_retriever.get_relevant_documents(query)

        if not relevant_docs:
            logger.warning("No relevant documents found in vector store.")
            return "No relevant documents found."

        bm25_retriever = BM25Retriever.from_documents(relevant_docs)
        ranked_docs = bm25_retriever.get_relevant_documents(query, top_k=5)

        output = "\n\n".join([doc.page_content for doc in ranked_docs]) or "No detailed response found."

        return output

    @handle_exceptions
    def get_agent(self):
        """
        Creates an OpenAI Tools Agent with function-calling capabilities.

        Returns:
            Agent: LangChain OpenAI agent instance.
        """
        vector_tool = Tool(
            name="VectorStoreQuery",
            func=self.query_vector_store,
            description="Retrieve full, step-by-step installation guides and detailed responses."
        )
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=[vector_tool],
            prompt=prompt
        )
        return agent

    @handle_exceptions
    def get_summaryfrom_llm(self, query: str) -> str:
        """
        Generates a summary of retrieved documents using LLM.

        Args:
            query (str): The user's input query.

        Returns:
            str: Summarized response from the LLM.
        """
        full_text = self.query_vector_store(query)
        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="Summarize the following text and provide a clear response:\n\n{context}"
        )
        summary_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        summary_response = summary_chain.invoke({"context": full_text})
        return summary_response.get("context", "Summary not generated.")

    def handle_conversation(self, query: str):
        """
        Manages the conversation flow, gets responses, and maintains memory.

        Args:
            query (str): User's input query.
        """
        try:
            response = self.run_agent_once(query)
            logger.info(f"Response: {response}")
            user_resp = self.ask_continue_or_new()

            if user_resp == "continue":
                new_query = input("Please ask your next question: ")
                self.handle_conversation(new_query)
            elif user_resp == "new":
                self.memory.clear()
                logger.info("Starting a new conversation...")
                new_query = input("Please ask your question: ")
                self.handle_conversation(new_query)
            else:
                logger.info("Conversation ended by the user.")
                print("Thank you for chatting! Goodbye!")

        except Exception as err:
            logger.exception(f"Error in handle_conversation: {err}")
        return self.memory
    def ask_continue_or_new(self) -> str:
        """
        Prompts the user to continue the chat, start a new one, or stop.

        Returns:
            str: User's choice ("continue", "new", or "stop").
        """
        while True:
            user_input = input("\nContinue, start new, or stop? (continue/new/stop): ").strip().lower()
            if user_input in ["continue", "new", "stop"]:
                return user_input
            logger.warning("Invalid input received.")
            print("Invalid input. Please enter 'continue', 'new', or 'stop'.")

    @handle_exceptions
    def get_agent_response(self, query: str) -> str:
        """
        Gets the response for a given query using the agent.

        Args:
            query (str): User's input query.

        Returns:
            str: Generated response from the agent.
        """
        agent = self.get_agent()
        return agent.run(query)

    @handle_exceptions
    def run_agent_once(self, query: str) -> str:
        """
        Runs the OpenAI Tools Agent and stops after the first response.

        Args:
            query (str): User's input query.

        Returns:
            str: First generated response from the agent.
        """
        agent = self.get_agent()

        agent_executor = AgentExecutor(
            agent=agent,
            tools=[Tool(
                name="VectorStoreQuery",
                func=self.query_vector_store,
                description="Retrieve full, step-by-step installation guides and detailed responses."
            )],
            memory=self.memory,
            return_intermediate_steps=True,
            max_iterations=1,
            verbose=True
        )

        result = agent_executor.invoke({"input": query})

        observations = [step[1] for step in result.get("intermediate_steps", []) if step[1]]

        if observations:
            return observations[0]
        else:
            return result.get("output", "I couldn't find relevant documents, but here’s what I think:\n" + result["output"])
