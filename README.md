# FastAPI Chatbot

This project is a **FastAPI-based chatbot** that leverages **vector search** using ChromaDB to retrieve relevant responses based on user queries. The chatbot is designed for efficient document retrieval, embedding storage, and response generation.

## Features
- **FastAPI Framework** for API handling
- **Vector Store with ChromaDB** for document search
- **OpenAI Integration** for conversational AI
- **Custom Embedding Creation** for storing and searching knowledge
- **Swagger UI** for easy API testing (`/docs`)
- **Dockerized Deployment** for easy scalability

---
## Project Structure

```
demo_bot_data/
│── app/
│   ├── __init__.py
│   ├── main.py  # FastAPI application
│   ├── chatBot.py  # Chatbot logic
│   ├── vectorstore.py  # Vector store logic
│   ├── config.py  # Configuration settings
│   ├── model.py  # Pydantic models for API validation
│   ├── utils.py  # Utility functions (logging, cleaning, error handling)
│── requirements.txt  # Dependencies
│── Dockerfile  # Docker configuration
│── .dockerignore  # Ignore unnecessary files
│── README.md  # Documentation
```

---
## Installation

### 1️⃣ Clone the Repository
```sh
git clone <repository_url>
cd demo_bot_data
```

### 2️⃣ Set Up Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
###Note-
```
Please provide Open AIkey in the config file to run the endpoint.
```

### 4️⃣ Run the FastAPI Server
```sh
uvicorn app.main:app --host 127.0.0.1 --port 1025
```

- Visit **Swagger UI**: [`http://127.0.0.1:1025/docs`](http://127.0.0.1:1025/docs)

---
## API Endpoints

### 🔹 Health Check
**GET /**
```json
{
  "status": "API is running"
}
```

### 🔹 Query Chatbot
**POST /query**
#### Request:
```json
{
  "query": "What is AI?",
  "collection_name": "default_collection"
}
```
#### Response:
```json
{
  "collection": "default_collection",
  "response": "AI stands for Artificial Intelligence..."
}
```

### 🔹 Create Embeddings
**POST /embed**
#### Request:
```json
{
  "collection_name": "my_collection",
  "directory": "./data"
}
```
#### Response:
```json
{
  "message": "Embeddings created successfully."
}
```

---
## Docker Deployment

### 1️⃣ Build the Docker Image
```sh
docker build -t chatbot-api .
```

### 2️⃣ Run the Container
```sh
docker run -d -p 1025:1025 --name chatbot-api chatbot-api
```

### 3️⃣ Access API in Swagger UI:
[`http://localhost:1025/docs`](http://localhost:1025/docs)


---
## Logging & Error Handling
- **Logging:** Logs are generated for every request and response.
- **Exception Handling:** Custom exception handler ensures robust error management.

---
## Future Improvements
- Implement a **better chunking strategy** for documents.
- Optimize **query retrieval** to reduce response time.
- Enhance **multi-turn conversation handling**.

---
## Contributors
- **Mohnish**



