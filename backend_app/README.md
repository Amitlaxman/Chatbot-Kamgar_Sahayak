✨ Features
Advanced RAG Pipeline: Goes beyond simple vector search by implementing a multi-stage process: Hypothetical Document Embeddings (HyDE) for superior retrieval, followed by a Cross-Encoder for re-ranking to ensure maximum relevance.

Source-Cited Responses: The chatbot explicitly cites the documents it used to formulate an answer, providing users with verifiable information.

Dynamic Link Integration: Automatically enriches responses with direct links to official documents by referencing a dedicated links collection in the database.

Scalable Architecture: Built with FastAPI for high-performance, asynchronous request handling, and MongoDB Atlas for a robust, scalable data and vector storage solution.

Optimized for Relevance: Handles both highly specific and broad, general questions effectively.

Intelligent Greeting Handling: Provides a friendly, conversational response to common greetings without unnecessarily querying the database.

🏛️ Architecture & Tech Stack
The application consists of two main processes: an Offline Indexing Process to prepare the data and an Online Query Process to handle user chats in real-time.

Offline Indexing Process
This is a one-time setup step (or re-run whenever the source documents are updated).

MongoDB Collections -> indexing.py Script -> Enrich with Links -> Chunk Documents -> Generate Embeddings -> Store in MongoDB Atlas Vector Search

Online Query Process
This is the real-time pipeline that runs for every user question.

User Query -> FastAPI Backend -> HyDE Generation -> Atlas Vector Search (Retrieve) -> Cross-Encoder (Re-rank) -> Build Context -> Groq LLM -> Generate Answer -> Stream to User

Technology Stack
Backend: Python, FastAPI

Database: MongoDB

Vector Store: MongoDB Atlas Vector Search

LLM Provider: Groq (using llama-3.1-8b-instant)

AI Framework: LangChain

Embedding Model: all-MiniLM-L6-v2 (HuggingFace)

Re-ranking Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Sentence Transformers)

Frontend: Flutter (connects to this backend)

📂 Project Structure
.
├── .env                  # Stores environment variables like API keys and database URIs
├── indexing.py           # Script to process and index all source documents
├── main.py               # The main FastAPI backend server application
├── README.md             # This file
└── requirements.txt      # Python dependencies

🚀 Setup and Installation
Follow these steps carefully to get the backend running on your local machine.

Prerequisites
Python 3.9+

A MongoDB Atlas account (free tier is sufficient).

A Groq API Key.

Step 1: Set up MongoDB Atlas
Create a Database: In your Atlas cluster, create a new database named legal_db.

Load Your Data: Populate the legal_db with all your source collections (e.g., Minimum_Wages_Act_1948, Advisory_Content, etc.).

Create the links Collection: Inside legal_db, create a collection named links. Each document in this collection must have the following structure:

{
  "collection_name": "NameOfSourceCollection", // e.g., "Shops_&_Establishments_Act_1958"
  "link": "[http://your-link-to-the-official-document.com](http://your-link-to-the-official-document.com)"
}

Create the Vector Search Index:

In your Atlas dashboard, navigate to the Search tab.

Click "Create Search Index" and select "JSON Editor".

Set the database to legal_db and the collection to legal_vectors (this collection will be created automatically by the script).

Important: Name the index vector_index.

Paste the following JSON configuration:

{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}

Create the index and wait for it to finish building.

Step 2: Configure the Environment
Clone the Repository (or set up your project folder).

Install Dependencies:

pip install -r requirements.txt

Create .env file: Create a file named .env in the root of your project and add the following, replacing the placeholders with your actual credentials:

# .env
MONGO_URI="your_mongodb_atlas_connection_string"
GROQ_API_KEY="your_groq_api_key"

🏃 Running the Application
The application has two distinct run steps. You must run them in this order.

1. Run the Indexing Script (One-Time Setup)
This script will read your source collections, enrich them with links, generate embeddings, and populate your legal_vectors collection in Atlas.

python indexing.py

Note: You only need to re-run this script if you update your source documents in MongoDB.

2. Start the Backend Server
Once indexing is complete, start the FastAPI server.

uvicorn main:app --reload --host 0.0.0.0 --port 8000

The server will now be running and accessible on your local network at http://<your-local-ip>:8000.

🔌 API Endpoint
The chatbot interacts with a single endpoint.

POST /chat
Description: Sends a user query to the RAG pipeline and receives a generated answer.

Request Body:

{
  "query": "What are the rules for night work for women?"
}

Success Response (200 OK):

{
  "response": "Based on the provided documents, women are permitted to work during night hours (9:00 PM to 7:00 AM) in all shops and commercial establishments [Source 1]. Employers must ensure their safety and welfare by implementing measures to prevent sexual harassment [Source 2].",
  "sources": [
    {
      "source_collection": "Shops_&_Establishments_Act_1958_Notification",
      "content_snippet": "English Content:\nIn exercise of the powers conferred by the second proviso to sub-section...",
      "source_link": "[http://example.com/shops_act_notification.pdf](http://example.com/shops_act_notification.pdf)"
    },
    {
      "source_collection": "MP_Shops_and_Establishments_Act_1958_Night_Work_Rules",
      "content_snippet": "English Content:\nRULES FOR EMPLOYMENT OF WOMEN WORKERS DURING NIGHT...",
      "source_link": "[http://example.com/night_work_rules.pdf](http://example.com/night_work_rules.pdf)"
    }
  ]
}

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.