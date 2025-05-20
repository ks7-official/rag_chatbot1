RAG Chatbot with Document Upload
A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, allowing users to upload documents (CSV, TXT, PDF, DOCX, JSON) and ask questions based on their content. The app uses LangChain, FAISS, and HuggingFace embeddings for document retrieval and question answering, featuring a vibrant UI with a bubble-style input, colorful messages, and an expandable chat history.
Features

I Dont have OPENAI API as it is a paid version hence i have used huggingface llm in this chatbot,openai would have reduced the code structure even more better way

Document Upload: Supports CSV, TXT, PDF, DOCX, and JSON files for content extraction and querying.
RAG Pipeline: Uses FAISS for vector storage and HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2) for similarity-based retrieval.
Interactive UI: Includes a sidebar for uploads and debug mode, a main chat area with bubble-style input, and a chat history column with expandable entries.
Debug Mode: Displays retrieved documents and source content for transparency during development.
Chat History: Persists user questions and bot responses, with a clear history option.
Styling: Vibrant gradient background (indigo to teal), purple user messages, yellow bot messages, and green buttons for a modern look.

Repository Structure
rag_chatbot/
├── streamlit_app.py        # Main Streamlit application with RAG logic and UI
├── document_loader.py      # Logic for loading various document types
├── rag_pipeline.py         # Local LLM setup using HuggingFace
├── sample_qa.txt           # Sample questions and chatbot responses
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation (this file)

Prerequisites

Python 3.8 or higher
Git
Streamlit Community Cloud account (for deployment)

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/rag_chatbot.git
cd rag_chatbot


Install Dependencies:Create a virtual environment and install required packages:
pip install -r requirements.txt


Run the App Locally:
streamlit run streamlit_app.py

The app will open in your default browser at http://localhost:8501.

Upload a Document:

Use the sidebar to upload a document (CSV, TXT, PDF, DOCX, or JSON).
Supported formats are processed using LangChain loaders (document_loader.py).
A success or error message will appear in the sidebar.


Check the "Debug Mode" box in the sidebar to view document content and retrieved chunks.

Ask Questions:

Enter a question in the bubble-style input field in the main chat area.
Press Enter to submit; the chatbot will respond based on the document content.
Responses appear as yellow bubbles, with user questions in purple.


View Chat History:

The right column displays chat history with expandable entries (click to view full question and answer).
Use the "Clear History" button to reset the conversation.

Sample QA:

See sample_qa.txt for example questions and responses based on a sample document about space exploration.

Test the Deployed App:

Upload a document, ask questions, and verify the UI and functionality.
Note: If using a large LLM (e.g., google/flan-t5-small), ensure it fits within Streamlit Cloud’s resource limits (1GB RAM).

Limitations

LLM Choice: The OpenAI API, being a paid service, was not used in this project. Instead, a HuggingFace LLM (google/flan-t5-small) was implemented to keep the solution cost-free. While the OpenAI API could have simplified the code structure and potentially improved performance, the HuggingFace LLM provides a robust, open-source alternative for the RAG pipeline.

