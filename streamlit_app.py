
import streamlit as st
import tempfile
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Placeholder for missing imports (replace with actual implementations if available)
def get_loaders():
    def load_documents(file_path):
        # Mock implementation for demonstration
        from langchain.docstore.document import Document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(page_content=content, metadata={'title': os.path.basename(file_path), 'content': content})]
    
    def get_local_llm():
        # Mock LLM (replace with actual LLM, e.g., from HuggingFace)
        from langchain.llms import FakeListLLM
        return FakeListLLM(responses=["Mock LLM response"])
    
    return load_documents, get_local_llm

# Custom CSS for vibrant UI and bubble input
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #c7d2fe, #a5f3fc);
    }
    .stSidebar {
        background: #f8fafc;
    }
    .stTextInput > div > input {
        background: #ffffff;
        border-radius: 1.5rem;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput > div > input:focus {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .message {
        margin: 0.5rem 0;
        padding: 0.75rem 1.25rem;
        border-radius: 1.5rem;
        max-width: 75%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: #7c3aed;
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background: #fef3c7;
        color: #1e293b;
    }
    .stButton > button {
        background: #10b981;
        color: white;
        border-radius: 1.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background: #059669;
    }
    .clear-button > button {
        background: #f43f5e;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .clear-button > button:hover {
        background: #e11d48;
    }
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        background: #ffffff;
    }
    .stExpander:hover {
        background: #f1f5f9;
    }
    .debug-output {
        font-size: 0.875rem;
        color: #475569;
    }
    h1, h3 {
        color: #4f46e5;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def create_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    processed_docs = []
    for doc in _docs:
        if 'title' in doc.metadata and 'content' in doc.metadata:
            doc.page_content = f"{doc.metadata.get('title', '')}: {doc.metadata.get('content', '')}"
        processed_docs.append(doc)
    
    vectorstore = FAISS.from_documents(processed_docs, embeddings)
    return vectorstore

QA_PROMPT = """
Answer the question based only on the context below. Be specific and provide details from the context.
If the answer is not found in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT,
)

@st.cache_resource(show_spinner=False)
def get_qa_chain(_vectorstore):
    _, get_local_llm = get_loaders()
    llm = get_local_llm()
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

def extract_content_from_context(doc_content, question_keywords):
    content_parts = doc_content.split("content:")
    if len(content_parts) > 1:
        content = content_parts[1].strip()
        return content
    return None

def main():
    st.title("RAG Chatbot with Document Upload")
    
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    load_documents, _ = get_loaders()

    uploaded_file = st.sidebar.file_uploader(
        "Upload a document (CSV, TXT, PDF, DOCX, JSON)",
        type=['csv', 'txt', 'pdf', 'docx', 'json']
    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            docs = load_documents(tmp_path)
            st.sidebar.success(f"Loaded {len(docs)} documents from {uploaded_file.name}")
            
            if debug_mode:
                st.sidebar.write("First 3 documents:")
                for i, doc in enumerate(docs[:3]):
                    st.sidebar.write(f"Doc {i+1} Content: {doc.page_content}")
                    st.sidebar.write(f"Doc {i+1} Metadata: {doc.metadata}")
                    
        except Exception as e:
            st.sidebar.error(f"Failed to load document: {e}")
            return
        finally:
            os.unlink(tmp_path)

        vectorstore = create_vectorstore(_docs=docs)
        qa_chain = get_qa_chain(_vectorstore=vectorstore)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Chat with your document")
            # Display chat messages
            for i, (q, a) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f'<div class="message user-message"><strong>You:</strong> {q}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="message bot-message"><strong>Bot:</strong> {a}</div>', unsafe_allow_html=True)
            
            user_question = st.text_input("Ask a question based on the uploaded document:", key="user_input")
            if user_question:
                with st.spinner("Generating response..."):
                    retrieved_docs = []
                    if debug_mode:
                        st.markdown('<div class="debug-output">Retrieved documents for this question:</div>', unsafe_allow_html=True)
                        retrieved_docs = vectorstore.similarity_search(user_question, k=3)
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f'<div class="debug-output">Doc {i+1}: {doc.page_content}</div>', unsafe_allow_html=True)
                    
                    try:
                        result = qa_chain({"query": user_question})
                        answer = result["result"].strip()
                        
                        if answer == "I don't know" or len(answer) < 10:
                            if debug_mode:
                                st.markdown('<div class="debug-output">Attempting direct extraction from context...</div>', unsafe_allow_html=True)
                            
                            if not retrieved_docs:
                                retrieved_docs = vectorstore.similarity_search(user_question, k=3)
                                
                            keywords = [word.lower() for word in re.findall(r'\w+', user_question)]
                            
                            for doc in retrieved_docs:
                                doc_lower = doc.page_content.lower()
                                if any(keyword in doc_lower for keyword in keywords):
                                    extracted_content = extract_content_from_context(doc.page_content, keywords)
                                    if extracted_content:
                                        title_part = doc.page_content.split('content:')[0]
                                        if any(keyword in title_part.lower() for keyword in keywords):
                                            answer = extracted_content
                                            break
                    except Exception as e:
                        if debug_mode:
                            st.error(f"Error in QA chain: {e}")
                        answer = "I couldn't process that question properly. Please try rephrasing."
                    
                    st.markdown(f'<div class="message bot-message"><strong>Bot:</strong> {answer}</div>', unsafe_allow_html=True)
                    
                    if debug_mode and "source_documents" in result:
                        st.markdown('<div class="debug-output">Source documents:</div>', unsafe_allow_html=True)
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f'<div class="debug-output">Source {i+1}: {doc.page_content}</div>', unsafe_allow_html=True)
                    
                    st.session_state.chat_history.append((user_question, answer))
        
        with col2:
            if st.session_state.chat_history:
                st.markdown("### Chat History")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {q[:30]}..." if len(q) > 30 else f"Q{i+1}: {q}"):
                        st.markdown(f"**Question:** {q}")
                        st.markdown(f"**Answer:** {a}")
                
                if st.button("Clear History", key="clear_history", type="primary"):
                    st.session_state.chat_history = []
                    st.rerun()

if __name__ == "__main__":
    main()
