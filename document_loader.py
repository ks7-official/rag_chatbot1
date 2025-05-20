import os
import csv
from langchain_community.document_loaders import (
    TextLoader, PDFPlumberLoader, Docx2txtLoader, JSONLoader
)
from langchain.schema import Document

def load_documents(file_path):
    """
    Load documents based on the file extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        # Manually parse CSV to ensure correct formatting
        documents = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip()
                if question and answer:  # Only include valid rows
                    page_content = f"Question: {question}\nAnswer: {answer}"
                    metadata = {"question": question, "answer": answer, "source": file_path}
                    documents.append(Document(page_content=page_content, metadata=metadata))
        return documents
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PDFPlumberLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".json":
        loader = JSONLoader(
            file_path,
            jq_schema=".[] | .question",
            text_content=False
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        documents = loader.load()
        return documents
    except Exception as e:
        raise ValueError(f"Failed to load document: {str(e)}")
