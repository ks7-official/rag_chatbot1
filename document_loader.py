import os
from langchain_community.document_loaders import (
    CSVLoader, TextLoader, PDFPlumberLoader,
    Docx2txtLoader, JSONLoader
)

def load_documents(file_path):
    """
    Load documents based on the file extension using LangChain loaders.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        # For CSV, we want to make sure the title and content columns are properly loaded
        loader = CSVLoader(
            file_path,
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
            },
            source_column="title"  # This will put the title in metadata
        )
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PDFPlumberLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".json":
        loader = JSONLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return loader.load()