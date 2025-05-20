from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def get_local_llm():
    # Load the tokenizer and model for a larger T5 variant for better performance
    model_name = "google/flan-t5-small"  # Using base instead of small for better quality
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create the pipeline with proper configuration
    local_llm = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,  # Shorter but sufficient for QA responses
        temperature=0.1,  # Slight randomness for more natural responses
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    )
    
    # Wrap the pipeline in LangChain's HuggingFacePipeline
    return HuggingFacePipeline(pipeline=local_llm)