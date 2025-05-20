
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def get_local_llm():
    # Load the tokenizer and model for a larger T5 variant for better performance
    model_name = "google/flan-t5-large"  # Upgraded model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    local_llm = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,  # Increased for detailed responses
        temperature=0.3,  # Slightly higher for natural responses
        device=0 if torch.cuda.is_available() else -1,
    )
    return HuggingFacePipeline(pipeline=local_llm)
