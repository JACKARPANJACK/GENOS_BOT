import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# Define the model name and token file path
model_name = "google/gemma-2b-it"
token_file = os.path.join(os.path.dirname(__file__), "token.txt")

# Load Hugging Face token from token.txt
try:
    with open(token_file, "r") as f:
        hf_token = f.read().strip()
        if not hf_token:
            raise ValueError("Token file is empty.")
except FileNotFoundError:
    raise FileNotFoundError(f"Missing token file: {token_file}")
except Exception as e:
    raise RuntimeError(f"Error reading token: {e}")

# Log in with the token for authenticated access
login(token=hf_token)

# Try to load the tokenizer and model with optimal settings for speed and memory efficiency
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    print("Loading model...")
    # Load the model on the correct device (GPU if available, otherwise CPU)
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

    # Move model to the chosen device
    model.to(device)

except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Create a fast inference pipeline
text_generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=device  # Use GPU if available, otherwise CPU
)

# Example usage: Generate text for a prompt
prompt = "Hello, I am Genos. How can I assist you today?"
result = text_generator(prompt, max_new_tokens=100, num_return_sequences=1)

# Print the generated text
print("Generated Text: ", result[0]["generated_text"])
