# Load model directly
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-125m"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

local_save_path = "./model"

tokenizer.save_pretrained(local_save_path)
model.save_pretrained(local_save_path)
