# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

local_save_path = "./model"

tokenizer.save_pretrained(local_save_path)
model.save_pretrained(local_save_path)
