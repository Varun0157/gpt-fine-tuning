from transformers import GPT2Tokenizer

# Download and save the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer.save_pretrained("./model/tokenizer")
