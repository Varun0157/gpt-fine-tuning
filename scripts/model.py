from transformers import GPTNeoForCausalLM

# Download and save the pretrained model
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
model.save_pretrained("./model/model")
