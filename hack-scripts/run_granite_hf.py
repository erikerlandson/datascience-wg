import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

#from: https://huggingface.co/ibm-granite/granite-3b-code-base
device = "cuda:0"

#model_path = "ibm-granite/granite-3b-code-base"
model_path = "ibm-granite/granite-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length=1024

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

def gen_answers(input_text = "how is the weather?"):
    input_tokens = tokenizer(input_text, return_tensors="pt")
    for i in input_tokens:
        input_tokens[i] = input_tokens[i].to(device)

    output = model.generate(**input_tokens, max_length=1024)
    output = tokenizer.batch_decode(output)
        
    return output

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]*20

outs = []
start = time.time()
#for p in prompts:
#    o = gen_answers(input_text = p)
#    outs.append(o)
end = time.time()

print(f'Time taken for {len(prompts)} prompts = {end-start} seconds')

