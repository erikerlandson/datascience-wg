import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

#from: https://huggingface.co/ibm-granite/granite-3b-code-base
device = "cuda:0"

model_path = "ibm-granite/granite-3b-code-base"
#model_path = "ibm-granite/granite-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length=1024

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

def gen_answers(input_text = "how is the weather?"):
    input_tokens = tokenizer(input_text, return_tensors="pt")
    for i in input_tokens:
        input_tokens[i] = input_tokens[i].to(device)

    output = model.generate(**input_tokens,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=True)
    output.sequences = None
    # output = tokenizer.batch_decode(output)
    return output

prompts = [
    "The president of the United States is",
]

outs = []
start = time.time()
for p in prompts:
    o = gen_answers(input_text = p)
    #outs.append(o)
    last_hidden = o.hidden_states[0][-1][0][0]
    print(f"output: {len(last_hidden)}, {last_hidden}")
end = time.time()

print(f'Time taken for {len(prompts)} prompts = {end-start} seconds')

