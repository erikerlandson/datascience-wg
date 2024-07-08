from vllm import LLM, SamplingParams
import time

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]*20

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

llm = LLM(model="ibm-granite/granite-7b-base")

start = time.time()
#outputs = llm.generate(prompts, sampling_params)
end = time.time()

print(f'Time taken for {len(prompts)} prompts = {end-start} seconds')
