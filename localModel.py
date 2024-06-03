from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel

import torch
import gc

def generateLocal(inputList, loraPath):
    sampling_params = SamplingParams(temperature=0, max_tokens=900)
    llm = LLM(model="NilanE/tinyllama-en_ja-translation-v3", enable_lora=True)
    lora = LoRARequest("dpo-lora", 1, loraPath)

    prompts = []
    for pair in inputList:
        src = pair['src'].strip()
        prompt = f"Translate this from Japanese to English:\n### JAPANESE:\n{src}\n### ENGLISH:\n"
        prompts.append(prompt)

        # send 5 prompts or more at once to take advantage of vllm batched?
    outputs = llm.generate(prompts, sampling_params, lora_request=lora)
    texts = []
    for output in outputs:
        src = output.prompt.replace("Translate this from Japanese to English:\n### JAPANESE:\n", "").replace("\n### ENGLISH:\n", "").strip()
        ref = ""
        id = -1
        for item in inputList:
            if src == item['src'].strip():
                ref = item['ref'].strip()
                id = item['id'] # misses the point, but works?
                break
        if id != -1:
            generated_text = output.outputs[0].text.strip()
            if generated_text.count('\n') > src.count('\n'):
                generated_text = "\n".join(generated_text.split('\n')[:src.count('\n')]).strip()
                texts.append({'id': id, 'src': src, 'ref': ref, 'output': generated_text})
            elif generated_text.count('\n') == src.count('\n'):
                texts.append({'id': id, 'src': src, 'ref': ref, 'output': generated_text})
        else:
            print("MISSED REF!")

    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()

    return texts
