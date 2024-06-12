from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", padding_side="left")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", quantization_config=quantization_config, device_map="auto", low_cpu_mem_usage=True)

input_text = "What is the best recipe for Pepperoni pizza?"
model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

generated_text = model.generate(**model_inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)

result = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
print(result)

