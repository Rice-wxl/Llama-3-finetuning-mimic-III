from huggingface_hub import login
from unsloth import FastLanguageModel


login(token="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")

local_path = "/users/xwang259/Llama-3-finetuning-mimic-III/outputs/checkpoint-2584"
model, tokenizer = FastLanguageModel.from_pretrained(local_path)

## Save and push to Hub
model.save_pretrained("finetuned_llama3")
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)
model.push_to_hub("wangrice/ft_llama_chat_template_modified", tokenizer, save_method = "lora", token = "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")

