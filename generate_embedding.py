import torch
import torch.nn.functional as F
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset, DatasetDict
from unsloth import is_bfloat16_supported
from prepare_data import formatting_func, balance_data
from tqdm import tqdm


## Login to Huggingface
login(token="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")

## Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "wangrice/ft_llama_sepsis_lmhead", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


## Load the dataset
dataset_name = "wangrice/MIMIC_III_NotesICD_100k_sepsis"
dataset = load_dataset(dataset_name, split="train")
print(dataset[0:5]["TEXT"])
balanced_dataset = balance_data(dataset, field="Sepsis_Diagnosis")
print(f"Train database size: {len(balanced_dataset)}")



# Mean Pooling - Take attention mask into account for correct averaging.
# Attention mask here specifies which tokens were originally attended to
# by others during forward pass, so we only average those token embeddings
# that are included in the mask.
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


max_length = 8192
def get_sentence_embedding(model, tokenizer, example):
    texts = example["TEXT"]
    if len(tokenizer(texts, return_tensors="pt")["input_ids"][0]) > max_length:
      print(f"Skipping example: Text length ({len(tokenizer(texts)['input_ids'])}) exceeds maximum ({max_length})")
      return -1
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
    with torch.no_grad():
      single_output = model(**inputs, output_hidden_states=True)
    last_hidden_state = single_output.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)
    sentence_embedding = mean_pooling(last_hidden_state, inputs['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    return sentence_embedding


vectorized_texts = []
for i in tqdm(range(len(dataset))):
  vectorized_texts.append(get_sentence_embedding(model, tokenizer, dataset[i]))

vectorized_texts = torch.stack(vectorized_texts) 
vectorized_texts = vectorized_texts.squeeze(-1)

print("Vectorized Discharge Summary Report Shape:", vectorized_texts.shape)

# Save the tensor to a file
file_path = 'vectorized_texts.pt'
torch.save(vectorized_texts, file_path)
