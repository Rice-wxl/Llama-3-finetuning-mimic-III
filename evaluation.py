import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader


## Login
login(token="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")

## Get the finetuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "wangrice/ft_icd_20k_hyperparameter_2", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


## Prepare the testing dataset
dataset_name = "wangrice/MIMIC_III_Notes_ICD_Test"
dataset = load_dataset(dataset_name, split = "test")

test_prompt = """
<|start_header_id|>system<|end_header_id|>
You are a medical AI assistant for diagnose whether the patient has diabetes mellitus given their discharge notes. You will provide a single yes/no as the answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

def format_batch(batch):
    inputs = batch["TEXT"]
    texts = []
    for input in inputs:
        input_text = test_prompt.format(input)
        texts.append(input_text)
    return { "testing" : texts, }

dataset = dataset.map(format_batch, batched = True,)

## Get all the labels
def get_label(example):
    code = example["CODES"]
    answer = 0
    for each in code.split(','):
        try:
          value = int(each)
          if value  >= 25000 and value <= 25099:
            answer = 1
            break
          else:
            continue
        except ValueError:
          continue
    return answer

labels = [get_label(example) for example in dataset]

## Get all the predictions
max_length = 8192

def get_prediction(model, tokenizer, example, delimiter="<|start_header_id|>assistant<|end_header_id|>"):
    texts = example["testing"]
    if len(tokenizer(texts, return_tensors="pt")["input_ids"][0]) > max_length:
      print(f"Skipping example: Text length ({len(tokenizer(texts)['input_ids'])}) exceeds maximum ({max_length})")
      return -1
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 32, use_cache = True)
    decoded_output = tokenizer.decode(outputs[0])

    # Extract the part of the output after the delimiter    
    if delimiter in decoded_output:
        response = decoded_output.split(delimiter)[-1].strip()
    else:
        response = decoded_output.strip()

    return 1 if "yes" in response.lower() else 0

predictions = [get_prediction(model, tokenizer, example) for example in tqdm(dataset)]

positives = sum(labels)
negatives = len(labels) - sum(labels)

print(f"Number of positives: {positives}")
print(f"Number of negatives: {negatives}")

## Calculate accuracy
correct_predictions = 0
total_predictions = 0
positive_prediction = 0
true_positives = 0

for pred, lab in zip(predictions, labels):
    if pred == -1:
        continue  # Skip predictions where input length exceeded max_length
    
    total_predictions += 1
    if pred == lab:
        correct_predictions += 1
    
    if pred == 1:
       positive_prediction += 1
       if lab == 1:
          true_positives += 1


if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
else:
    accuracy = 0.0  # Handle case where there are no valid predictions

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f"number of positive predictions: {positive_prediction}")
print(f"precision (TP/TP+FP): {true_positives / positive_prediction}")
print(f"recall (TP/TP+FN): {true_positives / positives}")
