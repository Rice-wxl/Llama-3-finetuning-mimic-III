import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from prepare_data import formatting_func
import random

## Login
login(token="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")

# Get the finetuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "wangrice/ft_llama_chat_template_modified", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


# local_path = "/users/xwang259/Llama-3-finetuning-mimic-III/outputs/checkpoint-497"
# model, tokenizer = FastLanguageModel.from_pretrained(local_path)
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference


## Prepare the testing dataset
dataset_name = "wangrice/MIMIC_III_Notes_ICD_Test"
dataset = load_dataset(dataset_name, split = "test")
dataset = dataset.map(formatting_func, batched=True, fn_kwargs={'tokenizer': tokenizer, "is_testing": True})



# test_prompt = """
# <|start_header_id|>system<|end_header_id|>
# You are a medical AI assistant for diagnose whether the patient has diabetes mellitus given their discharge notes. You will provide a single yes/no as the answer.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Here is a patient's discharge summary report from a hospital encounter. Provide a one word yes or no answer to the following question: does the patient have diabetes mellitus?
# *** Discharge summary report starts ***
# {}
# *** Discharge summary report ends ***<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

# def format_batch(batch):
#     inputs = batch["TEXT"]
#     texts = []
#     for input in inputs:
#         input_text = test_prompt.format(input)
#         texts.append(input_text)
#     return { "testing" : texts, }

# dataset = dataset.map(format_batch, batched = True,)



## Get all the labels
def get_label(example):
    code = example["CODES"]
    answer = 0
    for each in code.split(','):
        try:
          value = int(each)
          if value >= 25000 and value <= 25099:
            answer = 1
            break
          else:
            continue
        except ValueError:
          continue
    return answer

labels = [get_label(example) for example in dataset]
print(f"Original testing sample size: {len(labels)}")


## Balanced data
indexes_of_ones = [index for index, value in enumerate(labels) if value == 1]
indexes_of_zeros = [index for index, value in enumerate(labels) if value == 0]
random_zero_indexes = random.sample(indexes_of_zeros, len(indexes_of_ones))


indexes = indexes_of_ones + random_zero_indexes
subset_labels = [labels[index] for index in indexes]

## Get all the predictions
max_length = 8192

def get_prediction(model, tokenizer, example, delimiter="<|start_header_id|>assistant<|end_header_id|>"):
    texts = example["training_text"]
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

    if "yes" in response.lower():
      return 1
    elif "no" in response.lower():
      return 0
    else:
      return -2

predictions = []
for i in tqdm(range(len(subset_labels))):
   predictions.append(get_prediction(model, tokenizer, dataset[indexes[i]]))


## Calculate accuracy
positives = 0
negatives = 0

total_predictions = 0

correct_predictions = 0
positive_prediction = 0
true_positives = 0

for pred, lab in zip(predictions, subset_labels):
    if pred == -1:
        continue  # Skip predictions where input length exceeded max_length
    
    if pred == -2:
      print("no yes/no answer in the response, skips")
      continue

    total_predictions += 1

    if lab == 1:
       positives += 1
    else:
       negatives += 1

    if pred == lab:
        correct_predictions += 1

    if pred == 1:
       positive_prediction += 1
    
    if pred == 1 & lab == 1:
       true_positives += 1

if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
else:
    accuracy = 0.0  # Handle case where there are no valid predictions

precision = true_positives / positive_prediction
recall = true_positives / positives

print(f'positive samples: {positives}')
print(f'negative samples: {negatives}')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f"number of positive predictions: {positive_prediction}")
print(f"precision (TP/TP+FP): {precision}")
print(f"recall (TP/TP+FN): {recall}")
print(f"F1 score: {2/(precision**-1 + recall**-1)}")