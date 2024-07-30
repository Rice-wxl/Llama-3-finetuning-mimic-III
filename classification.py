from huggingface_hub import login
login(token="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")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from peft import get_peft_model, LoraConfig

peft_config = LoraConfig(
    task_type = "classification",
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    )
model = get_peft_model(
    model, peft_config
)

from datasets import load_dataset

dataset_name = "wangrice/MIMIC_III_NotesICD_100k_sepsis"
dataset = load_dataset(dataset_name, split = "train")

dataset

def add_label(row):
  label = 0
  if row["Sepsis_Diagnosis"] == 'YES':
    label = 1
  return {"label": label}

dataset = dataset.map(add_label)
dataset


# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['TEXT'], truncation=True, padding=True, max_length=8000)

tokenizer.pad_token = tokenizer.eos_token
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.remove_columns(["SUBJECT_ID", "HADM_ID", "CODES", "Sepsis_Diagnosis", "TEXT"])



from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported

model.config.num_labels = 2 

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="finetuned_llama3",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    logging_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    warmup_steps = 5,
    lr_scheduler_type="linear",
    remove_unused_columns=False,  # Prevent automatic column removal
)


trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = encoded_dataset,
    args = training_arguments,
)

trainer_stats = trainer.train()