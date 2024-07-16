## Required packages: cuda, pytorch, unsloth, transformers, datasets, xformers, peft, accelerate, bitsandbytes, huggingface_hub, wandb
## pip install huggingface_hub ipython wandb "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import random

import os
os.environ["WANDB_PROJECT"] = "<ft_icdcodes_balanced>"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"


## Login to Huggingface
login(token="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")

## Parameter definiton
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


## Loading the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


## Preparing the dataset
file_paths = ["Notes_ICD_0-5k.json", "Notes_ICD_5-10k.json", "Notes_ICD_15-20k.json", "Notes_ICD_25-30k.json"]
dataset_name = "wangrice/MIMIC_III_NotesICD_20k"
dataset = load_dataset(dataset_name, split="train", data_files=file_paths)
dataset_name = "wangrice/MIMIC-III-Notes-ICD-IDLess5000"

# split_dataset = dataset.train_test_split(test_size=0.2)
# train_dataset = split_dataset['train']
# test_dataset = split_dataset['test']

prompt_format = """
<|start_header_id|>system<|end_header_id|>
You are a medical AI assistant for diagnose whether the patient has diabetes mellitus given their discharge notes. You will provide a single yes/no as the answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>
<|end_of_text|>
"""

def formatting_prompts_func(examples):
    notes  =  examples["TEXT"]
    codes = examples["CODES"]
    texts = []
    for note, code in zip(notes, codes):
      answer = None
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
      if answer == 1:
        answer = "YES"   
      else:
        answer = "NO"
      text = prompt_format.format(note, answer)
      texts.append(text)
    return { "training_text" : texts, }
pass


test_prompt = """
<|start_header_id|>system<|end_header_id|>
You are a medical AI assistant for diagnose whether the patient has diabetes mellitus given their discharge notes. You will provide a single yes/no as the answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

def test_formatting_func(examples):
    notes  =  examples["TEXT"]
    texts = []
    for note in zip(notes):
      text = test_prompt.format(note)
      texts.append(text)
    return { "training_text" : texts, }
pass

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

dataset = dataset.map(formatting_prompts_func, batched = True,)

# train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
# test_dataset = test_dataset.map(test_formatting_func, batched = True,)


# Separate positive and negative samples
positive_samples = []
negative_samples = []

delimiter="<|start_header_id|>assistant<|end_header_id|>"
for sample in dataset:
  response = sample["training_text"].split(delimiter)[-1].strip()
  if "yes" in response.lower():
    positive_samples.append(sample)
  else:
    negative_samples.append(sample)

# Determine the number of positive samples
num_positive_samples = len(positive_samples)
num_negative_samples = len(negative_samples)
print(f"Number of rows that report yes: {num_positive_samples}")
print(f"Number of rows that report no: {num_negative_samples}")

# Randomly sample the same number of negative samples
sampled_negative_samples = random.sample(negative_samples, num_positive_samples)
balanced_samples = positive_samples + sampled_negative_samples

# Convert the list of balanced samples back to a Dataset
balanced_dataset = Dataset.from_dict({key: [sample[key] for sample in balanced_samples] for key in balanced_samples[0]})

# Setting training parameters
training_arguments = TrainingArguments(
    output_dir="outputs",
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
    save_strategy="no",
    run_name="balanced_trial"
)

# # Customize optimizer and scheduler
# trainable_params = [param for name, param in model.named_parameters() if param.requires_grad]
# optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

# scheduler = CyclicLR(
#     optimizer, 
#     base_lr=5e-6,                      # minimum learning rate
#     max_lr=1e-4,                       # maximum learning rate
#     step_size_up=500,                 # number of training steps in the increasing half of a cycle
#     mode='triangular2',                 # mode of the cycle ('triangular', 'triangular2', 'exp_range')
#     cycle_momentum=False               # whether to cycle the momentum (should be False for AdamW)
# )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=balanced_dataset,
    # eval_dataset=test_dataset,
    dataset_text_field = "training_text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_arguments,
    # optimizers=(optimizer, scheduler)
)

trainer_stats = trainer.train()

print("\n ######## \nAfter training\n")

# eval_results = trainer.evaluate()

## Save and push to Hub
model.save_pretrained("finetuned_llama3")
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)
model.push_to_hub("wangrice/ft_icd_20k_balanced", tokenizer, save_method = "lora", token = "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")
