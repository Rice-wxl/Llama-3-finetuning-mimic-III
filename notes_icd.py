## Required packages: cuda, pytorch, unsloth, transformers, datasets, xformers, peft, accelerate, bitsandbytes, huggingface_hub, wandb
## pip install huggingface_hub ipython wandb "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from prepare_data import formatting_func, balance_data

import os
os.environ["WANDB_PROJECT"] = "<ft_icdcodes_experiments>"  # name your W&B project
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
                      "gate_proj", "up_proj", "down_proj"],
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
file_paths = ["Notes_ICD_0-5k.json", "Notes_ICD_5-10k.json", "Notes_ICD_15-20k.json", "Notes_ICD_25-30k.json", "Notes_ICD_30-50k.json", "Notes_ICD_50-70k.json", "Notes_ICD_70-100k.json"]
dataset_name = "wangrice/MIMIC_III_NotesICD_100k"
dataset = load_dataset(dataset_name, split="train", data_files=file_paths)
dataset = dataset.map(formatting_func, batched=True, fn_kwargs={'tokenizer': tokenizer})
print(dataset[0:5]["training_text"])
balanced_dataset = balance_data(dataset)

# Split the dataset
split_dataset = balanced_dataset.train_test_split(test_size=0.1)
# Print the size of each split
print(f"Train size: {len(split_dataset['train'])}")
print(f"Test size: {len(split_dataset['test'])}")



# Setting training parameters
training_arguments = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    # optim="adamw_8bit",
    logging_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    # warmup_steps = 50,
    # lr_scheduler_type="cosine",
    save_strategy="epoch",
    eval_strategy='epoch',  # Evaluate at the end of each epoch
    run_name="chat_template_train_1epo_cyclic_1e-4_32", 
    load_best_model_at_end=True  # Load the best model at the end of training
)

# Customize optimizer and scheduler
trainable_params = [param for name, param in model.named_parameters() if param.requires_grad]
optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

scheduler = CyclicLR(
    optimizer, 
    base_lr=1e-5,                      # minimum learning rate
    max_lr=1e-4,                       # maximum learning rate
    step_size_up=300,                 # number of training steps in the increasing half of a cycle
    mode='triangular',                 # mode of the cycle ('triangular', 'triangular2', 'exp_range')
    cycle_momentum=False               # whether to cycle the momentum (should be False for AdamW)
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    dataset_text_field = "training_text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_arguments,
    optimizers=(optimizer, scheduler),
    callbacks=[early_stopping_callback],

)

trainer_stats = trainer.train()

print("\n ######## \nAfter training\n")

# eval_results = trainer.evaluate()

## Save and push to Hub
model.save_pretrained("finetuned_llama3")
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)
model.push_to_hub("wangrice/ft_llama_chat_template", tokenizer, save_method = "lora", token = "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")
