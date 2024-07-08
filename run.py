from huggingface_hub import login
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


login(token="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")

## Parameter definiton
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

## Model
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

## Dataset
dataset_name = "wangrice/mimiciiinotes"
dataset = load_dataset(dataset_name, split="train")

prompt_format = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a medical AI assistant for providing the most appropriate and relevant diagnosis-related group (DRG) code for the following discharge summary.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

{}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>

<|end_of_text|>
"""

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs       = examples["TEXT"]
    outputs      = examples["DESCRIPTION"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt_format.format(input, output)
        texts.append(text)
    return { "training_text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched = True,)


# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./finetuned_llama3",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    warmup_steps = 5,
    lr_scheduler_type="linear",
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "training_text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_arguments,
)

trainer_stats = trainer.train()