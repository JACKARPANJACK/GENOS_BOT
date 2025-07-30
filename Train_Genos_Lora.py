import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig

# Load dataset
dataset = load_dataset("json", data_files="genos_generated_from_wiki.jsonl", split="train")

# Tokenizer and base model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

# Format prompts
def format_example(example):
    prompt = f"<|system|>You are Genos, a cyborg with unwavering resolve.<|end|>\n<|user|>{example['instruction']}\n{example['input']}<|end|>\n<|genos|>{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format_example)
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), remove_columns=dataset.column_names)

# LoRA config
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, lora)

# Training args
args = TrainingArguments(
    output_dir="genos_lora_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("genos_lora_adapter")
