import math
import os
import time

import torch
from config import TrainingConfig
from data_utils import get_datasets
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def evaluate(finetuned_model_path, training_cfg):
    if training_cfg.is_lora:
        # Shouldn't load the model but should load the original and add adapter
        model = AutoModelForCausalLM.from_pretrained(training_cfg.model_path)
        tokenizer = AutoTokenizer.from_pretrained(training_cfg.model_path)
        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, finetuned_model_path)

        # model.load_adapter(finetuned_model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model.resize_token_embeddings(len(tokenizer))
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Model vocab size:", model.config.vocab_size)

    checkpoint = (
        finetuned_model_path.split("/")[-1].split("checkpoint")[-1].split("-")[-1]
    )
    evaluation_text = """
    Climate change is caused by an increase in greenhouse gases such as CO2.
    """
    tokens = tokenizer(
        evaluation_text, return_tensors="pt", truncation=True, padding=True
    )

    # Compute loss
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens["input_ids"])
        loss = outputs.loss.item()

    # Compute perplexity
    perplexity = math.exp(loss)
    print(f"Perplexity: {perplexity:.2f}")
    with open("training_evaluations.txt", "a") as f:
        for key, value in training_cfg.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write(f"checkpoint: {checkpoint}\n")
        f.write(f"perplexity: {perplexity:.3f}\n")
        f.write("###################\n")


def train(training_cfg: TrainingConfig):

    # Load GPT-2 model
    if training_cfg.precision_opt:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            training_cfg.model_path, quantization_config=quantization_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(training_cfg.model_path)
    train_dset, test_dset, tokenizer = get_datasets(
        root_dir=training_cfg.root_dir,
        model_name=training_cfg.model_path,
        preprocess=False,
        max_length=training_cfg.max_token_len,
    )
    print(f"train_dset len: {len(train_dset)}")
    print(f"len(test_dset): {len(test_dset)}")
    model.resize_token_embeddings(len(tokenizer))
    if training_cfg.gradient_acc:
        model.gradient_checkpointing_enable()

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=training_cfg.results_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=training_cfg.batch_size,
        num_train_epochs=training_cfg.num_epochs,
        dataloader_num_workers=8,
        auto_find_batch_size=True,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
    )

    if training_cfg.is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    last_checkpoint = trainer.state.global_step

    # Run evaluation on the test dataset
    trainer.evaluate()

    return last_checkpoint


if __name__ == "__main__":

    import time

    timestamp = time.time()
    time_local = time.localtime(timestamp)
    # Format the time struct into a string
    string_local = time.strftime("%m-%d_%H:%M:%S", time_local)
    training_cfg = TrainingConfig(
        # model_path="gpt2",
        # results_dir="./results-gpt",
        model_path="./Llama3.2-3B",
        # results_dir=f"./results-llamba-{string_local}",
        results_dir="results-llamba-2025-03-11_17:11:54",
        root_dir="climate_text_dataset",
        batch_size=32,
        num_epochs=5,
        gradient_accumulation_steps=8,
        max_token_len=128,
        is_lora=True,
        precision_opt=True,
        gradient_acc=False,
    )
    # print(f"config: {training_cfg}")
    # last_checkpoint = train(training_cfg)
    last_checkpoint = 10
    print(f"last_checkpoint: {last_checkpoint}")
    evaluate(
        finetuned_model_path=f"{training_cfg.results_dir}/checkpoint-{last_checkpoint}",
        training_cfg=training_cfg,
    )
