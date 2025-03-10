import math
import os
import time
from dataclasses import dataclass

import torch
from data_utils import get_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainingConfig:
    model_path: str = "gpt2"
    root_dir: str = "climate_text_dataset"
    results_dir: str = "./results"
    batch_size: int = 10
    num_epochs: int = 16
    gradient_accumulation_steps: int = 8
    is_lora: bool = False
    precision_opt: bool = False
    gradient_acc: bool = False
    max_token_len: int = 512


def evaluate(finetuned_model_path, training_cfg):
    model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model.resize_token_embeddings(len(tokenizer))

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
        f.write(f"checkpoint: {checkpoint}")
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
        model.gradient_checkpoint_enabled()

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=training_cfg.results_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=training_cfg.batch_size,
        num_train_epochs=training_cfg.num_epochs,
        dataloader_num_workers=16,
        weight_decay=0.01,
        fp16=training_cfg.precision_opt,  # TODO: Do ths bf16 as well?
        logging_dir="./logs",
    )

    if training_cfg.is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            # target_modules=["q_proj", "v_proj"],
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
    training_cfg = TrainingConfig(
        # model_path="gpt2",
        # results_dir="./results-gpt",
        model_path="/scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1/Llama3.2-3B",
        results_dir="./results-llamba",
        root_dir="climate_text_dataset",
        batch_size=16,
        num_epochs=10,
        gradient_accumulation_steps=8,
        max_token_len=512,
        is_lora=True,
        precision_opt=False,
        gradient_acc=False,
    )
    print(f"config: {training_cfg}")

    # last_checkpoint = train(training_cfg)
    # print(f"last_checkpoint: {last_checkpoint}")
    evaluate(
        finetuned_model_path=f"{training_cfg.results_dir}/checkpoint-3",
        training_cfg=training_cfg,
    )
