import math
import os
import time

import torch
from data_utils import get_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)


def evaluate(model_path, num_epochs, batch_size):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        f.write(
            f"NUM EPOCHS: {num_epochs} | BATCH SIZE: {batch_size} | PERPLEXITY: {perplexity:.3f}"
        )
        f.write("############")


def train(
    pretrained_model_path,
    num_epochs,
    batch_size,
    lora=False,
    precision_opt=False,
    gradient_acc=False,
):
    # Load GPT-2 model
    before_time = time.time()
    print(f"time before loading: {before_time}")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    after_time = time.time()
    print(f"time after loading: {after_time}")
    print(f"time passed: {before_time - after_time}")
    train_dset, test_dset, tokenizer = get_datasets(
        root_dir="climate_text_dataset",
        model_name=pretrained_model_path,
        preprocess=False,
        max_length=512,
    )
    print(f"train_dset len: {len(train_dset)}")
    print(f"len(test_dset): {len(test_dset)}")

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
    )

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

    # Run evaluation on the test dataset
    trainer.evaluate()


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 16
    pretrained_model_path = (
        # "/scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1/Llama3.2-3B",
        "gpt2"
    )
    train(
        pretrained_model_path=pretrained_model_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    evaluate(
        f"results/checkpoint-{num_epochs}",
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
