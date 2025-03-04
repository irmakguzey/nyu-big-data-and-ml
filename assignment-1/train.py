from data_utils import get_datasets
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)


def train():
    # Load GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(
        "/scratch/BDML25SP/Llama3.2-3B/consolidated.00.pth"
    )
    train_dset, test_dset = get_datasets(
        root_dir="climate_text_dataset",
        model_name="/scratch/BDML25SP/Llama3.2-3B/tokenizer.model",
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
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
    )

    # Train the model
    trainer.train()

    # Run evaluation on the test dataset
    trainer.evaluate()


if __name__ == "__main__":
    train()
