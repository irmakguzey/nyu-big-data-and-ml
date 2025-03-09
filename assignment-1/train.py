from data_utils import get_datasets
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
import time

def train():
    # Load GPT-2 model
    before_time = time.time()
    print(f'time before loading: {before_time}')
    model = AutoModelForCausalLM.from_pretrained(
        "/scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1/Llama3.2-3B"
        #"gpt2"
    )
    after_time = time.time()
    print(f'time after loading: {after_time}')
    print(f'time passed: {before_time - after_time}')
    train_dset, test_dset = get_datasets(
        root_dir="climate_text_dataset",
        model_name="/scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1/Llama3.2-3B",
        #model_name="gpt2",
        preprocess=False,
        max_length=512,
    )
    print(f"train_dset len: {len(train_dset)}")
    print(f"len(test_dset): {len(test_dset)}")

    # Define Training Arguments
    #training_args = TrainingArguments(
    #    output_dir="./results",
    #    evaluation_strategy="epoch",
    #    save_strategy="epoch",
    #    per_device_train_batch_size=8,
    #    per_device_eval_batch_size=8,
    #    num_train_epochs=10,
    #    weight_decay=0.01,
    #    logging_dir="./logs",
    #)

    # Create Trainer
    #trainer = Trainer(
    #    model=model,
    #    args=training_args,
    #    train_dataset=train_dset,
    #    eval_dataset=test_dset,
    #)

    # Train the model
    #trainer.train()

    # Run evaluation on the test dataset
    #trainer.evaluate()


if __name__ == "__main__":
    train()
