import math
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as data
from config import TrainingConfig
from data_utils import get_datasets
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from torch.distributed.pipeline.sync import Pipe
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# dist.init_process_group(
#     "ncll",
# )


def evaluate(finetuned_model_path, training_cfg):
    if training_cfg.is_lora:
        # Shouldn't load the model but should load the original and add adapter
        model = AutoModelForCausalLM.from_pretrained(training_cfg.model_path)
        tokenizer = AutoTokenizer.from_pretrained(training_cfg.model_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, finetuned_model_path)

        # model.load_adapter(finetuned_model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
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


def optimization_setup(training_cfg: TrainingConfig):

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

    if training_cfg.gradient_acc:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    train_dset, test_dset, tokenizer = get_datasets(
        root_dir=training_cfg.root_dir,
        model_name=training_cfg.model_path,
        preprocess=False,
        max_length=training_cfg.max_token_len,
    )
    model.resize_token_embeddings(len(tokenizer))

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

    return model, train_dset, test_dset, tokenizer


def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))







if __name__ == "__main__":

    import time

    # Highest batch size with everything -> 64
    # Without gradient_acc -> 16
    # Without anything -> 1
    # Only with lora -> 8
    # Lora + Precision -> 16
    # Lora + GradAcc -> (32)
    # NOTE: Precision Opt Makes things significantly faster
    timestamp = time.time()
    time_local = time.localtime(timestamp)
    # Format the time struct into a string
    string_local = time.strftime("%m-%d_%H:%M:%S", time_local)
    training_cfg = TrainingConfig(
        # model_path="gpt2",
        # results_dir="./results-gpt",
        model_path="./Llama3.2-3B",
        results_dir=f"./results-llamba-{string_local}",
        # results_dir="results-llamba-2025-03-11_17:11:54",
        root_dir="climate_text_dataset",
        batch_size=64,
        num_epochs=250,
        gradient_accumulation_steps=32,
        max_token_len=256,
        is_lora=True,  # Without any of them highest batch size: 1
        precision_opt=True,  #
        gradient_acc=True,  # Without this highest batch size is 16 - now 64 works
    )
    print(f"config: {training_cfg}")
    last_checkpoint = train(training_cfg)
    # last_checkpoint = 10
    print(f"last_checkpoint: {last_checkpoint}")
    evaluate(
        finetuned_model_path=f"{training_cfg.results_dir}/checkpoint-{last_checkpoint}",
        training_cfg=training_cfg,
    )


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.world_size = cfg.num_cpus

        # Create the results dir
        os.makedirs(cfg.results_dir, exist_ok=True)
        self.results_dir = cfg.results_dir

        self._get_dataloaders()

    def _get_dataloaders(self):
        train_dset, test_dset, self.tokenizer = get_datasets(
            root_dir=self.cfg.root_dir,
            model_name=self.cfg.model_path,
            preprocess=False,
            max_length=self.cfg.max_token_len,
        )

        train_sampler = data.DistributedSampler(
            train_dset, drop_last=True, shuffle=True
        )
        test_sampler = data.DistributedSampler(test_dset, drop_last=True, shuffle=False)

        self.train_loader = data.DataLoader(
            train_dset,
            batch_size=self.cfg.batch_size,
            shuffle=train_sampler is None,
            num_workers=self.cfg.num_workers,
            sampler=train_sampler,
        )
        self.test_loader = data.DataLoader(
            test_dset,
            batch_size=self.cfg.batch_size,
            shuffle=test_sampler is None,
            num_workers=self.cfg.num_workers,
            sampler=test_sampler,
        ) # NOTE: You might need to return these loaders in the future 

        # return train_loader, test_loader

    # def train_epoch(self, rank, train_loader): 
    #     for batch in train_loader:

    # This will have all the parallelism configs to be added
    def _get_model(self, rank):
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path)
        model.resize_token_embeddings(len(self.tokenizer))

        # Data parallelism
        if self.cfg.data_parallelism:
            model = DDP(
                model, device_ids=[rank], output_device=rank, broadcast_buffers=False
            )

        if self.cfg.model_parallelism: 
            model = parallelize_module(model, parallel_mode='column', devices=[0,1])

        if self.cfg.pipeline_parallelism: 
            model = Pipe(model, balance=[3,3], devices=[0,1])

        device = torch.device(f'cuda:{rank}')
        model = model.to(device)

        return model

    def train(self, rank):
        # Create default process group
        dist.init_process_group("ncll", rank=rank, world_size=self.world_size)
        dist.barrier()  # Wait for all of the processes to start

        # Set the device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Get dataloaders 
        # train_loader, test_loader = self._get_dataloaders()

        # Load the model

        # Start the training
        for epoch in range(self.cfg.num_epochs):
            # for batch in train_loader:


def main() -> None:
    # We are only training everything distributedly
    cfg = TrainingConfig(
        model_path="./Llama3.2-3B",
        results_dir=f"./results-llamba-{string_local}",
        root_dir="climate_text_dataset",
        batch_size=64,
        num_epochs=250,
        gradient_accumulation_steps=32,
        max_token_len=256,
        data_parallelism=True,
        model_parallelism=True,
        pipeline_parallelism=True,
    )
    workspace = Workspace(cfg)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"

    print("Distributed training enabled. Spawning {} processes.".format(cfg.num_gpus))
    mp.spawn(workspace.train, nprocs=cfg.num_gpus)


if __name__ == "__main__":
    main()
