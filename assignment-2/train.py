import math
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as data
from config import TrainingConfig
from data_utils import get_datasets

# from torch.distributed.pipeline.sync import Pipe
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from training_utils import Every
from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers import Pipe


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.world_size = cfg.num_gpus

        # Create the results dir
        os.makedirs(cfg.results_dir, exist_ok=True)
        self.results_dir = cfg.results_dir

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
        )  # NOTE: You might need to return these loaders in the future

        # return train_loader, test_loader

    # def train_epoch(self, rank, train_loader):
    #     for batch in train_loader:

    # This will have all the parallelism configs to be added
    def _get_model(self, rank):
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path)
        model.resize_token_embeddings(len(self.tokenizer))
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)

        # Data parallelism
        if self.cfg.data_parallelism:
            model = DDP(
                model, device_ids=[rank], output_device=rank, broadcast_buffers=False
            )

        if self.cfg.model_parallelism:
            model = parallelize_module(model, parallel_mode="column", devices=[0, 1])

        # if self.cfg.pipeline_parallelism:
        #     model = Pipe(model, balance=[3, 3], devices=[0, 1])

        return model

    def train_epoch(self, rank, model):
        device = torch.device(f"cuda:{rank}")
        model.train()
        train_loss = 0

        begin = time.time()
        for batch in self.train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            self.optimizer.zero_grad()

            outputs = model(
                inputs,
                labels=labels,
                attention_mask=attention_mask,
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        end = time.time()
        return train_loss / len(self.train_loader), end - begin

    def eval_epoch(self, rank, model):
        device = torch.device(f"cuda:{rank}")
        model.eval()
        test_loss = 0
        for batch in self.test_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    inputs,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                loss = outputs.loss

            test_loss += loss.item()

        return test_loss / len(self.test_loader)

    def _save_checkpoint(self, model, checkpoint_type):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_to_save = model.module  # Unwrap the actual model
        else:
            model_to_save = model
        model_to_save.save_pretrained(
            f"{self.results_dir}/epoch-{checkpoint_type}-checkpoint"
        )
        self.tokenizer.save_pretrained(
            f"{self.results_dir}/epoch-{checkpoint_type}-checkpoint"
        )

    def train(self, rank):
        # Create default process group
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)
        # dist.barrier()  # Wait for all of the processes to start

        try:
            eval_every_epoch = Every(self.cfg.eval_every)

            # Get dataloaders
            self._get_dataloaders()
            print(f"Received dataloaders")

            # Set the device
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")

            # Load the model
            model = self._get_model(rank)
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-4, weight_decay=1e-4
            )

            best_loss = torch.inf

            # Initialize pbar
            if rank == 0:
                print("Starting training...")
                pbar = tqdm(total=self.cfg.num_epochs)
                log_file_path = os.path.join(self.results_dir, "training_log.txt")
                with open(log_file_path, "w") as f:
                    f.write("Training Log\n")
                    f.write("=============\n")
                    f.write("Epoch|Train Loss|Epoch Time|Eval Loss|Best Loss\n")

            # Start the training
            for epoch in range(self.cfg.num_epochs):
                train_loss, time_spent = self.train_epoch(rank, model)

                if eval_every_epoch(epoch) and rank == 0:
                    # dist.barrier()
                    if rank == 0:
                        eval_loss = self.eval_epoch(rank, model)
                        self._save_checkpoint(model=model, checkpoint_type=epoch + 1)

                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            self._save_checkpoint(model=model, checkpoint_type="best")

                if rank == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{self.cfg.num_epochs}, Train Loss: {train_loss:.3f}, Best Loss: {best_loss:.3f}"
                    )
                    eval_loss_str = (
                        f"{eval_loss:.3f}" if eval_every_epoch(epoch) else "N/A"
                    )
                    best_loss_str = (
                        f"{best_loss:.3f}" if best_loss != torch.inf else "N/A"
                    )

                    with open(log_file_path, "a") as f:
                        f.write(
                            f"{epoch + 1}|{train_loss:.3f}|"
                            f"{time_spent:.3f}|{eval_loss_str}|{best_loss_str}\n"
                        )
        except KeyboardInterrupt:
            # dist.barrier()
            dist.destroy_process_group()
            if rank == 0:
                pbar.close()

        # dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            pbar.close()


def evaluate(finetuned_model_path, training_cfg):

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


def main() -> None:
    # We are only training everything distributedly
    timestamp = time.time()
    time_local = time.localtime(timestamp)
    string_local = time.strftime("%m-%d_%H:%M:%S", time_local)
    cfg = TrainingConfig(
        # model_path="./Llama3.2-3B",
        # results_dir=f"./results-llamba-{string_local}",
        model_path="gpt2",
        results_dir=f"./results-gpt-{string_local}",
        root_dir="climate_text_dataset",
        batch_size=8,
        num_epochs=11,
        eval_every=5,
        max_token_len=256,
        data_parallelism=True,
        model_parallelism=False,
        pipeline_parallelism=False,
        num_gpus=4,
    )
    workspace = Workspace(cfg)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"

    print("Distributed training enabled. Spawning {} processes.".format(cfg.num_gpus))
    mp.spawn(workspace.train, nprocs=cfg.num_gpus)


if __name__ == "__main__":
    main()
