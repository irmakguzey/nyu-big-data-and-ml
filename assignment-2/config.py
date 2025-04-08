from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_path: str = "gpt2"
    root_dir: str = "climate_text_dataset"
    results_dir: str = "./results"
    batch_size: int = 10
    num_epochs: int = 16
    gradient_accumulation_steps: int = 8
    max_token_len: int = 512
    data_parallelism: bool = False
    model_parallelism: bool = False
    pipeline_parallelism: bool = False
    num_gpus: int = 2
    num_workers: int = 8
