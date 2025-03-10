from dataclasses import dataclass


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
