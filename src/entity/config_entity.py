## 3. Update the entity

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    input_data_path: Path
    processed_data_path: Path
    X_train_data_path: Path
    X_test_data_path: Path
    y_train_data_path: Path 
    y_test_data_path: Path
    X_val_data_path: Path
    y_val_data_path: Path 
    raw_data_path: Path 

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    tokenizer_data_path: Path
    X_train_data_path: Path
    X_test_data_path: Path
    y_train_data_path: Path
    y_test_data_path: Path
    X_val_data_path: Path
    y_val_data_path: Path
    raw_data_path: Path
    train_dataset_path: Path
    val_dataset_path: Path
    test_dataset_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_dataset_path: Path
    val_dataset_path: Path
    test_dataset_path: Path
    model_data_path: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
  root_dir: Path
  model_data_path: Path
  tokenizer_data_path: Path