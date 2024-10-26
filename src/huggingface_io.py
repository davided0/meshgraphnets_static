from pathlib import Path
from typing import Iterator, Any
from datasets import load_dataset, Dataset

def huggingface_dataset_generator(path: str, split: str) -> Iterator[Any]:
    try:
        if (Path(path)/split).exists():
            dataset = Dataset.load_from_disk(str(Path(path)/split))
        else:
            dataset = load_dataset(path)[split]
        yield from dataset
    except KeyError:
        raise ValueError(f"Split '{split}' is not found in the dataset.")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")