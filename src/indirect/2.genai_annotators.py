"""
retrives the precomputed labels from the annotated files

"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from func_timeout import func_set_timeout
import re
import ujson as json
import google.generativeai as genai
import anthropic

def load_precomputed_labels(file_path, id_col='id', label_col='labels'):
    df = pd.read_excel(file_path)

    if id_col not in df.columns:
        raise ValueError(
            f"Could not find id column '{id_col}' in {file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if label_col not in df.columns:
        if 'label' in df.columns:
            label_col = 'label'
        else:
            raise ValueError(
                f"Could not find label column '{label_col}' or fallback 'label' in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )

    label_map = {}

    for _, row in df.iterrows():
        sample_id = str(row[id_col]).strip()
        raw_labels = row[label_col]

        if pd.isna(raw_labels):
            parsed_labels = []
        elif isinstance(raw_labels, str):
            parsed_labels = [x.strip() for x in raw_labels.split(',') if x.strip()]
        elif isinstance(raw_labels, list):
            parsed_labels = [str(x).strip() for x in raw_labels if str(x).strip()]
        else:
            parsed_labels = [str(raw_labels).strip()] if str(raw_labels).strip() else []

        label_map[sample_id] = parsed_labels

    return label_map

class Annotator:
    def __init__(
        self,
        engine: str = 'qwen',
        precomputed_files: dict = None
    ):
        self.engine = engine
        self.precomputed_files = precomputed_files or {}

        if engine not in self.precomputed_files:
            raise ValueError(f"No precomputed label file provided for engine '{engine}'")

        self.label_map = load_precomputed_labels(self.precomputed_files[engine])

    def online_annotate(self, sample):
        sample_id = str(sample["id"]).strip()
        return self.label_map.get(sample_id, [])
