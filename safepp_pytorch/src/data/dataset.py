from pathlib import Path
from typing import Callable, Optional, Dict, Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        required = {'path', 'label'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f'Missing columns in {csv_path}: {missing}')
        self.transform = transform

    @staticmethod
    def infer_source_from_path(path: str) -> str:
        parent = Path(path).parent.name.strip()
        return parent if parent else 'unknown'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        item: Dict[str, Any] = {
            'image': img,
            'label': float(row['label']),
            'path': str(row['path']),
            'source': str(row['source']) if 'source' in self.df.columns else self.infer_source_from_path(str(row['path'])),
            'sample_weight': float(row['sample_weight']) if 'sample_weight' in self.df.columns and pd.notna(row['sample_weight']) else 1.0,
            'is_hard_negative': int(row['is_hard_negative']) if 'is_hard_negative' in self.df.columns and pd.notna(row['is_hard_negative']) else 0,
        }

        for col in ['dataset', 'domain', 'generator', 'hard_type', 'score']:
            if col in self.df.columns:
                item[col] = row[col]
        return item
