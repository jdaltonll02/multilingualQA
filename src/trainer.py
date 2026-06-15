from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Seq2SeqTrainer

from src.dataset import get_lang_label


def build_language_weights(frame, alpha: float) -> np.ndarray:
    """Inverse-frequency sample weights so all languages appear equally during training."""
    langs = frame.apply(get_lang_label, axis=1)
    counts = Counter(langs)
    inv = {lang: 1.0 / (count ** alpha) for lang, count in counts.items()}
    weights = np.array([inv[lang] for lang in langs], dtype=np.float64)
    return weights / weights.mean()


class LanguageBalancedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with optional WeightedRandomSampler for language balancing."""

    def __init__(self, *args, sample_weights=None, use_weighted_sampler: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights      = sample_weights
        self.use_weighted_sampler = use_weighted_sampler

    def get_train_dataloader(self) -> DataLoader:
        if not self.use_weighted_sampler or self.sample_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )
