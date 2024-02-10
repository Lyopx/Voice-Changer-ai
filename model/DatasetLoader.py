import pickle
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import logging
import time
import sys

sys.path.append("../")
from config import data_path


class DatasetLoader(Dataset):
    def __init__(self, args, dataset_type):
        self.args = args
        self.dataset_type = dataset_type
        self.y_seq_len = eval(
            "self.args.{}_input_length".format(self.args.model.lower())
        )

        self.dataset_dir = os.path.join(data_path, self.args.dataset)

        logging.info("\n" + "=" * 20 + "\n")
        logging.info("{} Dataset".format(dataset_type))
        self.loading_data()
        logging.info("\n" + "=" * 20 + "\n")

    def loading_whisper(self):
        logging.info("Loading Whisper features...")
        with open(
            os.path.join(self.dataset_dir, "Whisper/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.whisper = pickle.load(f)
        logging.info(
            "Whisper: sz = {}, shape = {}".format(
                len(self.whisper), self.whisper[0].shape
            )
        )

    def loading_MCEP(self):
        logging.info("Loading MCEP features...")
        with open(
            os.path.join(self.dataset_dir, "MCEP/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.mcep = pickle.load(f)
        logging.info(
            "MCEP: sz = {}, shape = {}".format(len(self.mcep), self.mcep[0].shape)
        )
        self.y_d = self.mcep[0].shape[1]

        # Padding
        sz = len(self.mcep)
        self.y_gt = torch.zeros(
            (sz, self.y_seq_len, self.y_d), device=self.args.device, dtype=torch.float
        )
        self.y_mask = torch.zeros(
            (sz, self.y_seq_len, 1), device=self.args.device, dtype=torch.long
        )
        for idx in range(sz):
            y, mask = self.get_padding_y_gt(idx)
            self.y_gt[idx] = y
            self.y_mask[idx] = mask

    def loading_data(self):
        t = time.time()

        self.loading_whisper()
        self.loading_MCEP()

        logging.info("Done. It took {:.2f}s".format(time.time() - t))

    def __len__(self):
        return len(self.y_gt)

    def __getitem__(self, idx):
        # y_gt, mask = self.get_padding_y_gt(idx)
        sample = (idx, self.y_gt[idx], self.y_mask[idx])
        return sample

    def get_padding_y_gt(self, idx):
        y_gt = torch.zeros(
            (self.y_seq_len, self.y_d), device=self.args.device, dtype=torch.float
        )
        mask = torch.ones(
            (self.y_seq_len, 1), device=self.args.device, dtype=torch.long
        )

        mcep = self.mcep[idx]
        sz = min(self.y_seq_len, len(mcep))
        y_gt[:sz] = torch.as_tensor(mcep[:sz], device=self.args.device)
        mask[sz:] = 0

        return y_gt, mask
