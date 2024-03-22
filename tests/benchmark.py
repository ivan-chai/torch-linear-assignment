import logging
import os
import time

import torch
from torch_linear_assignment import batch_linear_assignment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cost = torch.randn(5000, 20, 40)
    start = time.time()
    for _ in range(10):
        batch_linear_assignment(cost)
    total = time.time() - start
    logging.info(f"CPU time: {total} s.")
    if device == "cuda":
        cost = cost.to(device)
        start = time.time()
        for _ in range(10):
            batch_linear_assignment(cost)
        torch.cuda.synchronize()
        total = time.time() - start
        logging.info(f"GPU time: {total} s.")


if __name__ == "__main__":
    main()
