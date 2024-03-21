import os
from unittest import TestCase, main

import torch
from torch_linear_assignment import batch_linear_assignment


class TestAssignment(TestCase):
    def test_simple(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cost = torch.tensor([
            8, 4, 7,
            5, 2, 3,
            9, 6, 7,
            9, 4, 8
        ]).reshape(1, 4, 3).to(device)
        gt_assignment = torch.tensor(
            [0, 2, -1, 1]
        ).reshape(1, 4)
        result = batch_linear_assignment(cost).cpu()
        print(result)
        self.assertTrue((result == gt_assignment).all())


if __name__ == "__main__":
    main()
