from unittest import TestCase, main

import torch
from torch_linear_assignment import batch_linear_assignment


class TestAssignment(TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_simple(self):
        cost = torch.tensor([
            8, 4, 7,
            5, 2, 3,
            9, 6, 7,
            9, 4, 8
        ]).reshape(1, 4, 3).to(self.device)
        gt_assignment = torch.tensor(
            [0, 2, -1, 1]
        ).reshape(1, 4)
        result = batch_linear_assignment(cost).cpu()
        print(result)
        self.assertTrue((result == gt_assignment).all())

    def test_cuda_equal_to_cpu(self):
        if self.device == "cpu":
            return

        for bs, rows, cols in [(16, 20, 40), (1, 30, 10), (0, 5, 5)]:
            cost = torch.randint(-10, 10, (bs, rows, cols))
            matching_cpu = batch_linear_assignment(cost)
            matching_gpu = batch_linear_assignment(cost.to(self.device)).cpu()
            self.assertEqual(matching_cpu.shape, matching_gpu.shape)
            self.assertEqual(matching_cpu.dtype, matching_gpu.dtype)
            self.assertTrue((matching_cpu == matching_gpu).all())


if __name__ == "__main__":
    main()
