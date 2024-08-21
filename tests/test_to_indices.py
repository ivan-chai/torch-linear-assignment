from unittest import TestCase, main

import torch
from scipy.optimize import linear_sum_assignment
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices


class TestAssignmentToIndices(TestCase):
    def test_compare_to_scipy(self):
        for shape in [(0, 10, 10), (1, 20, 100), (1, 100, 20), (5, 20, 100), (5, 100, 20)]:
            cost = torch.randn(*shape)
            assignment = batch_linear_assignment(cost)
            row_ind, col_ind = assignment_to_indices(assignment)
            if shape[0] == 0:
                self.assertEqual(row_ind.shape, (0, 0))
                self.assertEqual(col_ind.shape, (0, 0))
                continue
            gt_row_ind, gt_col_ind = [], []
            for c in cost.numpy():
                c_row_ind, c_col_ind = linear_sum_assignment(c)
                gt_row_ind.append(torch.tensor(c_row_ind))
                gt_col_ind.append(torch.tensor(c_col_ind))
            gt_row_ind = torch.stack(gt_row_ind)
            gt_col_ind = torch.stack(gt_col_ind)
            self.assertEqual(row_ind.shape, gt_row_ind.shape)
            self.assertEqual(col_ind.shape, gt_col_ind.shape)
            self.assertTrue((row_ind == gt_row_ind).all())
            self.assertTrue((col_ind == gt_col_ind).all())


if __name__ == "__main__":
    main()
