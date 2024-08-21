import torch

import torch_linear_assignment._backend as backend
from scipy.optimize import linear_sum_assignment


def batch_linear_assignment_cpu(cost):
    b, w, t = cost.shape
    matching = torch.full([b, w], -1, dtype=torch.long, device=cost.device)
    for i in range(b):
        workers, tasks = linear_sum_assignment(cost[i].numpy(), maximize=False)  # (N, 2).
        workers = torch.from_numpy(workers)
        tasks = torch.from_numpy(tasks)
        matching[i].scatter_(0, workers, tasks)
    return matching


def batch_linear_assignment_cuda(cost):
    b, w, t = cost.shape

    if not isinstance(cost, (torch.FloatTensor, torch.DoubleTensor)):
        cost = cost.to(torch.float)

    if t < w:
        cost = cost.transpose(1, 2)  # (B, T, W).
        col4row, row4col = backend.batch_linear_assignment(cost.contiguous())  # (B, T), (B, W).
        return row4col.long()
    else:
        col4row, row4col = backend.batch_linear_assignment(cost.contiguous())  # (B, W), (B, T).
        return col4row.long()


def batch_linear_assignment(cost):
    """Solve a batch of linear assignment problems.

    The method minimizes the cost.

    Args:
      cost: Cost matrix with shape (B, W, T), where W is the number of workers
            and T is the number of tasks.

    Returns:
      Matching tensor with shape (B, W), with assignments for each worker. If the
      task was not assigned, the corresponding index will be -1.
    """
    if cost.ndim != 3:
        raise ValueError("Need 3-dimensional tensor with shape (B, W, T).")
    if backend.has_cuda() and cost.is_cuda:
        return batch_linear_assignment_cuda(cost)
    else:
        return batch_linear_assignment_cpu(cost)


def assignment_to_indices(assignment):
    """Convert assignment to the SciPy format.

    Args:
        assignment: The assignment with shape (B, W).

    Returns:
        row_ind, col_ind: An array of row indices and one of corresponding column indices
            giving the optimal assignment, each with shape (B, K).

    Raises:
        ValueError if batch assignments have different sizes.
    """
    batch_size = assignment.shape[0]
    if batch_size == 0:
        indices = torch.zeros(0, 0, dtype=torch.long, device=assignment.device)
        return indices, indices
    mask = assignment >= 0
    n_matches = mask.sum(1)
    if (n_matches[1:] != n_matches[0]).any():
        raise ValueError("Inconsistent matching sizes.")
    n_matches = n_matches[0].item()
    row_ind = mask.nonzero()[:, 1].reshape(batch_size, n_matches)
    col_ind = assignment.masked_select(mask).reshape(batch_size, n_matches)
    return row_ind, col_ind
