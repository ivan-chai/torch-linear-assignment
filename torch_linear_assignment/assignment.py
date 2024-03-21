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
    if t < w:
        cost = cost.transpose(1, 2)  # (B, T, W).
        assignment = batch_linear_assignment_cuda(cost)  # (B, T).
        matching = torch.full([b, w], -1, dtype=torch.long, device=cost.device)  # (B, W).
        indices = torch.arange(t, dtype=torch.long, device=cost.device)[None].repeat(b, 1)  # (B, T).
        matching.scatter_(1, assignment, indices)
        return matching

    if not isinstance(cost, (torch.FloatTensor, torch.DoubleTensor)):
        cost = cost.to(torch.float)

    return backend.batch_linear_assignment(cost.contiguous())


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
