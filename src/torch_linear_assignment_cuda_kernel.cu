/*
  Implementation is based on the algorihtm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

#include <limits>


template <typename scalar_t>
__device__ __forceinline__
void array_fill(scalar_t *start, scalar_t *stop, scalar_t value) {
  for (; start < stop; ++start) {
    *start = value;
  }
}


template <typename scalar_t, typename index_t>
__device__ __forceinline__
index_t augmenting_path_cuda(index_t nr, index_t nc, index_t i,
			     scalar_t *cost, scalar_t *u, scalar_t *v,
			     index_t *path, index_t *row4col,
			     scalar_t *shortestPathCosts,
			     bool *SR, bool *SC,
			     index_t *remaining, scalar_t *p_minVal,
			     scalar_t infinity)
{
    scalar_t minVal = 0;
    index_t num_remaining = nc;
    for (index_t it = 0; it < nc; it++) {
        remaining[it] = nc - it - 1;
    }

    array_fill(SR, SR + nr, false);
    array_fill(SC, SC + nc, false);
    array_fill(shortestPathCosts, shortestPathCosts + nc, infinity);

    index_t sink = -1;
    while (sink == -1) {
        index_t index = -1;
        scalar_t lowest = infinity;
        SR[i] = true;

        for (index_t it = 0; it < num_remaining; it++) {
            index_t j = remaining[it];
            scalar_t r = minVal + cost[i * nc + j] - u[i] - v[j];
            if (r < shortestPathCosts[j]) {
	      path[j] = i;
	      shortestPathCosts[j] = r;
            }
            if (shortestPathCosts[j] < lowest ||
                (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        if (minVal == infinity) {
            return -1;
        }

        index_t j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
    }
    *p_minVal = minVal;
    return sink;
}


template <typename scalar_t, typename index_t>
__device__ __forceinline__
int solve_cuda_kernel(index_t nr, index_t nc,
		      scalar_t *cost, index_t *matching,
		      scalar_t infinity)
{
  scalar_t *u = new scalar_t[nr];
  scalar_t *v = new scalar_t[nc];
  scalar_t *shortestPathCosts = new scalar_t[nc];
  index_t *path = new index_t[nc];
  index_t *col4row = new index_t[nr];
  index_t *row4col = new index_t[nc];
  bool *SR = new bool[nr];
  bool *SC = new bool[nc];
  index_t *remaining = new index_t[nc];


  array_fill(u, u + nr, (scalar_t) 0);
  array_fill(v, v + nc, (scalar_t) 0);
  array_fill(path, path + nc, (index_t) -1);
  array_fill(row4col, row4col + nc, (index_t) -1);
  array_fill(col4row, col4row + nr, (index_t) -1);

  int exit_code = 0;
  // iteratively build the solution
  for (index_t curRow = 0; curRow < nr; ++curRow) {
    scalar_t minVal;
    index_t sink = augmenting_path_cuda(nr, nc, curRow, cost,
					u, v,
					path, row4col,
					shortestPathCosts,
					SR, SC,
					remaining,
					&minVal, infinity);

    CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");

    u[curRow] += minVal;
    for (index_t i = 0; i < nr; i++) {
      if (SR[i] && i != curRow) {
	u[i] += minVal - shortestPathCosts[col4row[i]];
      }
    }

    for (index_t j = 0; j < nc; j++) {
      if (SC[j]) {
	v[j] -= minVal - shortestPathCosts[j];
      }
    }

    index_t j = sink;
    index_t swap;
    while (1) {
      index_t i = path[j];
      row4col[j] = i;
      swap = j;
      j = col4row[i];
      col4row[i] = swap;
      if (i == curRow) {
	break;
      }
    }
  }

  if (exit_code == 0) {
    for (index_t i = 0; i < nr; i++) {
      matching[i] = col4row[i];
    }
  }

  delete[] u;
  delete[] v;
  delete[] shortestPathCosts;
  delete[] path;
  delete[] col4row;
  delete[] row4col;
  delete[] SR;
  delete[] SC;
  delete[] remaining;

  return exit_code;
}


template <typename scalar_t, typename index_t>
__global__
void solve_cuda_kernel_batch(index_t bs, index_t nr, index_t nc,
			     scalar_t *cost, index_t *matching,
			     scalar_t infinity) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= bs) {
    return;
  }
  solve_cuda_kernel(nr, nc, cost + i * nr * nc, matching + i * nr, infinity);
}


template <typename scalar_t, typename index_t>
void solve_cuda_batch(index_t bs, index_t nr, index_t nc,
		      scalar_t *cost, index_t *matching) {
  TORCH_CHECK(std::numeric_limits<scalar_t>::has_infinity, "Data type doesn't have infinity.");
  auto infinity = std::numeric_limits<scalar_t>::infinity();
  int nt = 256;
  const dim3 block(nt);
  const dim3 grid((bs + nt - 1) / nt);
  solve_cuda_kernel_batch<<<grid, block>>>(
    bs, nr, nc,
    cost, matching,
    infinity);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, cudaGetErrorString(err));
  }
}


torch::Tensor batch_linear_assignment_cuda(torch::Tensor cost) {
  auto sizes = cost.sizes();

  TORCH_CHECK(sizes[2] >= sizes[1], "The number of tasks must be greater or equal to the number of workers.");

  auto device = cost.device();
  auto matching_options = torch::TensorOptions()
    .dtype(torch::kLong)
    .device(device.type(), device.index());
  torch::Tensor matching = torch::empty({sizes[0], sizes[1]}, matching_options);

  // If sizes[2] is zero, then sizes[1] is also zero.
  if (sizes[0] * sizes[1] == 0) {
    return matching;
  }

  AT_DISPATCH_FLOATING_TYPES(cost.type(), "solve_cuda_batch", ([&] {
    solve_cuda_batch<scalar_t, long>(
        sizes[0], sizes[1], sizes[2],
	cost.data<scalar_t>(),
	matching.data<long>());
  }));
  return matching;
}
