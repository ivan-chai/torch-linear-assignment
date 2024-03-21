/*
  Implementation is based on the algorihtm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <torch/extension.h>

#include <limits>


typedef unsigned int uint32_t;


template <typename scalar_t>
__device__ __forceinline__
void array_fill(scalar_t *start, scalar_t *stop, scalar_t value) {
  for (; start < stop; ++start) {
    *start = value;
  }
}


template <typename scalar_t>
__device__ __forceinline__
uint32_t augmenting_path_cuda(uint32_t nr, uint32_t nc, uint32_t i,
			      scalar_t *cost, scalar_t *u, scalar_t *v,
			      uint32_t *path, uint32_t *row4col,
			      scalar_t *shortestPathCosts,
			      bool *SR, bool *SC,
			      uint32_t *remaining,
			      scalar_t *p_minVal,
			      scalar_t infinity)
{
    scalar_t minVal = 0;
    uint32_t num_remaining = nc;
    for (uint32_t it = 0; it < nc; it++) {
        remaining[it] = nc - it - 1;
    }

    array_fill(SR, SR + nr, false);
    array_fill(SC, SC + nc, false);
    array_fill(shortestPathCosts, shortestPathCosts + nc, infinity);

    uint32_t sink = -1;
    while (sink == -1) {
        uint32_t index = -1;
        scalar_t lowest = infinity;
        SR[i] = true;

        for (uint32_t it = 0; it < num_remaining; it++) {
            uint32_t j = remaining[it];
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

        uint32_t j = remaining[index];
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
void solve_cuda_kernel(uint32_t nr, uint32_t nc,
		       scalar_t *cost, index_t *matching,
		       scalar_t *u, scalar_t *v,
		       scalar_t *shortestPathCosts,
		       uint32_t *path, uint32_t *col4row, uint32_t *row4col,
		       bool *SR, bool *SC,
		       uint32_t *remaining,
		       scalar_t infinity)
{
  array_fill(u, u + nr, (scalar_t) 0);
  array_fill(v, v + nc, (scalar_t) 0);
  array_fill(path, path + nc, (uint32_t) -1);
  array_fill(row4col, row4col + nc, (uint32_t) -1);
  array_fill(col4row, col4row + nr, (uint32_t) -1);

  scalar_t minVal;
  for (uint32_t curRow = 0; curRow < nr; ++curRow) {
    auto sink = augmenting_path_cuda(nr, nc, curRow, cost,
				     u, v,
				     path, row4col,
				     shortestPathCosts,
				     SR, SC,
				     remaining,
				     &minVal, infinity);

    CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");

    u[curRow] += minVal;
    for (uint32_t i = 0; i < nr; i++) {
      if (SR[i] && i != curRow) {
	u[i] += minVal - shortestPathCosts[col4row[i]];
      }
    }

    for (uint32_t j = 0; j < nc; j++) {
      if (SC[j]) {
	v[j] -= minVal - shortestPathCosts[j];
      }
    }

    uint32_t i;
    uint32_t j = sink;
    uint32_t swap;
    while (1) {
      i = path[j];
      row4col[j] = i;
      swap = j;
      j = col4row[i];
      col4row[i] = swap;
      if (i == curRow) {
	break;
      }
    }
  }

  for (uint32_t i = 0; i < nr; i++) {
    matching[i] = col4row[i];
  }
}


template <typename scalar_t, typename index_t>
__global__
void solve_cuda_kernel_batch(uint32_t bs, uint32_t nr, uint32_t nc,
			     scalar_t *cost, index_t *matching,
			     scalar_t *u, scalar_t *v,
			     scalar_t *shortestPathCosts,
			     uint32_t *path, uint32_t *col4row, uint32_t *row4col,
			     bool *SR, bool *SC,
			     uint32_t *remaining,
			     scalar_t infinity) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= bs) {
    return;
  }

  solve_cuda_kernel(nr, nc,
		    cost + i * nr * nc,
		    matching + i * nr,
		    u + i * nr,
		    v + i * nc,
		    shortestPathCosts + i * nc,
		    path + i * nc,
		    col4row + i * nr,
		    row4col + i * nc,
		    SR + i * nr,
		    SC + i * nc,
		    remaining + i * nc,
		    infinity);
}


template <typename scalar_t, typename index_t>
void solve_cuda_batch(uint32_t bs, uint32_t nr, uint32_t nc,
		      scalar_t *cost, index_t *matching) {
  TORCH_CHECK(std::numeric_limits<scalar_t>::has_infinity, "Data type doesn't have infinity.");
  auto infinity = std::numeric_limits<scalar_t>::infinity();

  thrust::device_vector<scalar_t> u(bs * nr);
  thrust::device_vector<scalar_t> v(bs * nc);
  thrust::device_vector<scalar_t> shortestPathCosts(bs * nc);
  thrust::device_vector<uint32_t> path(bs * nc);
  thrust::device_vector<uint32_t> col4row(bs * nr);
  thrust::device_vector<uint32_t> row4col(bs * nc);
  thrust::device_vector<bool> SR(bs * nr);
  thrust::device_vector<bool> SC(bs * nc);
  thrust::device_vector<uint32_t> remaining(bs * nc);

  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
				     (void *) solve_cuda_kernel_batch<scalar_t, index_t>,
				     0, bs);

  int gridSize = (bs + blockSize - 1) / blockSize;
  solve_cuda_kernel_batch<<<gridSize, blockSize>>>(
    bs, nr, nc,
    cost, matching,
    thrust::raw_pointer_cast(&u.front()),
    thrust::raw_pointer_cast(&v.front()),
    thrust::raw_pointer_cast(&shortestPathCosts.front()),
    thrust::raw_pointer_cast(&path.front()),
    thrust::raw_pointer_cast(&col4row.front()),
    thrust::raw_pointer_cast(&row4col.front()),
    thrust::raw_pointer_cast(&SR.front()),
    thrust::raw_pointer_cast(&SC.front()),
    thrust::raw_pointer_cast(&remaining.front()),
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
