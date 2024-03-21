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
#include <thrust/fill.h>

#include <torch/extension.h>

#include <limits>


typedef unsigned char uint8_t;


int SMPCores()
{
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  switch (devProp.major){
  case 2: // Fermi
    if (devProp.minor == 1)
      return 48;
    else return 32;
  case 3: // Kepler
    return 192;
  case 5: // Maxwell
    return 128;
  case 6: // Pascal
    if ((devProp.minor == 1) || (devProp.minor == 2)) return 128;
    else if (devProp.minor == 0) return 64;
  case 7: // Volta and Turing
    if ((devProp.minor == 0) || (devProp.minor == 5)) return 64;
  case 8: // Ampere
    if (devProp.minor == 0) return 64;
    else if (devProp.minor == 6) return 128;
    else if (devProp.minor == 9) return 128; // ada lovelace
  case 9: // Hopper
    if (devProp.minor == 0) return 128;
  // Unknown device;
  }
  return 128;
}


template <typename scalar_t>
__device__ __forceinline__
void array_fill(scalar_t *start, scalar_t *stop, scalar_t value) {
  for (; start < stop; ++start) {
    *start = value;
  }
}


template <typename scalar_t>
__device__ __forceinline__
int augmenting_path_cuda(int nr, int nc, int i,
                         scalar_t *cost, scalar_t *u, scalar_t *v,
                         int *path, int *row4col,
                         scalar_t *shortestPathCosts,
                         uint8_t *SR, uint8_t *SC,
                         int *remaining,
                         scalar_t *p_minVal,
                         scalar_t infinity)
{
    scalar_t minVal = 0;
    int num_remaining = nc;
    for (int it = 0; it < nc; ++it) {
        SC[it] = 0;
        remaining[it] = nc - it - 1;
        shortestPathCosts[it] = infinity;
    }

    array_fill(SR, SR + nr, (uint8_t) 0);

    int sink = -1;
    while (sink == -1) {
        int index = -1;
        scalar_t lowest = infinity;
        SR[i] = 1;

        scalar_t *cost_row = cost + i * nc;
        scalar_t base_r = minVal - u[i];
        for (int it = 0; it < num_remaining; it++) {
            int j = remaining[it];
            scalar_t r = base_r + cost_row[j] - v[j];
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

        int j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = 1;
        remaining[index] = remaining[--num_remaining];
    }
    *p_minVal = minVal;
    return sink;
}


template <typename scalar_t>
__device__ __forceinline__
void solve_cuda_kernel(int nr, int nc,
                       scalar_t *cost,
                       scalar_t *u, scalar_t *v,
                       scalar_t *shortestPathCosts,
                       int *path, int *col4row, int *row4col,
                       uint8_t *SR, uint8_t *SC,
                       int *remaining,
                       scalar_t infinity)
{
  scalar_t minVal;
  for (int curRow = 0; curRow < nr; ++curRow) {
    auto sink = augmenting_path_cuda(nr, nc, curRow, cost,
                                     u, v,
                                     path, row4col,
                                     shortestPathCosts,
                                     SR, SC,
                                     remaining,
                                     &minVal, infinity);

    CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");

    u[curRow] += minVal;
    for (int i = 0; i < nr; i++) {
      if (SR[i] && i != curRow) {
        u[i] += minVal - shortestPathCosts[col4row[i]];
      }
    }

    for (int j = 0; j < nc; j++) {
      if (SC[j]) {
        v[j] -= minVal - shortestPathCosts[j];
      }
    }

    int i = -1;
    int j = sink;
    int swap;
    while (i != curRow) {
      i = path[j];
      row4col[j] = i;
      swap = j;
      j = col4row[i];
      col4row[i] = swap;
    }
  }
}


template <typename scalar_t>
__global__
void solve_cuda_kernel_batch(int bs, int nr, int nc,
                             scalar_t *cost,
                             scalar_t *u, scalar_t *v,
                             scalar_t *shortestPathCosts,
                             int *path, int *col4row, int *row4col,
                             uint8_t *SR, uint8_t *SC,
                             int *remaining,
                             scalar_t infinity) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= bs) {
    return;
  }

  solve_cuda_kernel(nr, nc,
                    cost + i * nr * nc,
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


template <typename scalar_t>
void solve_cuda_batch(int bs, int nr, int nc,
                      scalar_t *cost, int *col4row, int *row4col) {
  TORCH_CHECK(std::numeric_limits<scalar_t>::has_infinity, "Data type doesn't have infinity.");
  auto infinity = std::numeric_limits<scalar_t>::infinity();

  thrust::device_vector<scalar_t> u(bs * nr);
  thrust::device_vector<scalar_t> v(bs * nc);
  thrust::device_vector<scalar_t> shortestPathCosts(bs * nc);
  thrust::device_vector<int> path(bs * nc);
  thrust::device_vector<uint8_t> SR(bs * nr);
  thrust::device_vector<uint8_t> SC(bs * nc);
  thrust::device_vector<int> remaining(bs * nc);

  thrust::fill(u.begin(), u.end(), (scalar_t) 0);
  thrust::fill(v.begin(), v.end(), (scalar_t) 0);
  thrust::fill(path.begin(), path.end(), -1);

  int blockSize = SMPCores();
  int gridSize = (bs + blockSize - 1) / blockSize;
  solve_cuda_kernel_batch<<<gridSize, blockSize>>>(
    bs, nr, nc,
    cost,
    thrust::raw_pointer_cast(&u.front()),
    thrust::raw_pointer_cast(&v.front()),
    thrust::raw_pointer_cast(&shortestPathCosts.front()),
    thrust::raw_pointer_cast(&path.front()),
    col4row, row4col,
    thrust::raw_pointer_cast(&SR.front()),
    thrust::raw_pointer_cast(&SC.front()),
    thrust::raw_pointer_cast(&remaining.front()),
    infinity);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, cudaGetErrorString(err));
  }
}


std::vector<torch::Tensor> batch_linear_assignment_cuda(torch::Tensor cost) {
  auto sizes = cost.sizes();

  TORCH_CHECK(sizes[2] >= sizes[1], "The number of tasks must be greater or equal to the number of workers.");

  auto device = cost.device();
  auto options = torch::TensorOptions()
    .dtype(torch::kInt)
    .device(device.type(), device.index());
  torch::Tensor col4row = torch::full({sizes[0], sizes[1]}, -1, options);
  torch::Tensor row4col = torch::full({sizes[0], sizes[2]}, -1, options);

  // If sizes[2] is zero, then sizes[1] is also zero.
  if (sizes[0] * sizes[1] == 0) {
    return {col4row, row4col};
  }

  AT_DISPATCH_FLOATING_TYPES(cost.type(), "solve_cuda_batch", ([&] {
    solve_cuda_batch<scalar_t>(
        sizes[0], sizes[1], sizes[2],
        cost.data<scalar_t>(),
        col4row.data<int>(),
        row4col.data<int>());
  }));
  return {col4row, row4col};
}
