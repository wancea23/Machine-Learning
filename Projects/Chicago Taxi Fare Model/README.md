# Chicago Taxi Fare Prediction - Linear Regression from Scratch

## Overview

This project implements linear regression using gradient descent to predict taxi fares based on trip miles. Two versions are provided:

1. **Pure NumPy implementation (CPU only)** - manual gradient calculations
2. **PyTorch implementation (CPU/GPU)** - with automatic differentiation and convergence detection

Both implementations use full-batch gradient descent and achieve the same mathematical result.

## Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Required Libraries

For the NumPy version:

```bash
pip install numpy pandas matplotlib
```

For the PyTorch version (with CUDA 11.8 support):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you don't have an NVIDIA GPU, PyTorch will automatically run in CPU mode:

```bash
pip install torch torchvision torchaudio
```

## Dataset

Place the `chicago_taxi_train.csv` file in the following path or update the path in both scripts:

```
...\ML\Projects\Chicago Taxi Fare Model\Dataset\chicago_taxi_train.csv
```

## Program 1: NumPy Implementation (Manual Gradients)

This implementation uses pure NumPy on CPU with manually derived gradient formulas.

**Key features:**

* Manual gradient calculation: `dw = np.mean(2*(y_pred - y) * x)`
* Fixed number of iterations (100)
* Learning rate: 0.001
* Runs exclusively on CPU
* **IT NEEDS MORE ITERATIONS FOR A THEORETICAL CONVERGENCE**

## Program 2: PyTorch Implementation (Automatic Gradients)

This implementation uses PyTorch with automatic differentiation and dynamic convergence detection. This implementation uses PyTorch with automatic differentiation and dynamic convergence detection. Using GPU Acceleration or even just the **torch library** leverages highly optimized C++ code and vectorized operations that execute 100-1000x faster than pure Python loops. PyTorch's autograd automatically computes gradients using efficient reverse-mode differentiation, eliminating manual derivative calculations while maintaining peak performance through optimized BLAS libraries (MKL on CPU, cuBLAS on GPU).

**Key features:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Dynamic while loop that stops when MSE change < 0.000001
* Learning rate: 0.007 (found it optimal)
* Real-time convergence detection

## Results

Both implementations converge to the same model:

```
Final model: FARE = 2.2795 * TRIP_MILES + 5.0000
Final MSE: 14.0254
```

This means:

* Each additional mile increases the fare by approximately $2.28
* Base fare (intercept) is approximately $5.00
* Root Mean Square Error (RMSE): √14.0254 ≈ $3.75

## Key Differences

| Aspect | NumPy Version | PyTorch Version |
|--------|---------------|-----------------|
| Device | CPU only | CPU or GPU (automatic) |
| Gradient Calculation | Manual formulas | Automatic (loss.backward()) |
| Convergence | Fixed iterations (100) | Dynamic (error threshold) |
| Speed | Baseline | Faster on GPU |
| Learning Rate | 0.001 | 0.007 |
| Complexity | Simple, educational | Production-ready |

## Conclusions

1. **Mathematical Equivalence**: Both implementations produce identical results, validating the gradient descent mathematics.

2. **Performance**: The PyTorch GPU version converges faster and can handle larger datasets efficiently, while the NumPy version is excellent for learning the fundamentals.

3. **Educational Value**: The NumPy implementation helps understand the underlying mathematics of gradient descent, while PyTorch demonstrates industry-standard tools.

4. **Model Quality**: With an MSE of 14.0254, the model explains taxi fares well, showing a strong linear relationship between distance and fare.

5. **Scalability**: The PyTorch version automatically scales from CPU to GPU, making it suitable for production deployment.