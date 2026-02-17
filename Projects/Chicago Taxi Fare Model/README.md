# Chicago Taxi Fare Prediction

A linear regression implementation from scratch to predict taxi fares based on trip characteristics, featuring both NumPy and PyTorch versions with comprehensive feature engineering experiments.

## Overview

This project explores multiple approaches to predicting Chicago taxi fares using gradient descent:

- **Pure NumPy implementation** - Manual gradient calculations for educational purposes (CPU only)
- **PyTorch implementation** - Automatic differentiation with GPU acceleration and convergence detection
- **Feature engineering experiments** - Comparing prediction quality across different feature sets
- **Day/Night split analysis** - Investigating fare differences by time of day

Both core implementations use full-batch gradient descent and achieve mathematically equivalent results.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (optional, for PyTorch GPU acceleration)

### Required Libraries

**For the NumPy version:**

```bash
pip install numpy pandas matplotlib
```

**For the PyTorch version (with CUDA 11.8 support):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only PyTorch:**

```bash
pip install torch torchvision torchaudio
```

> Note: If you don't have an NVIDIA GPU, PyTorch will automatically run in CPU mode.

## Dataset

Place the `chicago_taxi_train.csv` file in the following path or update the path in all scripts:

```
...\ML\Projects\Chicago Taxi Fare Model\Dataset\chicago_taxi_train.csv
```
![Data_set_visualization](Images\mpl.png)

## Program Variations

## Model_0.1: NumPy Implementation (Manual Gradients)

Pure NumPy implementation on CPU with manually derived gradient formulas.

**Features:**
* TRIP_MILES

**Label:**
* FARE

**Key points:**
* Manual gradient calculation: `dw = np.mean(2*(y_pred - y) * x)`
* Fixed number of iterations (100)
* Learning rate: 0.001
* Runs exclusively on CPU
* **IT NEEDS MORE ITERATIONS FOR A THEORETICAL CONVERGENCE**

## Disclaimer
*These next implementation uses PyTorch with automatic differentiation and dynamic convergence detection. This implementation uses PyTorch with automatic differentiation and dynamic convergence detection. Using GPU Acceleration or even just the **torch library** leverages highly optimized C++ code and vectorized operations that execute 100-1000x faster than pure Python loops. PyTorch's autograd automatically computes gradients using efficient reverse-mode differentiation, eliminating manual derivative calculations while maintaining peak performance through optimized BLAS libraries (MKL on CPU, cuBLAS on GPU).*

*Each program is trained with different features, but with the same implementation logic, the number at the end of each model's name means the number of features it was trained on*

*The models are listed in the same order they have been created, because one model leads to new questions*

## Model_1.1.1: PyTorch Implementation (TRIP_MILES)

**Features:**
* TRIP_MILES

**Label:**
* FARE

**Key points:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Dynamic while loop that stops when MSE change < 0.000001
* Learning rate: 0.007 (found it optimal)
* Real-time convergence detection

### Results

```
Final model: FARE = 2.2795 * TRIP_MILES + 5.0000
Final MSE: 14.0254
```

![Data_set_visualization](Images\Figure_1.1.1.png)


#### This means:

* Each additional mile increases the fare by approximately $2.28
* Base fare (intercept) is approximately $5.00
* Root Mean Square Error (RMSE): √14.0254 ≈ $3.75

**The error is still big! The actual equation for a Chicago Taxi Fare is:**
```
FARE = 2.25 × MILES + 0.12 × MINUTES + 3.25
```

## Model_1.1.5 (Miles + Seconds + Speed + Tips + TipRate)

I was wondering, if I was to use all the features the data set provides, the model should predict the most accurate fares right?

**Features:**
* TRIP_MILES
* TRIP_SECONDS
* TRIP_SPEED
* TIPS
* TIP_RATE

**Label:**
* FARE

**Key points:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Convergence threshold: 1e-05 (MSE change < 0.00001)
* Learning rate: 0.00001 (reduced for stability with 5 features)
* Required 495,112 iterations to converge
* Demonstrates that more features ≠ better predictions

### Output:
```
Converged at iteration 495112!
MSE change: 0.0000095367 < 1e-05
------------------------------------------------------------
Final model: FARE = 0.2355 * TRIP_MILES + 0.0108 * TRIP_SECONDS + 0.4287 * TRIP_SPEED + 0.0771 * TIPS + -0.0301 * TIP_RATE + -0.0017
Final MSE: 30.4975

Model trained successfully!
Used device: cuda
GPU: NVIDIA GeForce RTX 3060 Ti

------------------------------------------------------------
Random 10 predictions:
------------------------------------------------------------
Features: Miles=4.55, Sec=589, Speed=27.8, Tips=0.00, TipRate=0.00
  Actual: 14.00$, Predicted: 19.32$, Error: 5.32$

Features: Miles=1.99, Sec=654, Speed=11.0, Tips=3.00, TipRate=33.30
  Actual: 9.00$, Predicted: 11.44$, Error: 2.44$

Features: Miles=1.37, Sec=560, Speed=8.8, Tips=0.00, TipRate=0.00
  Actual: 7.75$, Predicted: 10.12$, Error: 2.37$

Features: Miles=3.50, Sec=794, Speed=15.9, Tips=0.00, TipRate=0.00
  Actual: 12.00$, Predicted: 16.18$, Error: 4.18$

Features: Miles=19.87, Sec=2446, Speed=29.2, Tips=10.85, TipRate=20.20
  Actual: 49.75$, Predicted: 43.73$, Error: 6.02$

Features: Miles=0.74, Sec=323, Speed=8.2, Tips=0.00, TipRate=0.00
  Actual: 5.50$, Predicted: 7.16$, Error: 1.66$

Features: Miles=5.80, Sec=1020, Speed=20.5, Tips=4.75, TipRate=25.70
  Actual: 18.50$, Predicted: 20.71$, Error: 2.21$

Features: Miles=16.30, Sec=1940, Speed=30.2, Tips=11.75, TipRate=25.30
  Actual: 42.50$, Predicted: 37.79$, Error: 4.71$

Features: Miles=11.82, Sec=1769, Speed=24.1, Tips=7.69, TipRate=25.40
  Actual: 30.25$, Predicted: 31.97$, Error: 1.72$

Features: Miles=17.30, Sec=1980, Speed=31.5, Tips=9.50, TipRate=20.00
  Actual: 43.50$, Predicted: 39.00$, Error: 4.50$


Average Error on 10 random samples: 3.51$
RMSE: 5.52$
```

![Data_set_visualization](ML\Projects\Chicago Taxi Fare Model\Images\Figure_1.png)


#### This means:

* Each additional mile increases the fare by approximately $0.24
* Each additional second increases the fare by approximately $0.01
* Each unit increase in speed (mph) increases the fare by approximately $0.43
* Each dollar of tip increases the fare by approximately $0.08
* Each percentage point increase in tip rate decreases the fare by approximately $0.03
* Base fare (intercept) is approximately -$0.00
* Root Mean Square Error (RMSE): $5.52
* Average Error on random samples: $3.51

**Note:** This model performs poorly compared to the simpler model (Model_1.1.1), which got RMSE = $3.75. The inclusion of tips and tip rate introduces noise, even though at first might sound as a good idea: bigger fare = bigger tips left, but tips are a consequence of the fare, not a predictor of it. The model struggles with predictions, showing errors ranging from $1.66 to $6.02 on sample predictions. Also the base fare for a taxi could never be zero, nevertheless negative.

**In order to train a model with these many features, the learning rate used was very small, which lead to a slower convergence**

## Model_1.1.3 (Miles + Seconds + Speed)

After the failure that **Model_1.1.5** predicted, I decided to train a model without the creates noise from tips and tip rate.

**Features:**
* TRIP_MILES
* TRIP_SECONDS
* TRIP_SPEED

**Label:**
* FARE

**Key points:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Convergence threshold: 1e-05 (MSE change < 0.00001)
* Learning rate: 0.00001
* Required 429,546 iterations to converge
* Multicollinearity issue: Speed is derived from Miles/Seconds, creating redundant information

### Output:

```
Converged at iteration 429546!
MSE change: 0.0000095367 < 1e-05
------------------------------------------------------------
Final model: FARE = 0.9430 * TRIP_MILES + 0.0072 * TRIP_SECONDS + 0.3494 * TRIP_SPEED + -0.0641
Final MSE: 16.7746

Model trained successfully!
Used device: cuda
GPU: NVIDIA GeForce RTX 3060 Ti

------------------------------------------------------------
Random 10 predictions:
------------------------------------------------------------
Features: Miles=4.51, Sec=1514, Speed=10.7
  Actual: 16.20$, Predicted: 18.90$, Error: 2.70$

Features: Miles=15.82, Sec=1971, Speed=28.9
  Actual: 40.50$, Predicted: 39.24$, Error: 1.26$

Features: Miles=17.65, Sec=2406, Speed=26.4
  Actual: 44.25$, Predicted: 43.25$, Error: 1.00$

Features: Miles=18.18, Sec=2926, Speed=22.4
  Actual: 45.50$, Predicted: 46.12$, Error: 0.62$

Features: Miles=1.48, Sec=510, Speed=10.4
  Actual: 7.50$, Predicted: 8.66$, Error: 1.16$

Features: Miles=1.32, Sec=355, Speed=13.4
  Actual: 6.75$, Predicted: 8.43$, Error: 1.68$

Features: Miles=31.02, Sec=3501, Speed=31.9
  Actual: 76.50$, Predicted: 65.72$, Error: 10.78$

Features: Miles=4.38, Sec=1060, Speed=14.9
  Actual: 14.25$, Predicted: 16.95$, Error: 2.70$

Features: Miles=9.71, Sec=1582, Speed=22.1
  Actual: 27.50$, Predicted: 28.28$, Error: 0.78$

Features: Miles=17.99, Sec=3954, Speed=16.4
  Actual: 48.75$, Predicted: 51.29$, Error: 2.54$


Average Error on 10 random samples: 2.52$
RMSE: 4.10$
```

![Data_set_visualization](Images\Figure_1.3.png)

#### This means:

* Each additional mile increases the fare by approximately $0.94
* Each additional second increases the fare by approximately $0.01 (or $0.43 per minute)
* Each unit increase in speed (mph) increases the fare by approximately $0.35
* Base fare (intercept) is approximately -$0.06
* Root Mean Square Error (RMSE): $4.10
* Average Error on random samples: $2.52

**Note:** From the start we can see that something is wrong, just by looking at the Base Fare price. Removing tips and tip rate significantly improved the model (RMSE dropped from $5.52 to $4.10). However, the model still underperforms compared to the simpler Miles **Model_1.1.1**. The issue is **multicollinearity** - speed is mathematically derived from miles and seconds (Speed = Miles/Seconds), creating redundant information that confuses the model rather than improving it. Sample predictions show generally good accuracy, with most errors under $3, except for one outlier ($10.78 error on a 31-mile trip).

**In order to train a model with these many features, the learning rate used was very small, which lead to a slower convergence**

## Model_1.1.2 (Miles + Minutes)

After another failure from the previous model, I decided to stick only with two features. Also the TRIP_SECONDS was transformed in TRIP_MINUTES, because it is important that all numeric values are roughly on the same scale. The mean value for TRIP_MILES is 8.3 and the mean for TRIP_SECONDS is 1,320; that is two orders of magnitude difference **(x159 bigger)**. Unstable means the training process becomes chaotic and may not converge properly, but by normalizing the features, we are able to use a bigger LR.

**Features:**
* TRIP_MILES
* TRIP_MINUTES (TRIP_SECONDS/60)

**Label:**
* FARE

**Key points:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Convergence threshold: 1e-05 (MSE change < 0.00001)
* Learning rate: 0.0001 (increased due to better feature scaling)
* Required 125,212 iterations to converge
* Feature normalization: Converted TRIP_SECONDS to TRIP_MINUTES to match scale with TRIP_MILES

### Output:

```
Converged at iteration 125212!
MSE change: 0.0000095367 < 1e-05
------------------------------------------------------------
Final model: FARE = 2.0218 * TRIP_MILES + 0.2012 * TRIP_MINUTES + 2.1681
Final MSE: 13.0103

Model trained successfully!
Used device: cuda
GPU: NVIDIA GeForce RTX 3060 Ti

------------------------------------------------------------
Random 10 predictions:
------------------------------------------------------------
Features: Miles=16.54, Min=25
  Actual: 41.00$, Predicted: 40.63$, Error: 0.37$

Features: Miles=0.80, Min=8
  Actual: 6.50$, Predicted: 5.39$, Error: 1.11$

Features: Miles=0.89, Min=4
  Actual: 5.50$, Predicted: 4.84$, Error: 0.66$

Features: Miles=17.96, Min=31
  Actual: 45.00$, Predicted: 44.81$, Error: 0.19$

Features: Miles=1.18, Min=9
  Actual: 7.50$, Predicted: 6.38$, Error: 1.12$

Features: Miles=9.77, Min=32
  Actual: 30.25$, Predicted: 28.26$, Error: 1.99$

Features: Miles=15.30, Min=20
  Actual: 37.50$, Predicted: 37.21$, Error: 0.29$

Features: Miles=17.60, Min=32
  Actual: 43.50$, Predicted: 44.19$, Error: 0.69$

Features: Miles=1.63, Min=10
  Actual: 9.00$, Predicted: 7.50$, Error: 1.50$

Features: Miles=15.05, Min=33
  Actual: 40.00$, Predicted: 39.26$, Error: 0.74$


Average Error on 10 random samples: 0.87$
RMSE: 3.61$
```

![Data_set_visualization](Images\Figure_1.1.2.png)

#### This means:

* Each additional mile increases the fare by approximately $2.02
* Each additional minute increases the fare by approximately $0.20 (or $12.07 per hour)
* Base fare (intercept) is approximately $2.17
* Root Mean Square Error (RMSE): $3.61
* Average Error on random samples: $0.87

**Note:** This simplified model dramatically outperforms the more complex models (RMSE improved from $4.10 to $3.61). By removing redundant features and focusing on the two most predictive variables, the model achieves excellent accuracy. Sample predictions show remarkable precision, with most errors under $1.50. Demonstrating that **simpler models with better feature engineering outperform complex models with noisy or redundant features**.

**But, we can see that there still is plenty of room for convergence:**

```
MSE change: 0.0000095367 < 1e-05
```

## Model_1.1.2 (Miles + Minutes + Smaller Err)

The same model will not be trained with:

```
err = 0.0000001
alpha = 0.00001
```

**Features:**
* TRIP_MILES
* TRIP_MINUTES

**Label:**
* FARE

**Key points:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Convergence threshold: **1e-07** (MSE change < 0.0000001) - 100× tighter than previous
* Learning rate: 0.00001 (reduced for finer convergence)
* Required 236,018 iterations (88% more than standard threshold)
* Extended training discovered true Chicago base fare ($3.04 vs actual $3.25)

### Output:

```
Converged at iteration 236018!
MSE change: 0.0000000000 < 1e-07
------------------------------------------------------------
Final model: FARE = 2.0218 * TRIP_MILES + 0.1746 * TRIP_MINUTES + 3.0377
Final MSE: 12.2968

Model trained successfully!
Used device: cuda
GPU: NVIDIA GeForce RTX 3060 Ti

------------------------------------------------------------
Random 10 predictions:
------------------------------------------------------------
Features: Miles=2.14, Min=11
  Actual: 8.75$, Predicted: 9.24$, Error: 0.49$

Features: Miles=0.88, Min=6
  Actual: 6.00$, Predicted: 5.86$, Error: 0.14$

Features: Miles=8.06, Min=14
  Actual: 22.25$, Predicted: 21.74$, Error: 0.51$

Features: Miles=1.33, Min=7
  Actual: 7.00$, Predicted: 7.01$, Error: 0.01$

Features: Miles=12.71, Min=20
  Actual: 32.50$, Predicted: 32.14$, Error: 0.36$

Features: Miles=17.91, Min=30
  Actual: 44.50$, Predicted: 44.42$, Error: 0.08$

Features: Miles=18.01, Min=42
  Actual: 45.00$, Predicted: 46.77$, Error: 1.77$

Features: Miles=4.73, Min=20
  Actual: 19.94$, Predicted: 16.17$, Error: 3.77$

Features: Miles=1.44, Min=7
  Actual: 9.00$, Predicted: 7.09$, Error: 1.91$

Features: Miles=6.17, Min=12
  Actual: 17.75$, Predicted: 17.64$, Error: 0.11$


Average Error on 10 random samples: 0.92$
RMSE: 3.51$
```

![Data_set_visualization](Images\Figure_1.1.2err.png)

#### This means:

* Each additional mile increases the fare by approximately $2.02
* Each additional minute increases the fare by approximately $0.17 (or $10.48 per hour)
* Base fare (intercept) is approximately $3.04
* Root Mean Square Error (RMSE): $3.51
* Average Error on random samples: $0.92

**Note:** Tightening the convergence threshold (from 1e-05 to 1e-07) required **88% more iterations** (236,018 vs 125,212) but improved RMSE from $3.61 to $3.51. The extended training allowed the model to fine-tune its parameters: the per-minute coefficient decreased from $0.20 to $0.17, while the base fare increased from $2.17 to **$3.04** - much closer to Chicago's actual $3.25 base fare. Sample predictions demonstrate exceptional accuracy, with 7 out of 10 predictions within $0.51 of actual fares. This shows that **patience in training pays off** - the model discovered the true underlying fare structure with sufficient convergence tolerance.

**MSE change is almost an absolute 0 (0.0000000000), it is known that the taxi fare price is a predifined linear equation, then why is the equation a bit out of its place?**

## Model_NVD (Miles + Minutes)

This model will analyze if there is some sort of difference in fare prices during the day and during the night.

**Features:**
1. Day Model:
    + DAY_TRIP_MILES
    + DAY_TRIP_MINUTES
2. Night Model:
    + NIGHT_TRIP_MILES
    + NIGHT_TRIP_MINUTES

**Label:**
1. Day Model:
    * DAY_FARE
2. Night Model:
    * NIGHT_FARE

**Key points:**

* Automatic gradient computation via `loss.backward()`
* GPU acceleration if CUDA is available (falls back to CPU)
* Convergence threshold: 1e-05 (MSE change < 0.00001)
* Learning rate: 0.0001
* Two separate models trained independently on time-segmented data
* Day dataset: 25,976 samples (6am - 8pm)
* Night dataset: 5,718 samples (8pm - 6am)
* Reveals pricing differences: Night trips charge 29% more per minute

### Output:

```
Both models have converged!

============================================================
FINAL MODELS
============================================================
DAY MODEL: FARE = 1.9863 * MILES + 0.1864 * MINUTES + 2.9136
NIGHT MODEL: FARE = 1.9991 * MILES + 0.2398 * MINUTES + 2.5321
Final Day MSE: 10.7496
Final Night MSE: 19.1927


------------------------------------------------------------
RANDOM SAMPLE PREDICTIONS
------------------------------------------------------------

DAY TRIPS:
  Miles=1.59, Min=8.7 → Actual=$7.75, Pred=$7.70, Error=$0.05
  Miles=12.75, Min=25.1 → Actual=$33.68, Pred=$32.91, Error=$0.77
  Miles=4.60, Min=10.0 → Actual=$14.25, Pred=$13.91, Error=$0.34
  Miles=11.93, Min=26.3 → Actual=$32.00, Pred=$31.51, Error=$0.49
  Miles=17.81, Min=34.8 → Actual=$44.75, Pred=$44.77, Error=$0.02

NIGHT TRIPS:
  Miles=10.46, Min=24.1 → Actual=$27.50, Pred=$29.22, Error=$1.72
  Miles=16.40, Min=23.7 → Actual=$40.50, Pred=$41.00, Error=$0.50
  Miles=18.64, Min=31.9 → Actual=$47.25, Pred=$47.44, Error=$0.19
  Miles=3.60, Min=11.0 → Actual=$13.75, Pred=$12.37, Error=$1.38
  Miles=0.63, Min=3.0 → Actual=$5.00, Pred=$4.51, Error=$0.49

============================================================
MODEL COMPARISON
============================================================
Metric               DAY             NIGHT           DIFF
------------------------------------------------------------
Miles weight         1.9863          1.9991          +0.0128
Minutes weight       0.1864          0.2398          +0.0534
Bias                 2.9136          2.5321          -0.3815
Final MSE            10.7496         19.1927         +8.4431
RMSE                 $3.28           $4.38           $+1.10
```

![Data_set_visualization](Images\nvd.png)

#### This means:

**Day Model (6am - 8pm):**
* Each additional mile increases the fare by approximately $1.99
* Each additional minute increases the fare by approximately $0.19 (or $11.18 per hour)
* Base fare (intercept) is approximately $2.91
* Root Mean Square Error (RMSE): $3.28

**Night Model (8pm - 6am):**
* Each additional mile increases the fare by approximately $2.00
* Each additional minute increases the fare by approximately $0.24 (or $14.39 per hour)
* Base fare (intercept) is approximately $2.53
* Root Mean Square Error (RMSE): $4.38

**Key Differences:**
* **Per-minute rate**: Night trips are **29% more expensive per minute** ($0.24 vs $0.19) - likely due to slower traffic and longer wait times
* **Base fare**: Day trips have a **15% higher base fare** ($2.91 vs $2.53)
* **Accuracy**: Day model performs **79% better** (MSE 10.75 vs 19.19) due to having **4.5× more training data** (25,976 day samples vs 5,718 night samples)
* **Per-mile rate**: Nearly identical (~$2.00), showing consistent distance-based pricing

**Note:** The split model reveals distinct pricing patterns. Night trips show higher variability (RMSE $4.38 vs $3.28) due to factors like airport surcharges, inconsistent traffic, and fewer data points. Sample predictions demonstrate excellent day model accuracy (errors mostly under $0.50) while night predictions show more variance ($0.50-$1.72 errors). This suggests **time-of-day segmentation improves day predictions but struggles with night trips due to limited training data**.

**Thats why the equation is a bit out of place, the night time generates more outliers and cant be really studied because of the smaller amount of data**

## Results Summary

| Model | Features | MSE | RMSE | Avg Error |
|-------|----------|-----|------|-----------|
| 1.1.2 (1e-05) | Miles, Minutes | 13.01 | $3.61 | $0.87 |
| 1.1.2 (1e-07) | Miles, Minutes | 12.30 | $3.51 | $0.92 |
| 1.1.3 | Miles, Seconds, Speed | 16.77 | $4.10 | $2.52 |
| 1.1.5 | Miles, Seconds, Speed, Tips, TipRate | 30.50 | $5.52 | $3.51 |
| Day Model | Miles, Minutes | 10.75 | $3.28 | - |
| Night Model | Miles, Minutes | 19.19 | $4.38 | - |

## Conclusions
### 1. Feature Selection Impact

Adding more features does not always improve prediction quality:

- **Miles + Minutes**: Best overall performance (MSE 12.30-13.01)
- **Adding Speed**: Increased error significantly (MSE 16.77)
- **Adding Tips**: Dramatically worsened predictions (MSE 30.50)

**Why?** Tips are inherently unpredictable - they represent human behavior and are a *consequence* of the fare, not a predictor of it. Including tips creates a circular dependency where the model tries to predict the fare using a value that *depends on* the fare itself.

Speed creates **multicollinearity** issues since it's mathematically derived from distance and time (Speed = Miles/Time). This redundancy confuses the model rather than providing new information, causing the weights to become unstable and the base fare to even turn negative.

### 2. Feature Scaling Matters

Converting TRIP_SECONDS (mean: 1,320) to TRIP_MINUTES (mean: 22) dramatically improved training:

- **Before scaling**: Required learning rate of 0.00001, slow convergence
- **After scaling**: Enabled learning rate of 0.0001 (10× faster), stable convergence
- **Result**: Converged in 125,212 iterations instead of 429,546+ (71% faster)

When features are on vastly different scales (×159 magnitude difference), gradient descent struggles because the loss surface becomes elongated and narrow, causing the optimizer to oscillate and converge slowly.

### 3. Day vs Night Pricing Dynamics

Night trips show **79% higher MSE** (19.19 vs 10.75) due to several factors:

- **Data scarcity**: Only 5,718 night samples vs 25,976 day samples (4.5× less data)
- **Higher variability**: Airport surcharges, late-night premiums, inconsistent traffic patterns
- **Different rate structures**: 
  - Night per-minute rate 29% higher ($0.24 vs $0.19)
  - Day base fare 15% higher ($2.91 vs $2.53)
  - Per-mile rates nearly identical (~$2.00)

This explains why **Model_1.1.2** couldn't perfectly match Chicago's official rates - the combined dataset mixes two distinct pricing regimes. The model compromises between day and night patterns, landing on intermediate coefficients ($2.02/mile, $0.17/min, $3.04 base) that don't exactly match either period.

### 4. Convergence Threshold Trade-offs

Tightening the convergence threshold from 1e-05 to 1e-07:

- **Computational cost**: Required 88% more iterations (236,018 vs 125,212)
- **Accuracy gain**: RMSE improved by only $0.10 (from $3.61 to $3.51)
- **Parameter refinement**: 
  - Per-minute weight: $0.20 → $0.17 (closer to reality)
  - Base fare: $2.17 → $3.04 (much closer to actual $3.25)

**Insight**: Extended training allows the model to escape local minima and discover parameters that better represent the true underlying pricing structure, but with diminishing returns on prediction accuracy.

### 5. The Simplicity Principle

The best performing model used just **2 features** (Miles + Minutes):

| Model Complexity | Features | RMSE | Training Time |
|-----------------|----------|------|---------------|
| Simple | 2 | $3.51 | Fast (125k iter) |
| Medium | 3 | $4.10 | Slow (430k iter) |
| Complex | 5 | $5.52 | Very slow (495k iter) |

**Occam's Razor validated**: When features are redundant, correlated, or introduce noise, simpler models with carefully selected features outperform complex ones. The model doesn't need to know speed if it already knows distance and time - that's just teaching it the same information twice.

### 6. Real-World Validation

Best model coefficients vs Chicago's official rates:

| Component | Official Rate | Model (1e-07) | Error |
|-----------|--------------|---------------|-------|
| Base fare | $3.25 | $3.04 | -6.5% |
| Per mile | $2.25 | $2.02 | -10.2% |
| Per minute | $0.12 | $0.17 | +41.7% |

The model captured the general pricing structure remarkably well, especially considering it learned these patterns purely from data without being told the official rates. The per-minute discrepancy likely reflects:

- Mixed day/night data creating an averaged rate
- Real-world fare variations (surcharges, tolls, tips included in fare field)
- Time-of-day and traffic condition effects not captured by simple time measurement

## Key Differences

| Aspect | NumPy Version | PyTorch Version |
|--------|---------------|-----------------|
| Device | CPU only | CPU or GPU (automatic) |
| Gradient Calculation | Manual formulas | Automatic (loss.backward()) |
| Convergence | Fixed iterations (100) | Dynamic (error threshold) |
| Speed | Baseline | Faster on GPU |
| Learning Rate | 0.001 | Depends |
| Complexity | Simple, educational | Production-ready |