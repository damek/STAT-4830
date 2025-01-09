---
layout: course_page
title: Basic Linear Algebra in PyTorch
---

# 1. Basic Linear Algebra in PyTorch

In Lecture 0, we classified spam using word frequencies as vectors. Each email became a point in high-dimensional space, where dimensions represented word counts. Linear algebra revealed the underlying geometry - similar emails clustered together, and a linear boundary separated spam from non-spam.

Now we'll explore how PyTorch implements these operations efficiently. Three key ideas emerge:

1. Tensors extend vectors and matrices to arbitrary dimensions, enabling us to process thousands of emails simultaneously
2. Memory layout and broadcasting determine computational efficiency - critical when processing large datasets
3. SVD reveals structure in data, like discovering which word combinations best distinguish spam

Let's start with vectors and their computational counterpart, tensors.

## Vectors and Tensors: The Foundation

Vectors form the building blocks of data representation. In the spam example, each dimension measured a word's frequency. Here, we'll use temperature readings to build intuition:

```python
# Temperature readings (Celsius)
readings = torch.tensor([22.5, 23.1, 21.8])  # Morning, noon, night
print(readings)  # tensor([22.5000, 23.1000, 21.8000])
```

PyTorch implements vectors as tensors, optimizing the underlying memory and computation:

```python
# Compare two days
morning = torch.tensor([22.5, 23.1, 21.8])  # Yesterday
evening = torch.tensor([21.0, 22.5, 20.9])  # Today
alpha = 0.5  # Averaging weight

# Vector addition: component-wise operation
total = morning + evening  # Parallel computation
print(total)  # tensor([43.5000, 45.6000, 42.7000])

# Scalar multiplication: uniform scaling
weighted = alpha * morning  # Efficient broadcast
print(weighted)  # tensor([11.2500, 11.5500, 10.9000])
```

### Creating Tensors
PyTorch generalizes vectors to n-dimensional arrays. The shape property defines the array structure and guides computation:

```python
# Vector creation methods
temps = torch.tensor([22.5, 23.1, 21.8])     # From data
print(f"Vector shape: {temps.shape}")         # torch.Size([3])

zeros = torch.zeros(3)                        # Initialized
print(f"Zeros shape: {zeros.shape}")          # torch.Size([3])

# Matrix: week of readings
weekly = torch.randn(7, 3)                    # Random normal
print(f"Matrix shape: {weekly.shape}")        # torch.Size([7, 3])
```

### Quick Check 1: Understanding Shapes
```python
morning = torch.tensor([22.5, 23.1, 21.8])
evening = torch.zeros_like(morning)           # Same shape as morning
combined = morning + evening
print(combined.shape, combined)               # Shape preserved in operations
```

### Vector Operations
PyTorch implements these operations:

- Addition: $x + y = [x_1+y_1, x_2+y_2, \ldots, x_n+y_n]$
- Scalar multiplication: $\alpha x = [\alpha x_1, \alpha x_2, \ldots, \alpha x_n]$
- Dot product: $x \cdot y = \sum_{i=1}^n x_iy_i$ measures alignment between vectors
- Vector norm: $\|x\| = \sqrt{x \cdot x} = \sqrt{\sum_{i=1}^n x_i^2}$ measures magnitude

The dot product reveals relationships between vectors. For temperature data, it shows if two days follow similar patterns - high values indicate similar temperature variations. The norm quantifies the overall magnitude of temperature fluctuations.

```python
# Combining sensor readings
day1 = torch.tensor([22.5, 23.1, 21.8])  # Warmer day
day2 = torch.tensor([21.0, 22.5, 20.9])  # Cooler day

# Average readings
avg = (day1 + day2) / 2
print(avg)  # tensor([21.75, 22.80, 21.35])

# Dot product reveals pattern similarity
similarity = torch.dot(day1, day2)
print(f"Similarity: {similarity:.1f}")  # 1447.9: high similarity
print(f"Day 1 magnitude: {torch.norm(day1):.1f}")  # 38.9
print(f"Day 2 magnitude: {torch.norm(day2):.1f}")  # 37.2
```

These numbers reveal the temperature patterns:
1. High dot product (1447.9) shows days follow similar patterns
2. Day 1's larger magnitude (38.9 vs 37.2) confirms it was warmer
3. Small magnitude difference (4.6%) indicates mild temperature variation

The angle between vectors, computed as $\cos(\theta) = \frac{x \cdot y}{\|x\|\|y\|}$, measures pattern similarity independent of magnitude. For these days:
```python
cos_theta = similarity / (torch.norm(day1) * torch.norm(day2))
print(f"Pattern similarity: {cos_theta:.3f}")  # 0.999: nearly identical patterns
```

This near-1 cosine shows the temperature curves are almost identical shapes, just shifted slightly in magnitude.

### Quick Check 2: Vector Norms
```python
# Computing average deviation from mean
readings = torch.tensor([22.5, 23.1, 21.8])
mean = readings.mean()
deviations = readings - mean
magnitude = torch.sqrt(torch.dot(deviations, deviations))
print(f"Average deviation: {magnitude/3:.4f}")  # Average deviation: 0.3067
```

### Memory Layout
Computers store tensors in physical memory as contiguous blocks. Row-major ordering means elements within a row are adjacent:

![Memory Layout](figures/memory_layout.png)

This layout affects performance:
1. Row operations are fast - elements are contiguous in memory, maximizing cache usage
2. Column operations are slower - elements are separated by row stride, causing cache misses
3. Matrix multiplication performance depends on access patterns and memory hierarchy

For temperature data:
```python
# Fast: accessing one day's readings (row)
day_readings = week_temps[0]  # Contiguous memory access

# Slower: accessing one time across days (column)
morning_temps = week_temps[:, 0]  # Strided memory access

# Matrix multiply organizes computation to maximize cache usage
result = torch.mm(week_temps, weights.view(-1, 1))
```

Understanding memory layout helps choose efficient operations:
- Prefer row operations when possible
- Batch similar operations to maximize cache usage
- Consider transposing matrices to align access patterns with memory layout

### Puzzle: Vector Operations
Given temperature readings from two days:
```python
day1 = torch.tensor([22.5, 23.1, 21.8])  # Morning, noon, night
day2 = torch.tensor([21.0, 22.5, 20.9])  # Morning, noon, night
```
1. Compute their similarity (dot product)
2. Find the angle between them: cos(θ) = (u·v)/(‖u‖‖v‖)
3. Interpret: are the temperature patterns similar?

## Matrix Operations

Matrices help analyze multiple days of temperature readings at once:

```python
# One week of temperature readings (7 days × 3 times per day)
week_temps = torch.tensor([
    [22.5, 23.1, 21.8],  # Monday
    [21.0, 22.5, 20.9],  # Tuesday
    [23.1, 24.0, 22.8],  # Wednesday
    [22.8, 23.5, 21.9],  # Thursday
    [21.5, 22.8, 21.2],  # Friday
    [20.9, 21.8, 20.5],  # Saturday
    [21.2, 22.0, 20.8]   # Sunday
])
print(f"Shape: {week_temps.shape}")  # torch.Size([7, 3])
```

### Basic Matrix Operations
For matrices $A$ and $B$ of the same size:

- Addition: 
$$(A + B)_{ij} = a_{ij} + b_{ij}$$ 
combines corresponding elements
- Scalar multiplication: 
$$(\alpha A)_{ij} = \alpha a_{ij}$$
scales all elements
- Mean along dimension: 
$$\text{mean}(A, \text{dim}=0)_j = \frac{1}{m}\sum_{i=1}^m a_{ij}$$
collapses rows

These operations preserve the structure of the data while revealing patterns:

```python
# Last week's temperatures
last_week = torch.tensor([
    [21.5, 22.1, 20.8],  # Morning, noon, night
    [20.0, 21.5, 19.9],
    [22.1, 23.0, 21.8]
])

# This week's temperatures
this_week = torch.tensor([
    [22.5, 23.1, 21.8],
    [21.0, 22.5, 20.9],
    [23.1, 24.0, 22.8]
])

# Temperature change shows consistent warming
temp_change = this_week - last_week
print("Temperature changes:")
print(temp_change)
# tensor([[1., 1., 1.],  # Uniform 1°C increase
#         [1., 1., 1.],  # across all times
#         [1., 1., 1.]]) # and days

# Average temperatures reveal daily pattern
daily_means = this_week.mean(dim=0)
print("\nAverage temperatures:")
print(daily_means)  # tensor([22.2000, 23.2000, 21.8333])
#                     Morning   Noon     Night
```

The outputs reveal clear patterns:
1. Consistent 1°C warming trend across all measurements
2. Daily cycle: warmest at noon (23.2°C), coolest at night (21.8°C)
3. Morning temperatures (22.2°C) between extremes

### Quick Check: Matrix Operations
```python
# What's the output shape and meaning?
week_temps = torch.randn(7, 3)       # Week of readings
day_weights = torch.ones(7) / 7      # Equal weights for each day
weighted_means = ???
```

### Matrix Multiplication
Matrix multiplication combines information across dimensions. For matrices $A$ and $B$:
$c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}$

For matrix-vector multiplication $(Ax = b)$:
$b_i = \sum_{j=1}^n a_{ij}x_j$

This operation is fundamental because:
- Each output combines an entire row with a column
- The operation preserves linear relationships
- Computation parallelizes efficiently

![Matrix Multiplication](figures/matrix_multiply.png)

```python
# Temperature readings and weights
temps = torch.tensor([
    [22.5, 23.1, 21.8],  # Day 1: morning, noon, night
    [21.0, 22.5, 20.9],  # Day 2: morning, noon, night
    [23.1, 24.0, 22.8]   # Day 3: morning, noon, night
])

weights = torch.tensor([0.5, 0.3, 0.2])  # Recent days matter more
```

Matrix multiplication computes weighted combinations:
```
Day 1: [22.5, 23.1, 21.8] × 0.5 = [11.25, 11.55, 10.90]  # 50% influence
Day 2: [21.0, 22.5, 20.9] × 0.3 = [ 6.30,  6.75,  6.27]  # 30% influence
Day 3: [23.1, 24.0, 22.8] × 0.2 = [ 4.62,  4.80,  4.56]  # 20% influence
                                  [22.17, 23.10, 21.73] Final averages
```

PyTorch optimizes this computation:
```python
# Efficient matrix-vector multiply
weighted_means = torch.mv(temps, weights)  # Uses BLAS
print("Weighted averages per time:")
print(weighted_means)  # tensor([22.5400, 21.4300, 23.3100])
#                        Morning   Noon     Night
```

The weighted averages reveal:
1. Morning temperatures stable around 22.5°C
2. Noon shows most variation (21.4°C)
3. Night temperatures highest recently (23.3°C)

This analysis weights recent days more heavily, capturing current trends rather than long-term averages.

### Broadcasting
Broadcasting generalizes operations between tensors of different shapes. It extends the mathematical concept of scalar multiplication to more general shape-compatible operations. For a vector $v \in \mathbb{R}^n$ and matrix $A \in \mathbb{R}^{m \times n}$:

$$(A * v)_{ij} = a_{ij} * v_j$$

This operation implicitly replicates the vector across rows, but without copying memory. The computational advantages are significant:
- Memory efficient: No need to materialize the replicated tensor
- Cache friendly: Access pattern matches memory layout
- Parallelizable: Each output element computed independently

```python
# Temperature readings across days
day_temps = torch.tensor([
    [22.5, 23.1, 21.8],  # Day 1: morning, noon, night
    [21.0, 22.5, 20.9],  # Day 2
    [23.1, 24.0, 22.8]   # Day 3
])

# Sensor calibration factors (per time of day)
calibration = torch.tensor([1.02, 0.98, 1.01])

# Broadcasting: each time slot gets its calibration
calibrated = day_temps * calibration  # Shape [3,3] * [3] -> [3,3]
print("Original vs Calibrated:")
print(day_temps[0])      # Before calibration
print(calibrated[0])     # After calibration
```

![Broadcasting figure](figures/broadcasting.png)

Broadcasting rules follow mathematical intuition:
1. Trailing dimensions must match exactly
2. Missing dimensions are implicitly size 1
3. Size 1 dimensions stretch to match larger dimensions

This enables concise, efficient code:
```python
# Temperature adjustments
base = torch.tensor([[22.5, 23.1, 21.8],    # Base readings
                    [21.0, 22.5, 20.9]])
offset = torch.tensor([0.5, 0.0, -0.5])     # Per-time adjustments
scale = torch.tensor([1.02, 0.98]).view(-1, 1)  # Per-day scaling

# Multiple broadcasts in one expression
adjusted = scale * (base + offset)  # Combines both adjustments
print("Adjusted shape:", adjusted.shape)
```

### Quick Check: Broadcasting
```python
temps = torch.tensor([[22.5, 23.1, 21.8],
                     [21.0, 22.5, 20.9]])
offset = torch.tensor([0.5, 0.0, -0.5])

adjusted = temps + offset  # What's the shape?
```

### Puzzle: Temperature Analysis
Given a week of readings and calibrations:
```python
week_temps = torch.tensor([
    [22.5, 23.1, 21.8],  # Each row: morning, noon, night
    [21.0, 22.5, 20.9],
    [23.1, 24.0, 22.8]
])
calibration = torch.tensor([1.02, 0.98, 1.01])  # Per sensor
importance = torch.tensor([0.5, 0.3, 0.2])      # Per day
```

1. Apply sensor calibrations
2. Compute weighted average for each time
3. Find which time has most stable temperatures

## Finding Patterns with SVD

SVD (Singular Value Decomposition) reveals structure in data by factoring a matrix $A$ into orthogonal components: $A = U\Sigma V^T$. Let's see how this helps analyze our spam data:

```python
# Feature matrix: 10 emails × 5 features
# Features: exclamation_count, urgent_words, suspicious_links, caps_ratio, length

U, S, V = torch.linalg.svd(X)
print("Singular values:", S)
# tensor([11.3077,  1.4219,  0.5334,  0.2697,  0.0496])

print("Energy per pattern:", 100 * S**2 / torch.sum(S**2), "%")
# tensor([98.17, 1.55, 0.22, 0.06, 0.00]) %
```

The decomposition reveals:
1. Feature patterns: Which features occur together ($V^T$)
2. Email patterns: How emails combine features ($U$)
3. Pattern strengths: How important each pattern is ($\Sigma$)

Looking at the first pattern:
```python
print("Feature pattern:", V[0])
# tensor([-0.9767, -0.0892, -0.1925, -0.0136, -0.0289])

print("Email pattern:", U[:, 0])
# tensor([-0.3710, -0.6300, -0.4493, -0.3630, -0.3630,
#         -0.0004, -0.0004, -0.0004, -0.0082, -0.0004])
```

This dominant pattern shows:
1. Features: Exclamation marks (-0.98) and suspicious links (-0.19) cluster together
2. Emails: Clear separation between spam (large negative values) and legitimate (near zero)
3. Strength: First singular value (11.3) captures 98.17% of data variation

### Measuring Pattern Quality
To quantify how well these patterns represent our data, we need a way to measure matrix size. The Frobenius norm extends our vector norm concept:

For vectors: $\|x\| = \sqrt{\sum_i x_i^2}$ measures total magnitude
For matrices: 
$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2}$$
measures total variation

This norm is natural because:
1. It treats matrices as vectors in $\mathbb{R}^{mn}$
2. It's rotationally invariant: $\|A\|_F = \|UAV^T\|_F$
3. It decomposes via singular values: $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$

For our spam features:
```python
# Three equivalent ways to measure total variation
print(f"Element-wise: {torch.sqrt((X**2).sum()):.1f}")  # 11.4
print(f"As vector:    {X.view(-1).norm():.1f}")        # 11.4
print(f"From SVD:     {S.norm():.1f}")                 # 11.4
```

### Building Low-Rank Approximations
SVD expresses matrices as sums of rank-1 patterns:

$A = \sum_{i=1}^n \sigma_i u_i v_i^T$

Each term contributes:
- Pattern: Outer product $u_iv_i^T$ (rank 1 matrix)
- Strength: Singular value $\sigma_i$ (importance weight)
- Structure: $u_i$ and $v_i$ are orthonormal (independent patterns)

We can approximate $A$ using just the first $k$ terms:

$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T = U_k\Sigma_kV_k^T$

For our spam data:
```python
def reconstruct(k):
    return (U[:, :k] @ torch.diag(S[:k]) @ V[:k, :])

# Compare reconstructions
original = X[0]  # First email features
rank1 = reconstruct(1)[0]  # Using only top pattern
rank2 = reconstruct(2)[0]  # Using top two patterns

print("Original:", original)
print("Rank 1:", rank1)
print("Rank 2:", rank2)
```

### The Eckart-Young-Mirsky Theorem
This truncation isn't just simple - it's optimal. The theorem states that for any rank-$k$ matrix $B$:

$$\|A - A_k\|_F \leq \|A - B\|_F$$

Moreover, the error is exactly the discarded singular values:

$$\|A - A_k\|_F^2 = \sum_{i=k+1}^n \sigma_i^2$$

For our spam data:
```python
def approx_error(k):
    """Compute relative error for rank-k approximation."""
    truncated = S[k:].norm()**2  # Squared Frobenius norm of discarded values
    total = S.norm()**2         # Total squared Frobenius norm
    return torch.sqrt(truncated/total)

# Show error decreases with rank
for k in range(1, 4):
    print(f"Rank {k} relative error: {approx_error(k):.2e}")
# Rank 1: 1.35e-01  (13.5% error)
# Rank 2: 5.03e-02  (5% error)
# Rank 3: 2.75e-02  (2.75% error)
```

The rapid decay of singular values (11.3 → 1.4 → 0.5) explains why low-rank approximations work well.

### Pattern Analysis Example: Checkerboard
Some patterns require multiple components. Consider this checkerboard:

```python
pattern_image = torch.tensor([
    [200,  50,  200,  50],  # Alternating bright-dark
    [50,  200,  50,  200],  # Opposite pattern
    [200,  50,  200,  50],  # First pattern repeats
    [50,  200,  50,  200]   # Second pattern repeats
], dtype=torch.float)

U, S, V = torch.linalg.svd(pattern_image)
print("Singular values:", S)
# tensor([5.0000e+02, 3.0000e+02, 8.8238e-06, 2.4984e-06])

print("Energy per pattern:", 100 * S**2 / torch.sum(S**2), "%")
# tensor([7.3529e+01, 2.6471e+01, 2.2900e-14, 1.8359e-15]) %
```

The SVD reveals two essential patterns:
1. First pattern (73.5%):
   - $u_1$: Vertical pattern [-0.5, -0.5, -0.5, -0.5]
   - $v_1$: Horizontal pattern [-0.5, -0.5, -0.5, -0.5]
   - Product creates uniform intensity

2. Second pattern (26.5%):
   - $u_2$: Alternating vertical [0.5, -0.5, 0.5, -0.5]
   - $v_2$: Alternating horizontal [0.5, -0.5, 0.5, -0.5]
   - Product creates checkerboard

Both patterns are necessary because:
1. First pattern sets overall intensity level
2. Second pattern creates alternation
3. Both patterns needed for reconstruction
4. Remaining patterns are numerical noise (~10⁻¹⁴%)

These principles extend to high-dimensional data:
- Natural images: Rapidly decaying singular values enable compression
- Text data: Word co-occurrence patterns reveal semantics
- Time series: Temporal patterns emerge at different scales

## Summary

PyTorch implements linear algebra efficiently through three key mechanisms:

1. Tensors: Flexible N-dimensional arrays
   - Natural representation for data (temperature readings, images)
   - Automatic differentiation for optimization (later)
   - Hardware acceleration (CPU/GPU)

2. Broadcasting: Implicit shape matching
   - Eliminates explicit loops (sensor calibration)
   - Memory efficient (no copies needed)
   - Parallelizes naturally

3. SVD: Pattern discovery and compression
   - Optimal low-rank approximation (Eckart-Young-Mirsky)
   - Separates signal from noise
   - Reveals underlying structure

### Essential Operations
```python
# Data representation
x = torch.tensor([1, 2, 3])          # Vector (like temperature readings)
y = torch.zeros_like(x)              # Initialize (like sensor baseline)
z = torch.randn(3, 3)               # Random sampling
A = z.float()                       # Type conversion for computation

# Core computations
b = x + 2                           # Broadcasting (calibration)
c = torch.dot(x, x)                 # Inner product (pattern similarity)
d = A @ x                           # Matrix multiply (weighted average)
e = torch.mean(A, dim=0)            # Reduction (daily averages)

# Structure manipulation
f = A.t()                           # Transpose (change access pattern)
g = A.view(-1)                      # Reshape (flatten for computation)
h = A[:, :2]                        # Slice (select time window)

# Pattern analysis
U, S, V = torch.linalg.svd(A)       # Decomposition (find patterns)
norm = torch.norm(x)                # Vector norm (measure magnitude)
```

These operations combine to solve real problems:
1. Data processing: Broadcasting + reduction for sensor calibration
2. Pattern matching: Dot products + norms for similarity detection
3. Dimensionality reduction: SVD for compression and denoising
4. Feature extraction: Singular vectors capture dominant patterns

The key is choosing the right operation for each task:
- Use broadcasting for element-wise operations
- Use matrix multiply for weighted combinations
- Use SVD for pattern discovery and compression
