# Work Progress - simplecnn.py Debugging

## Problem
Training loss stuck at ~2.3 (ln(10), random guessing baseline for 10 classes). Operators pass unit tests individually, but full training doesn't work.

## Root Cause Found
**Reshape scrambles batch data** between conv/pool layers and FC layer.

`pool2` output shape is `(H=4, W=4, C=12, N=batch_size)`. In C-order, the last axis (N) varies fastest. Direct `reshape(N, -1)` causes each row of the FC input matrix to contain features from **ALL batch samples at a single spatial position**, instead of all features for a single sample.

## Fix Applied
Three changes in `simplecnn.py`:

### 1. Forward pass (line ~252)
```python
# OLD (buggy):
output = output.reshape(output.shape[3], -1)

# NEW (fixed):
output = np.transpose(output, (3, 0, 1, 2))  # (N, H, W, C)
output = output.reshape(output.shape[0], -1)  # (N, H*W*C)
```

### 2. Backward - FC weight gradient (line ~290)
```python
# OLD:
pool2_flat = pool2_out.reshape(N, -1)

# NEW:
pool2_flat = np.transpose(pool2_out, (3, 0, 1, 2)).reshape(N, -1)
```

### 3. Backward - gradient unflattening from FC to pool2 (line ~299)
```python
# OLD:
d = d.reshape(self.pool2['output'].shape)  # (N, 192) -> (4, 4, 12, N)

# NEW:
d = d.reshape(N, 4, 4, 12)  # (N, 192) -> (N, H, W, C)
d = np.transpose(d, (1, 2, 3, 0))  # (H, W, C, N)
```

## Verification
- All 6 unit tests in `test_simplecnn.py` pass
- Gradient check (numerical vs analytical) passes with ~1e-7 relative error
- Loss now decreases below 2.3 (was stuck there before)
- Test accuracy improves above random baseline

## Remaining Note
Weight initialization uses `np.random.randn` (std=1) without scaling, causing softmax saturation and slow initial convergence. This is not a bug but is why loss starts very high (~25) and takes many epochs to drop below 2.3. Consider Xavier/He initialization for faster training, but not required for correctness.
