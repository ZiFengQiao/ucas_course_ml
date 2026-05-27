import numpy as np
import torch

from simplecnn import SimpleCNN


def _log_array(name, arr):
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
    )


def _log_close(name, a, b):
    diff = np.abs(a - b)
    print(
        f"{name}: max_abs_diff={diff.max():.6e}, "
        f"mean_abs_diff={diff.mean():.6e}"
    )

def test_conv2d_vs_torch():
    np.random.seed(0)

    x = np.random.randn(8, 8, 3, 2)

    w = np.random.randn(3, 3, 3, 4)

    b = np.random.randn(4)

    print("[test_conv2d_vs_torch] input setup")
    _log_array("x", x)
    _log_array("w", w)
    _log_array("b", b)

    y_my = SimpleCNN.conv2d(x, w, b)

    x_t = torch.tensor(
        np.transpose(x, (3, 2, 0, 1)),
        dtype=torch.float64
    )

    w_t = torch.tensor(
        np.transpose(w, (3, 2, 0, 1)),
        dtype=torch.float64
    )

    b_t = torch.tensor(
        b,
        dtype=torch.float64
    )

    y_t = torch.nn.functional.conv2d(
        x_t,
        w_t,
        b_t
    )

    y_t = y_t.detach().numpy()

    y_t = np.transpose(y_t, (2, 3, 1, 0))

    print("[test_conv2d_vs_torch] outputs")
    _log_array("y_my", y_my)
    _log_array("y_t", y_t)
    _log_close("y_my vs y_t", y_my, y_t)

    assert np.allclose(y_my, y_t, atol=1e-8)

def test_avgpool_vs_torch():
    np.random.seed(0)

    x = np.random.randn(8, 8, 3, 2)

    print("[test_avgpool_vs_torch] input setup")
    _log_array("x", x)

    y_my = SimpleCNN.avg_pool2d(x)

    x_t = torch.tensor(
        np.transpose(x, (3, 2, 0, 1)),
        dtype=torch.float64
    )

    y_t = torch.nn.functional.avg_pool2d(
        x_t,
        kernel_size=2
    )

    y_t = y_t.detach().numpy()

    y_t = np.transpose(y_t, (2, 3, 1, 0))

    print("[test_avgpool_vs_torch] outputs")
    _log_array("y_my", y_my)
    _log_array("y_t", y_t)
    _log_close("y_my vs y_t", y_my, y_t)

    assert np.allclose(y_my, y_t, atol=1e-8)

def test_relu_vs_torch():
    x = np.random.randn(100)

    print("[test_relu_vs_torch] input setup")
    _log_array("x", x)

    y_my = SimpleCNN.relu(x)

    y_t = torch.relu(
        torch.tensor(x)
    ).numpy()

    print("[test_relu_vs_torch] outputs")
    _log_array("y_my", y_my)
    _log_array("y_t", y_t)
    _log_close("y_my vs y_t", y_my, y_t)

    assert np.allclose(y_my, y_t)

def test_softmax_vs_torch():
    x = np.random.randn(16, 10)

    print("[test_softmax_vs_torch] input setup")
    _log_array("x", x)

    y_my = SimpleCNN.softmax(x)

    y_t = torch.softmax(
        torch.tensor(x),
        dim=1
    ).numpy()

    print("[test_softmax_vs_torch] outputs")
    _log_array("y_my", y_my)
    _log_array("y_t", y_t)
    _log_close("y_my vs y_t", y_my, y_t)

    assert np.allclose(y_my, y_t, atol=1e-8)

def test_conv_backward_vs_torch():
    np.random.seed(0)

    x = np.random.randn(8, 8, 3, 2)

    w = np.random.randn(3, 3, 3, 4)

    dout = np.random.randn(6, 6, 4, 2)

    print("[test_conv_backward_vs_torch] input setup")
    _log_array("x", x)
    _log_array("w", w)
    _log_array("dout", dout)

    dx_my, dw_my, db_my = SimpleCNN.conv2d_backward(
        x,
        w,
        dout
    )

    x_t = torch.tensor(
        np.transpose(x, (3, 2, 0, 1)),
        dtype=torch.float64,
        requires_grad=True
    )

    w_t = torch.tensor(
        np.transpose(w, (3, 2, 0, 1)),
        dtype=torch.float64,
        requires_grad=True
    )

    y = torch.nn.functional.conv2d(
        x_t,
        w_t
    )

    dout_t = torch.tensor(
        np.transpose(dout, (3, 2, 0, 1)),
        dtype=torch.float64
    )

    y.backward(dout_t)

    dx_t = np.transpose(
        x_t.grad.numpy(),
        (2, 3, 1, 0)
    )

    dw_t = np.transpose(
        w_t.grad.numpy(),
        (2, 3, 1, 0)
    )

    print("[test_conv_backward_vs_torch] outputs")
    _log_array("dx_my", dx_my)
    _log_array("dx_t", dx_t)
    _log_close("dx_my vs dx_t", dx_my, dx_t)
    _log_array("dw_my", dw_my)
    _log_array("dw_t", dw_t)
    _log_close("dw_my vs dw_t", dw_my, dw_t)
    _log_array("db_my", db_my)

    assert np.allclose(dx_my, dx_t, atol=1e-8)

def test_avgpool_backward_vs_torch():
    np.random.seed(0)

    x = np.random.randn(8, 8, 3, 2)

    dout = np.random.randn(4, 4, 3, 2)

    print("[test_avgpool_backward_vs_torch] input setup")
    _log_array("x", x)
    _log_array("dout", dout)

    dx_my = SimpleCNN.avg_pool2d_backward(
        dout,
        x.shape
    )

    x_t = torch.tensor(
        np.transpose(x, (3, 2, 0, 1)),
        dtype=torch.float64,
        requires_grad=True
    )

    y = torch.nn.functional.avg_pool2d(
        x_t,
        2
    )

    dout_t = torch.tensor(
        np.transpose(dout, (3, 2, 0, 1)),
        dtype=torch.float64
    )

    y.backward(dout_t)

    dx_t = np.transpose(
        x_t.grad.numpy(),
        (2, 3, 1, 0)
    )

    print("[test_avgpool_backward_vs_torch] outputs")
    _log_array("dx_my", dx_my)
    _log_array("dx_t", dx_t)
    _log_close("dx_my vs dx_t", dx_my, dx_t)

    assert np.allclose(dx_my, dx_t, atol=1e-8)


def test_full_model_vs_torch():
    """用完全相同的参数，比对 NumPy SimpleCNN 与 PyTorch SimpleCNN
    的前向传播输出和反向传播梯度。"""
    from simplecnn_ref import SimpleCNN as SimpleCNNTorch

    # ---- 1. 创建 NumPy 模型，固定随机种子 ----
    np.random.seed(42)
    cnn_np = SimpleCNN()

    # ---- 2. 创建 PyTorch 模型，并从 NumPy 模型拷贝参数 ----
    cnn_t = SimpleCNNTorch().to(torch.float64)

    # conv1: (5, 5, 1, 6) -> (6, 1, 5, 5)
    cnn_t.conv1.weight.data = torch.tensor(
        np.transpose(cnn_np.conv1['weights'].value, (3, 2, 0, 1)).copy(),
        dtype=torch.float64,
    )
    cnn_t.conv1.bias.data = torch.tensor(
        cnn_np.conv1['bias'].value.copy(), dtype=torch.float64,
    )

    # conv2: (5, 5, 6, 12) -> (12, 6, 5, 5)
    cnn_t.conv2.weight.data = torch.tensor(
        np.transpose(cnn_np.conv2['weights'].value, (3, 2, 0, 1)).copy(),
        dtype=torch.float64,
    )
    cnn_t.conv2.bias.data = torch.tensor(
        cnn_np.conv2['bias'].value.copy(), dtype=torch.float64,
    )

    # fc: (192, 10) -> (10, 192)
    cnn_t.fc.weight.data = torch.tensor(
        cnn_np.fc['weights'].value.T.copy(), dtype=torch.float64,
    )
    cnn_t.fc.bias.data = torch.tensor(
        cnn_np.fc['bias'].value.copy(), dtype=torch.float64,
    )

    # ---- 3. 生成相同的输入数据和标签 ----
    np.random.seed(123)
    x_np = np.random.randn(4, 28, 28).astype(np.float64)
    y_true = np.array([3, 7, 1, 5], dtype=int)

    # ---- 4. 前向传播比对 ----
    print("\n[test_full_model_vs_torch] ======== Forward ========")
    y_np = cnn_np.forward(x_np.copy())

    x_t = torch.tensor(x_np, dtype=torch.float64).unsqueeze(1)  # (N, 1, 28, 28)
    logits_t = cnn_t(x_t)
    y_t = torch.softmax(logits_t, dim=1).detach().numpy()

    _log_array("y_np", y_np)
    _log_array("y_t", y_t)
    _log_close("forward output", y_np, y_t)
    assert np.allclose(y_np, y_t, atol=1e-8), "Forward output mismatch"

    # ---- 5. 反向传播比对 ----
    print("\n[test_full_model_vs_torch] ======== Backward ========")

    # NumPy backward
    loss_np = cnn_np.backward(y_true)

    # PyTorch backward
    cnn_t.zero_grad()
    loss_t = torch.nn.functional.cross_entropy(
        logits_t, torch.tensor(y_true, dtype=torch.long)
    )
    loss_t.backward()

    print(f"loss_np: {loss_np:.8f}, loss_t: {loss_t.item():.8f}")
    assert np.allclose(loss_np, loss_t.item(), atol=1e-8), "Loss mismatch"

    # 比对每一组参数的梯度
    # conv1 weights: PyTorch (6,1,5,5) -> NumPy (5,5,1,6)
    dw1_t = np.transpose(cnn_t.conv1.weight.grad.numpy(), (2, 3, 1, 0))
    _log_close("conv1.weight.grad", cnn_np.conv1['weights'].grad, dw1_t)
    assert np.allclose(cnn_np.conv1['weights'].grad, dw1_t, atol=1e-8)

    _log_close("conv1.bias.grad", cnn_np.conv1['bias'].grad, cnn_t.conv1.bias.grad.numpy())
    assert np.allclose(cnn_np.conv1['bias'].grad, cnn_t.conv1.bias.grad.numpy(), atol=1e-8)

    # conv2 weights: PyTorch (12,6,5,5) -> NumPy (5,5,6,12)
    dw2_t = np.transpose(cnn_t.conv2.weight.grad.numpy(), (2, 3, 1, 0))
    _log_close("conv2.weight.grad", cnn_np.conv2['weights'].grad, dw2_t)
    assert np.allclose(cnn_np.conv2['weights'].grad, dw2_t, atol=1e-8)

    _log_close("conv2.bias.grad", cnn_np.conv2['bias'].grad, cnn_t.conv2.bias.grad.numpy())
    assert np.allclose(cnn_np.conv2['bias'].grad, cnn_t.conv2.bias.grad.numpy(), atol=1e-8)

    # fc weights: PyTorch (10,192) -> NumPy (192,10)
    dfc_t = cnn_t.fc.weight.grad.numpy().T
    _log_close("fc.weight.grad", cnn_np.fc['weights'].grad, dfc_t)
    assert np.allclose(cnn_np.fc['weights'].grad, dfc_t, atol=1e-8)

    _log_close("fc.bias.grad", cnn_np.fc['bias'].grad, cnn_t.fc.bias.grad.numpy())
    assert np.allclose(cnn_np.fc['bias'].grad, cnn_t.fc.bias.grad.numpy(), atol=1e-8)

    print("\n[test_full_model_vs_torch] All checks passed!")