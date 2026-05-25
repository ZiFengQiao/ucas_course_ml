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