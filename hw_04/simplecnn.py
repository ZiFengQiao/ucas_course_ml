#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
File Name     : simplecnn.py
Author        : 王峤宇
Created Date  : 5/25/26
Version       : 1.0
Description   :
------------------------------------------------------------
Usage Example:
    python your_script_name.py [options]
Notes:
    - 请在此处补充代码使用说明、输入输出格式等
------------------------------------------------------------
"""
import numpy as np

# def softmax(z):
#     exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定
#     return exp_z / np.sum(exp_z, axis=1, keepdims=True)
#
# 加载MNIST数据集
def load_mnist():
    with open('mnist.npz', 'rb') as f:
        data = np.load(f)
        train_x, train_y = data['x_train'], data['y_train']
        test_x, test_y = data['x_test'], data['y_test']
    return train_x, train_y, test_x, test_y

# 对卷积层的参数，其本身形如conv1['weights'].value
# 同时具有梯度grad，可通过如
# conv1['weights'].grad = your_calculation
# 对梯度进行保存
class nn_Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

# CNN网络结构类
class SimpleCNN:
    def __init__(self):

        self.conv1 = {
            'weights': nn_Parameter(np.random.randn(5, 5, 1, 6)),
            'bias': nn_Parameter(np.zeros(6)),
            'output': np.zeros((6, 24, 24))
        }

        self.pool1 = {
            'weights': np.ones((2, 2)) / 4,
            'output': np.zeros((6, 12, 12))
        }

        self.conv2 = {
            'weights': nn_Parameter(np.random.randn(5, 5, 6, 12)),
            'bias': nn_Parameter(np.zeros(12)),
            'output': np.zeros((12, 8, 8))
        }

        self.pool2 = {
            'weights': np.ones((2, 2)) / 4,
            'output': np.zeros((12, 4, 4))
        }

        self.fc = {
            'weights': nn_Parameter(np.random.randn(12 * 4 * 4, 10)),
            'bias': nn_Parameter(np.zeros(10)),
            'output': np.zeros((10))
        }

    @staticmethod
    def conv2d(input, filters, bias, stride=1, padding=0):
        # input   : N x C_in x H x W
        # filters : F_h x F_w x C_in x C_out
        # bias    : C_out

        N, C_in, H, W = input.shape
        F_h, F_w, _, C_out = filters.shape

        if padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
            _, _, H, W = input.shape

        H_out = (H - F_h) // stride + 1
        W_out = (W - F_w) // stride + 1

        res = np.zeros((N, C_out, H_out, W_out))

        for n in range(N):
            for c_in in range(C_in):
                _region = input[n, c_in, :, :]
                for c_out in range(C_out):
                    _kernel = filters[:, :, c_in, c_out]
                    for i in range(0, H_out):
                        for j in range(0, W_out):
                            h_start = i * stride
                            w_start = j * stride
                            res[n, c_out, i, j] += np.sum(
                                _region[h_start:h_start + F_h, w_start:w_start + F_w] * _kernel
                            )
        res += bias.reshape(1, C_out, 1, 1)
        return res

    @staticmethod
    def avg_pool2d(input, kernel_size=2):
        # input : N x C x H x W
        N, C, H, W = input.shape
        H_out = (H - kernel_size) // kernel_size + 1
        W_out = (W - kernel_size) // kernel_size + 1

        res = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                _region = input[n, c, :, :]
                for i in range(0, H_out):
                    for j in range(0, W_out):
                        h_start = i * kernel_size
                        w_start = j * kernel_size
                        res[n, c, i, j] = np.mean(
                            _region[h_start:h_start + kernel_size, w_start:w_start + kernel_size]
                        )
        return res

    @staticmethod
    def avg_pool2d_backward(dout, input_shape, kernel_size=2):
        # dout : N x C x H_out x W_out
        # input_shape : N x C x H x W
        N, C, _, _ = input_shape
        _, _, H_out, W_out = dout.shape
        dx = np.zeros(input_shape)
        scale = 1.0 / (kernel_size * kernel_size)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * kernel_size
                        w_start = j * kernel_size
                        dx[n, c, h_start:h_start + kernel_size, w_start:w_start + kernel_size] += dout[n, c, i, j] * scale
        return dx

    @staticmethod
    def conv2d_backward(input, filters, d_out, stride=1, padding=0):
        # input   : N x C_in x H x W
        # filters : F_h x F_w x C_in x C_out
        # d_out   : N x C_out x H_out x W_out

        N, C_in, _, _ = input.shape
        F_h, F_w, _, C_out = filters.shape
        _, _, H_out, W_out = d_out.shape

        if padding > 0:
            input_pad = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        else:
            input_pad = input

        d_input_pad = np.zeros_like(input_pad)
        d_filters = np.zeros_like(filters)
        d_bias = np.zeros((C_out,))

        for n in range(N):
            for k in range(C_out):
                d_bias[k] += np.sum(d_out[n, k, :, :])
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        patch = input_pad[n, :, h_start:h_start + F_h, w_start:w_start + F_w]  # C_in x F_h x F_w
                        d_filters[:, :, :, k] += patch.transpose(1, 2, 0) * d_out[n, k, i, j]
                        d_input_pad[n, :, h_start:h_start + F_h, w_start:w_start + F_w] += filters[:, :, :, k].transpose(2, 0, 1) * d_out[n, k, i, j]

        if padding > 0:
            d_input = d_input_pad[:, :, padding:-padding, padding:-padding]
        else:
            d_input = d_input_pad

        return d_input, d_filters, d_bias

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        # x : N x C
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # 前向传播函数 请编写其中的内容
    def forward(self, x):
        # 输入数据是 N x H x W 的形式，N是批次大小，H和W是图像的高度和宽度
        # 卷积层和池化层的输入输出都是 N x C x H x W 的形式
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        inputs = x[:, np.newaxis, :, :]    # N x 1 x H x W

        # conv1 -> relu1 -> pool1 -> conv2 -> relu2 -> pool2 -> fc -> softmax
        self.conv1['input'] = inputs
        output = self.relu(self.conv2d(inputs, self.conv1['weights'].value, self.conv1['bias'].value))
        self.conv1['output'] = output  # N x 6 x 24 x 24

        self.pool1['input'] = output
        output = self.avg_pool2d(output, kernel_size=2)
        self.pool1['output'] = output  # N x 6 x 12 x 12

        self.conv2['input'] = output
        output = self.relu(self.conv2d(output, self.conv2['weights'].value, self.conv2['bias'].value))
        self.conv2['output'] = output  # N x 12 x 8 x 8

        self.pool2['input'] = output
        output = self.avg_pool2d(output, kernel_size=2)
        self.pool2['output'] = output  # N x 12 x 4 x 4

        # fc: 展平 N x C x H x W -> N x (C*H*W)
        self.fc['input'] = output
        N = output.shape[0]
        output = output.reshape(N, -1)
        output = output @ self.fc['weights'].value + self.fc['bias'].value
        output = self.softmax(output)

        self.fc['output'] = output
        return output

    # 后向传播函数 请编写其中的内容
    def backward(self, y_true):
        # y_true : N x 1
        # 前向: conv1 -> relu1 -> pool1 -> conv2 -> relu2 -> pool2 -> fc -> softmax

        fc_output = self.fc['output']
        y_true = np.asarray(y_true).reshape(-1)
        if y_true.size == 1 and fc_output.ndim == 2 and fc_output.shape[0] > 1:
            y_true = np.repeat(y_true, fc_output.shape[0])
        y_true = y_true.astype(int)
        N = y_true.shape[0]

        # softmax + cross-entropy的梯度计算
        loss = -np.mean(
            np.log(
                fc_output[np.arange(N), y_true] + 1e-12
            )
        )

        d = fc_output.copy()
        d[np.arange(N), y_true] -= 1
        d /= N

        # ----------------------------------------------------------
        # softmax -> FC
        pool2_out = self.pool2['output']  # N x 12 x 4 x 4
        pool2_flat = pool2_out.reshape(N, -1)  # N x (12*4*4)
        self.fc['weights'].grad = pool2_flat.T @ d
        self.fc['bias'].grad = np.sum(d, axis=0)

        d = d @ self.fc['weights'].value.T  # N x (12*4*4)

        # ----------------------------------------------------------
        # softmax -> FC -> pool2
        d = d.reshape(N, 12, 4, 4)  # N x C x H x W

        # pool2 -> relu2
        d = self.avg_pool2d_backward(d, self.pool2['input'].shape, kernel_size=2)
        d *= self.relu_derivative(self.conv2['output'])

        # relu2 -> conv2
        d, d_w2, d_b2 = self.conv2d_backward(self.conv2['input'], self.conv2['weights'].value, d)
        self.conv2['weights'].grad = d_w2
        self.conv2['bias'].grad = d_b2

        # conv2 -> pool1
        d = self.avg_pool2d_backward(d, self.pool1['input'].shape, kernel_size=2)
        d *= self.relu_derivative(self.conv1['output'])

        # pool1 -> conv1
        d, d_w1, d_b1 = self.conv2d_backward(self.conv1['input'], self.conv1['weights'].value, d)
        self.conv1['weights'].grad = d_w1
        self.conv1['bias'].grad = d_b1

        return loss

    # 更新权重函数 请编写其中的内容
    def update(self, learning_rate):
        self.fc['weights'].value    -= learning_rate * self.fc['weights'].grad
        self.fc['bias'].value       -= learning_rate * self.fc['bias'].grad
        self.conv2['weights'].value -= learning_rate * self.conv2['weights'].grad
        self.conv2['bias'].value    -= learning_rate * self.conv2['bias'].grad
        self.conv1['weights'].value -= learning_rate * self.conv1['weights'].grad
        self.conv1['bias'].value    -= learning_rate * self.conv1['bias'].grad

# 加载MNIST数据集
train_x, train_y, test_x, test_y = load_mnist()

# # 裁剪数据集
# train_x, train_y = train_x[:6000], train_y[:6000]
# test_x, test_y = test_x[:1000], test_y[:1000]

train_n = len(train_x)
test_n = len(test_x)

train_x = np.reshape(train_x, (train_n, 28, 28)) / 255.0
test_x = np.reshape(test_x, (test_n, 28, 28)) / 255.0
train_y = train_y.astype(float)
test_y = test_y.astype(float)

# 初始化模型
cnn = SimpleCNN()

total_train = len(train_x)
# 超参数设置
learning_rate = 0.01
num_epochs = 1
batch_size = 32
num_batchs = total_train // batch_size

loss_array = []
acc_array = []
# 训练模型
for epoch in range(num_epochs):
    for i in range(0, total_train, batch_size):
        # 从训练数据中获取当前批次的数据和标签
        batch_x = train_x[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]

        # 前向传播
        output = cnn.forward(batch_x)

        # 反向传播
        loss = cnn.backward(batch_y)

        # 更新权重
        cnn.update(learning_rate)

        accuracy = np.mean(np.argmax(output, axis=1) == batch_y)
        
        loss_array.append(loss)
        acc_array.append(accuracy)
        print(f'\r Epoch {epoch + 1}/{num_epochs}, Batch {i // batch_size + 1} / {num_batchs}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%', end='')

# 绘制损失和准确率曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_array, label='Loss')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc_array, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 测试模型
predictions = [np.argmax(cnn.forward(x), axis=1) for x in test_x]
accuracy = np.mean(np.array(predictions) == test_y)
print(f'Test accuracy: {accuracy * 100}%')
