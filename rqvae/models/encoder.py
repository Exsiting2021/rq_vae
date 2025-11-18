# RQ-VAE编码器模块
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    编码器模块，将输入的高维embedding压缩到低维潜在空间
    
    Args:
        input_dim (int): 输入embedding维度
        hidden_dims (list): 隐藏层维度列表
        latent_dim (int): 潜在空间维度
        activation (str): 激活函数类型，默认为'relu'
        dropout (float): dropout概率，默认为0.0
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu', dropout=0.0):
        super(Encoder, self).__init__()
        
        # 根据激活函数名称选择激活函数
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"不支持的激活函数类型: {activation}")
        
        # 构建编码器网络
        layers = []
        current_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # 添加输出层，映射到潜在空间
        layers.append(nn.Linear(current_dim, latent_dim))
        
        # 将所有层组合成序列
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]
        
        Returns:
            torch.Tensor: 编码后的潜在表示，形状为 [batch_size, latent_dim]
        """
        return self.encoder(x)