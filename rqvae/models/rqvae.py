# RQ-VAE完整模型
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .quantizer import ResidualVectorQuantizer

class RQVAE(nn.Module):
    """
    残差量化变分自编码器（RQ-VAE），用于将高维embedding压缩为离散的SID表示
    
    Args:
        input_dim (int): 输入embedding维度
        encoder_hidden_dims (list): 编码器隐藏层维度列表
        decoder_hidden_dims (list): 解码器隐藏层维度列表
        latent_dim (int): 潜在空间维度
        num_codebooks (int): 码本数量
        codebook_size (int): 每个码本的大小
        commitment_cost (float): 承诺损失权重
        decay (float): EMA更新衰减率
        activation (str): 激活函数类型
        dropout (float): dropout概率
    """
    def __init__(self,
                 input_dim=512,
                 encoder_hidden_dims=[512, 256, 128],
                 decoder_hidden_dims=[128, 256, 512],
                 latent_dim=32,
                 num_codebooks=8,
                 codebook_size=256,
                 commitment_cost=0.25,
                 decay=0.99,
                 activation='relu',
                 dropout=0.1):
        super(RQVAE, self).__init__()
        
        # 初始化编码器
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout
        )
        
        # 初始化残差向量量化器
        self.quantizer = ResidualVectorQuantizer(
            latent_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
            decay=decay
        )
        
        # 初始化解码器
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            activation=activation,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入embedding，形状为 [batch_size, input_dim]
        
        Returns:
            tuple:
                - torch.Tensor: 重建的embedding，形状为 [batch_size, input_dim]
                - torch.Tensor: 量化损失
                - torch.Tensor: 潜在表示，形状为 [batch_size, latent_dim]
                - torch.Tensor: 量化后的潜在表示，形状为 [batch_size, latent_dim]
                - torch.Tensor: 码本索引，形状为 [batch_size, num_codebooks]
        """
        # 编码
        z = self.encoder(x)
        
        # 量化
        z_quantized, quantization_loss, codebook_indices = self.quantizer(z)
        
        # 解码
        x_recon = self.decoder(z_quantized)
        
        return x_recon, quantization_loss, z, z_quantized, codebook_indices
    
    def encode(self, x):
        """
        仅执行编码操作
        
        Args:
            x (torch.Tensor): 输入embedding，形状为 [batch_size, input_dim]
        
        Returns:
            torch.Tensor: 潜在表示，形状为 [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def quantize(self, z):
        """
        仅执行量化操作
        
        Args:
            z (torch.Tensor): 潜在表示，形状为 [batch_size, latent_dim]
        
        Returns:
            tuple:
                - torch.Tensor: 量化后的潜在表示，形状为 [batch_size, latent_dim]
                - torch.Tensor: 量化损失
                - torch.Tensor: 码本索引，形状为 [batch_size, num_codebooks]
        """
        return self.quantizer(z)
    
    def decode(self, z_quantized):
        """
        仅执行解码操作
        
        Args:
            z_quantized (torch.Tensor): 量化后的潜在表示，形状为 [batch_size, latent_dim]
        
        Returns:
            torch.Tensor: 重建的embedding，形状为 [batch_size, input_dim]
        """
        return self.decoder(z_quantized)
    
    def extract_sid(self, x):
        """
        从输入embedding提取SID（码本索引）
        
        Args:
            x (torch.Tensor): 输入embedding，形状为 [batch_size, input_dim]
        
        Returns:
            torch.Tensor: SID表示，形状为 [batch_size, num_codebooks]
        """
        with torch.no_grad():
            z = self.encode(x)
            _, _, codebook_indices = self.quantize(z)
        return codebook_indices
    
    def reconstruct_from_sid(self, codebook_indices):
        """
        从SID重建embedding
        
        Args:
            codebook_indices (torch.Tensor): SID表示，形状为 [batch_size, num_codebooks]
        
        Returns:
            torch.Tensor: 重建的embedding，形状为 [batch_size, input_dim]
        """
        with torch.no_grad():
            z_quantized = self.quantizer.decode(codebook_indices)
            x_recon = self.decode(z_quantized)
        return x_recon
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path, config):
        """
        加载模型
        
        Args:
            path (str): 模型路径
            config (dict): 模型配置
        
        Returns:
            RQVAE: 加载的模型
        """
        model = cls(
            input_dim=config['model'].get('input_dim', 512),
            encoder_hidden_dims=config['model']['encoder']['hidden_dims'],
            decoder_hidden_dims=config['model']['decoder']['hidden_dims'],
            latent_dim=config['model']['quantizer']['latent_dim'],
            num_codebooks=config['model']['quantizer']['num_codebooks'],
            codebook_size=config['model']['quantizer']['codebook_size'],
            commitment_cost=config['model']['quantizer']['commitment_cost'],
            decay=config['model']['quantizer']['decay'],
            activation=config['model']['encoder']['activation'],
            dropout=config['model']['encoder']['dropout']
        )
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model