# RQ-VAE残差向量量化器模块
import torch
import torch.nn as nn

class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器，将连续的潜在表示转换为离散的码本索引（SID）
    
    Args:
        latent_dim (int): 潜在空间维度
        num_codebooks (int): 码本数量
        codebook_size (int): 每个码本的大小
        commitment_cost (float): 承诺损失权重
        decay (float): EMA更新衰减率
        epsilon (float): 小的常数，避免除零错误
    """
    def __init__(self, latent_dim, num_codebooks, codebook_size, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(ResidualVectorQuantizer, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 为每个码本创建嵌入表
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim))
            for _ in range(num_codebooks)
        ])
        
        # 初始化EMA参数（用于非训练模式下的码本更新）
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', torch.zeros_like(torch.stack(self.codebooks, dim=0)))
    
    def forward(self, z):
        """
        前向传播，执行量化操作
        
        Args:
            z (torch.Tensor): 输入潜在表示，形状为 [batch_size, latent_dim]
        
        Returns:
            tuple:
                - torch.Tensor: 量化后的潜在表示，形状为 [batch_size, latent_dim]
                - torch.Tensor: 量化损失
                - torch.Tensor: 每个样本在每个码本中的索引，形状为 [batch_size, num_codebooks]
        """
        batch_size = z.shape[0]
        
        # 初始化量化输出和残差
        quantized = torch.zeros_like(z)
        residual = z.clone()
        
        # 初始化量化损失
        loss = 0.0
        
        # 初始化码本索引
        codebook_indices = torch.zeros(batch_size, self.num_codebooks, dtype=torch.long, device=z.device)
        
        # 对每个码本进行量化
        for i in range(self.num_codebooks):
            # 计算残差与码本向量的距离
            codebook = self.codebooks[i]
            # 展开residual和codebook以计算距离
            residual_flat = residual.view(-1, self.latent_dim)
            codebook_flat = codebook.view(-1, self.latent_dim)
            
            # 计算L2距离: ||r - c||^2 = ||r||^2 + ||c||^2 - 2r·c
            distances = torch.sum(residual_flat ** 2, dim=1, keepdim=True) \
                      + torch.sum(codebook_flat ** 2, dim=1) \
                      - 2 * torch.matmul(residual_flat, codebook_flat.t())
            
            # 找到最近的码本向量索引
            encoding_indices = torch.argmin(distances, dim=1)
            codebook_indices[:, i] = encoding_indices
            
            # 计算量化向量
            quantized_flat = torch.index_select(codebook_flat, 0, encoding_indices)
            quantized_i = quantized_flat.view(batch_size, self.latent_dim)
            
            # 更新残差
            residual = residual - quantized_i
            
            # 更新量化输出
            quantized = quantized + quantized_i
            
            # 计算承诺损失
            if self.training:
                # 推动编码器输出接近量化向量
                commitment_loss = self.commitment_cost * torch.mean((quantized_i.detach() - z) ** 2)
                # 推动量化向量接近编码器输出
                codebook_loss = torch.mean((quantized_i - z.detach()) ** 2)
                
                loss = loss + commitment_loss + codebook_loss
                
                # EMA更新码本（非训练时使用）
                with torch.no_grad():
                    # 更新聚类大小
                    self.cluster_size[i].data.mul_(self.decay).add_(
                        (1 - self.decay) * torch.bincount(encoding_indices, minlength=self.codebook_size)
                    )
                    
                    # 更新嵌入平均值
                    embed_sum = torch.zeros_like(codebook_flat)
                    embed_sum.scatter_add_(0, encoding_indices.unsqueeze(1).repeat(1, self.latent_dim), residual_flat)
                    self.embed_avg[i].data.mul_(self.decay).add_((1 - self.decay) * embed_sum)
                    
                    # 重新归一化码本
                    n = self.cluster_size[i].sum()
                    cluster_size = ((self.cluster_size[i] + self.epsilon) / (n + self.codebook_size * self.epsilon) * n)
                    embed_normalized = self.embed_avg[i] / cluster_size.unsqueeze(1)
                    self.codebooks[i].data.copy_(embed_normalized)
        
        return quantized, loss, codebook_indices
    
    def get_codebook(self, codebook_idx):
        """
        获取特定的码本
        
        Args:
            codebook_idx (int): 码本索引
        
        Returns:
            torch.Tensor: 码本张量
        """
        if codebook_idx < 0 or codebook_idx >= self.num_codebooks:
            raise ValueError(f"码本索引 {codebook_idx} 超出范围 [0, {self.num_codebooks-1}]")
        return self.codebooks[codebook_idx]
    
    def decode(self, codebook_indices):
        """
        从码本索引解码得到量化向量
        
        Args:
            codebook_indices (torch.Tensor): 码本索引，形状为 [batch_size, num_codebooks]
        
        Returns:
            torch.Tensor: 解码后的向量，形状为 [batch_size, latent_dim]
        """
        batch_size = codebook_indices.shape[0]
        quantized = torch.zeros(batch_size, self.latent_dim, device=codebook_indices.device)
        
        for i in range(self.num_codebooks):
            indices = codebook_indices[:, i]
            quantized_i = torch.index_select(self.codebooks[i], 0, indices)
            quantized = quantized + quantized_i
        
        return quantized