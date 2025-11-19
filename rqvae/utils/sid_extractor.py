# SID (Semantic Identifier) 提取工具模块
import torch
import numpy as np
import os
from typing import List, Tuple, Optional, Union

from rqvae.models import RQVAE
from rqvae.utils.config import load_config
from rqvae.utils.logger import setup_logger

def load_rqvae_model(model_path: str, config_path: Optional[str] = None) -> RQVAE:
    """
    加载训练好的RQVAE模型
    
    Args:
        model_path (str): 模型权重文件路径
        config_path (str, optional): 配置文件路径，如果为None则从模型文件中推断
    
    Returns:
        RQVAE: 加载好的RQVAE模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    if config_path is not None:
        config = load_config(config_path)
    else:
        # 尝试从模型路径所在目录查找config.yaml
        model_dir = os.path.dirname(model_path)
        if model_dir == '':
            model_dir = '.'
        
        # 尝试多个可能的配置文件路径
        possible_config_paths = [
            os.path.join(model_dir, 'config.yaml'),
            os.path.join(os.path.dirname(model_dir), 'config.yaml'),
            os.path.join(model_dir, '../config.yaml')
        ]
        
        config_path = None
        for path in possible_config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"无法找到配置文件，请提供config_path参数")
        
        config = load_config(config_path)
    
    # 确保配置结构符合RQVAE.load期望的格式
    # RQVAE.load期望的格式: config['model']['encoder'], config['model']['decoder'], config['model']['quantizer']
    # 如果配置结构不是嵌套的，需要将其转换为嵌套格式
    if 'model' in config:
        model_config = config['model']
        # 确保encoder, decoder和quantizer子配置存在
        if 'encoder' not in model_config:
            # 创建encoder子配置
            model_config['encoder'] = {
                'hidden_dims': model_config.get('encoder_hidden_dims', [512, 256, 128]),
                'activation': model_config.get('activation', 'relu'),
                'dropout': model_config.get('dropout', 0.1)
            }
        
        if 'decoder' not in model_config:
            # 创建decoder子配置
            model_config['decoder'] = {
                'hidden_dims': model_config.get('decoder_hidden_dims', [128, 256, 512])
            }
        
        if 'quantizer' not in model_config:
            # 创建quantizer子配置
            model_config['quantizer'] = {
                'latent_dim': model_config.get('latent_dim', 32),
                'num_codebooks': model_config.get('num_codebooks', 8),
                'codebook_size': model_config.get('codebook_size', 256),
                'commitment_cost': model_config.get('commitment_cost', 0.25),
                'decay': model_config.get('decay', 0.99)
            }
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 创建模型实例 - 由于RQVAE.load类方法需要直接的模型权重文件而非完整检查点
    # 我们直接创建模型实例并手动加载权重
    try:
        # 获取模型配置
        model_config = config['model']
        
        # 创建模型实例
        model = RQVAE(
            input_dim=model_config.get('input_dim', 512),
            encoder_hidden_dims=model_config['encoder'].get('hidden_dims', [512, 256, 128]),
            decoder_hidden_dims=model_config['decoder'].get('hidden_dims', [128, 256, 512]),
            latent_dim=model_config['quantizer'].get('latent_dim', 32),
            num_codebooks=model_config['quantizer'].get('num_codebooks', 8),
            codebook_size=model_config['quantizer'].get('codebook_size', 256),
            commitment_cost=model_config['quantizer'].get('commitment_cost', 0.25),
            decay=model_config['quantizer'].get('decay', 0.99),
            activation=model_config['encoder'].get('activation', 'relu'),
            dropout=model_config['encoder'].get('dropout', 0.1)
        ).to(device)
        
        # 处理完整检查点格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 尝试直接加载（如果是只包含模型权重的文件）
            model.load_state_dict(checkpoint)
    except Exception as e:
        # 如果出错，尝试另一种方式创建模型
        # 直接使用配置中的参数（更宽松的错误处理）
        model_config = config['model']
        
        # 提取必要参数，使用默认值避免类型错误
        input_dim = int(model_config.get('input_dim', 512))
        
        # 处理encoder_hidden_dims，确保是列表
        encoder_hidden_dims = model_config.get('encoder_hidden_dims', [512, 256, 128])
        if isinstance(encoder_hidden_dims, dict):
            encoder_hidden_dims = [int(encoder_hidden_dims.get(k, 512)) for k in sorted(encoder_hidden_dims.keys())]
        else:
            encoder_hidden_dims = [int(dim) for dim in encoder_hidden_dims]
        
        # 处理decoder_hidden_dims，确保是列表
        decoder_hidden_dims = model_config.get('decoder_hidden_dims', [128, 256, 512])
        if isinstance(decoder_hidden_dims, dict):
            decoder_hidden_dims = [int(decoder_hidden_dims.get(k, 512)) for k in sorted(decoder_hidden_dims.keys())]
        else:
            decoder_hidden_dims = [int(dim) for dim in decoder_hidden_dims]
        
        # 提取其他参数，确保类型正确
        latent_dim = int(model_config.get('latent_dim', 32))
        num_codebooks = int(model_config.get('num_codebooks', 8))
        codebook_size = int(model_config.get('codebook_size', 256))
        commitment_cost = float(model_config.get('commitment_cost', 0.25))
        decay = float(model_config.get('decay', 0.99))
        activation = str(model_config.get('activation', 'relu'))
        dropout = float(model_config.get('dropout', 0.1))
        
        # 创建模型实例
        model = RQVAE(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
            decay=decay,
            activation=activation,
            dropout=dropout
        ).to(device)
        
        # 处理完整检查点格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 尝试直接加载（如果是只包含模型权重的文件）
            model.load_state_dict(checkpoint)
    
    # 确保模型处于评估模式
    model.eval()
    
    return model

class SIDExtractor:
    """
    SID提取器类，用于从embedding提取语义标识符和从语义标识符重建embedding
    """
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        初始化SID提取器
        
        Args:
            model_path (str): 模型权重文件路径
            config_path (str, optional): 配置文件路径
        """
        # 设置日志
        self.logger = setup_logger('SID-Extractor')
        
        # 加载模型
        self.logger.info(f"加载RQVAE模型: {model_path}")
        self.model = load_rqvae_model(model_path, config_path)
        
        # 获取设备
        self.device = next(self.model.parameters()).device
        
        # 获取模型配置
        # 注意：RQVAE模型使用input_dim表示嵌入维度
        self.embedding_dim = getattr(self.model, 'input_dim', 512)  # 默认512
        self.latent_dim = getattr(self.model, 'latent_dim', 32)  # 默认32
        self.num_codebooks = self.model.quantizer.num_codebooks
        self.codebook_size = self.model.quantizer.codebook_size
        
        self.logger.info(f"模型加载完成，设备: {self.device}")
        self.logger.info(f"模型配置: 嵌入维度={self.embedding_dim}, "
                       f"潜在维度={self.latent_dim}, "
                       f"码本数量={self.num_codebooks}, "
                       f"码本大小={self.codebook_size}")
    
    def extract_sid(self, embeddings: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从embedding提取SID（语义标识符）
        
        Args:
            embeddings (np.ndarray or torch.Tensor): 输入的embedding向量，形状为 [batch_size, embedding_dim]
        
        Returns:
            tuple:
                - np.ndarray: SID码本索引，形状为 [batch_size, num_codebooks]
                - np.ndarray: 量化后的潜在表示，形状为 [batch_size, latent_dim]
        """
        # 转换为torch.Tensor
        if isinstance(embeddings, np.ndarray):
            embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        else:
            embeddings_tensor = embeddings.float().to(self.device)
        
        # 检查输入维度
        if embeddings_tensor.shape[1] != self.embedding_dim:
            raise ValueError(f"输入embedding维度不匹配，期望: {self.embedding_dim}, 实际: {embeddings_tensor.shape[1]}")
        
        with torch.no_grad():
            # 提取SID
            codebook_indices, quantized = self.model.encode(embeddings_tensor)
        
        # 转换为numpy数组
        codebook_indices_np = codebook_indices.cpu().numpy()
        quantized_np = quantized.cpu().numpy()
        
        return codebook_indices_np, quantized_np
    
    def extract_sid_batch(self, embeddings: Union[np.ndarray, torch.Tensor], 
                         batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量提取SID，适用于大规模数据
        
        Args:
            embeddings (np.ndarray or torch.Tensor): 输入的embedding向量
            batch_size (int): 批次大小
        
        Returns:
            tuple:
                - np.ndarray: SID码本索引
                - np.ndarray: 量化后的潜在表示
        """
        # 转换为torch.Tensor
        if isinstance(embeddings, np.ndarray):
            embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        else:
            embeddings_tensor = embeddings.float().to(self.device)
        
        # 获取总样本数
        total_samples = embeddings_tensor.shape[0]
        
        # 初始化结果数组
        all_codebook_indices = np.zeros((total_samples, self.num_codebooks), dtype=np.int32)
        all_quantized = np.zeros((total_samples, self.latent_dim), dtype=np.float32)
        
        # 批量处理
        self.logger.info(f"开始批量提取SID，总样本数: {total_samples}, 批次大小: {batch_size}")
        
        for i in range(0, total_samples, batch_size):
            # 获取当前批次
            end_idx = min(i + batch_size, total_samples)
            batch = embeddings_tensor[i:end_idx]
            
            # 提取SID
            with torch.no_grad():
                codebook_indices, quantized = self.model.encode(batch)
            
            # 保存结果
            all_codebook_indices[i:end_idx] = codebook_indices.cpu().numpy()
            all_quantized[i:end_idx] = quantized.cpu().numpy()
            
            # 记录进度
            if (i + batch_size) % (batch_size * 10) == 0 or end_idx == total_samples:
                self.logger.info(f"已处理: {end_idx}/{total_samples} ({end_idx/total_samples*100:.1f}%)")
        
        return all_codebook_indices, all_quantized
    
    def reconstruct_from_sid(self, codebook_indices: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        从SID重建embedding
        
        Args:
            codebook_indices (np.ndarray or torch.Tensor): SID码本索引，形状为 [batch_size, num_codebooks]
        
        Returns:
            np.ndarray: 重建的embedding向量，形状为 [batch_size, embedding_dim]
        """
        # 转换为torch.Tensor
        if isinstance(codebook_indices, np.ndarray):
            indices_tensor = torch.from_numpy(codebook_indices).long().to(self.device)
        else:
            indices_tensor = codebook_indices.long().to(self.device)
        
        # 检查输入维度
        if indices_tensor.shape[1] != self.num_codebooks:
            raise ValueError(f"输入SID维度不匹配，期望: {self.num_codebooks}, 实际: {indices_tensor.shape[1]}")
        
        with torch.no_grad():
            # 从SID重建
            reconstructed = self.model.decode_from_indices(indices_tensor)
        
        # 转换为numpy数组
        reconstructed_np = reconstructed.cpu().numpy()
        
        return reconstructed_np
    
    def compute_sid_similarity(self, sid1: np.ndarray, sid2: np.ndarray) -> np.ndarray:
        """
        计算两个SID之间的相似度
        相似度定义为相同码本索引的比例
        
        Args:
            sid1 (np.ndarray): 第一个SID，形状为 [num_codebooks] 或 [batch_size, num_codebooks]
            sid2 (np.ndarray): 第二个SID，形状为 [num_codebooks] 或 [batch_size, num_codebooks]
        
        Returns:
            np.ndarray: 相似度得分，范围为 [0, 1]
        """
        # 确保维度一致
        if sid1.ndim == 1 and sid2.ndim == 1:
            # 单个SID的情况
            return np.mean(sid1 == sid2)
        elif sid1.ndim == 2 and sid2.ndim == 2:
            # 批量SID的情况
            return np.mean(sid1 == sid2, axis=1)
        else:
            raise ValueError("SID维度不匹配，请确保两者都是一维或二维数组")
    
    def search_similar_sids(self, query_sid: np.ndarray, database_sids: np.ndarray, 
                          top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        在数据库中搜索与查询SID最相似的SID
        
        Args:
            query_sid (np.ndarray): 查询SID，形状为 [num_codebooks]
            database_sids (np.ndarray): 数据库SID，形状为 [n, num_codebooks]
            top_k (int): 返回前k个最相似的结果
        
        Returns:
            tuple:
                - np.ndarray: 相似度得分，形状为 [top_k]
                - np.ndarray: 索引，形状为 [top_k]
        """
        # 计算与每个数据库SID的相似度
        similarities = self.compute_sid_similarity(
            np.tile(query_sid, (len(database_sids), 1)), 
            database_sids
        )
        
        # 获取前k个最相似的
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_similarities, top_indices
    
    def sid_to_string(self, sid: np.ndarray) -> str:
        """
        将SID转换为字符串表示
        
        Args:
            sid (np.ndarray): SID，形状为 [num_codebooks]
        
        Returns:
            str: SID的字符串表示，格式如 "123-456-789"
        """
        return "-".join(map(str, sid))
    
    def string_to_sid(self, sid_str: str) -> np.ndarray:
        """
        将字符串表示的SID转换为数组
        
        Args:
            sid_str (str): SID的字符串表示，格式如 "123-456-789"
        
        Returns:
            np.ndarray: SID数组，形状为 [num_codebooks]
        """
        sid = np.array(list(map(int, sid_str.split("-"))), dtype=np.int32)
        
        if len(sid) != self.num_codebooks:
            raise ValueError(f"SID字符串长度不匹配，期望: {self.num_codebooks}, 实际: {len(sid)}")
        
        return sid
    
    def save_sids(self, sids: np.ndarray, file_path: str):
        """
        保存SID到文件
        
        Args:
            sids (np.ndarray): SID数组，形状为 [n, num_codebooks]
            file_path (str): 保存路径
        """
        np.save(file_path, sids)
        self.logger.info(f"SID已保存到: {file_path}")
    
    def load_sids(self, file_path: str) -> np.ndarray:
        """
        从文件加载SID
        
        Args:
            file_path (str): 文件路径
        
        Returns:
            np.ndarray: SID数组
        """
        sids = np.load(file_path)
        self.logger.info(f"从 {file_path} 加载了 {len(sids)} 个SID")
        return sids
    
    def export_codebooks(self, output_dir: str):
        """
        导出码本到文件
        
        Args:
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理ParameterList类型的码本
        codebook_params = self.model.quantizer.codebooks
        
        # 保存每个码本
        for i in range(len(codebook_params)):
            # 单独转换每个参数为numpy数组，使用detach()分离梯度
            codebook = codebook_params[i].detach().cpu().numpy()
            codebook_path = os.path.join(output_dir, f'codebook_{i}.npy')
            np.save(codebook_path, codebook)
            self.logger.info(f"码本 {i} 已保存到: {codebook_path}")
        
        # 保存所有码本为列表，使用detach()分离梯度
        all_codebooks = [codebook_params[i].detach().cpu().numpy() for i in range(len(codebook_params))]
        all_codebooks_path = os.path.join(output_dir, 'all_codebooks.npy')
        np.save(all_codebooks_path, all_codebooks)
        self.logger.info(f"所有码本已保存到: {all_codebooks_path}")
        self.logger.info(f"共导出 {len(codebook_params)} 个码本")

# 示例使用函数
def extract_sids_from_embeddings_file(model_path: str, embeddings_file: str, 
                                     output_file: str, batch_size: int = 1024,
                                     config_path: Optional[str] = None):
    """
    从embeddings文件提取SID并保存
    
    Args:
        model_path (str): 模型路径
        embeddings_file (str): embeddings文件路径 (.npy 或 .npz)
        output_file (str): SID输出文件路径
        batch_size (int): 批次大小
        config_path (str, optional): 配置文件路径
    """
    # 设置日志
    logger = setup_logger('Extract-SIDs')
    
    # 加载embeddings
    logger.info(f"加载embeddings文件: {embeddings_file}")
    if embeddings_file.endswith('.npz'):
        # 对于.npz文件，尝试找到名为'embeddings'、'features'或第一个键
        data = np.load(embeddings_file)
        keys = list(data.keys())
        if 'embeddings' in keys:
            embeddings = data['embeddings']
        elif 'features' in keys:
            embeddings = data['features']
        else:
            embeddings = data[keys[0]]
    else:
        # 对于.npy文件，直接加载
        embeddings = np.load(embeddings_file)
    
    logger.info(f"加载了 {len(embeddings)} 个embeddings，维度: {embeddings.shape[1]}")
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path, config_path)
    
    # 提取SID
    sids, _ = extractor.extract_sid_batch(embeddings, batch_size)
    
    # 保存SID
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    extractor.save_sids(sids, output_file)
    
    logger.info(f"SID提取完成，共提取 {len(sids)} 个SID")
    return sids

def reconstruct_embeddings_from_sids(model_path: str, sids_file: str, 
                                     output_file: str, batch_size: int = 1024,
                                     config_path: Optional[str] = None):
    """
    从SID文件重建embeddings并保存
    
    Args:
        model_path (str): 模型路径
        sids_file (str): SID文件路径 (.npy)
        output_file (str): 重建embeddings输出文件路径
        batch_size (int): 批次大小
        config_path (str, optional): 配置文件路径
    """
    # 设置日志
    logger = setup_logger('Reconstruct-Embeddings')
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path, config_path)
    
    # 加载SID
    sids = extractor.load_sids(sids_file)
    
    # 批量重建
    total_samples = len(sids)
    reconstructed_embeddings = np.zeros((total_samples, extractor.embedding_dim), dtype=np.float32)
    
    logger.info(f"开始从SID重建embeddings，总样本数: {total_samples}")
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_sids = sids[i:end_idx]
        
        # 重建
        batch_reconstructed = extractor.reconstruct_from_sid(batch_sids)
        
        # 保存结果
        reconstructed_embeddings[i:end_idx] = batch_reconstructed
        
        # 记录进度
        if (i + batch_size) % (batch_size * 10) == 0 or end_idx == total_samples:
            logger.info(f"已重建: {end_idx}/{total_samples} ({end_idx/total_samples*100:.1f}%)")
    
    # 保存重建的embeddings
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, reconstructed_embeddings)
    
    logger.info(f"embeddings重建完成，已保存到: {output_file}")
    return reconstructed_embeddings