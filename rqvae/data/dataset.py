# 大规模Embedding数据集处理
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm

class EmbeddingDataset(Dataset):
    """
    处理大规模embedding数据的数据集类
    
    Args:
        data_path (str): 数据文件路径，可以是.npy、.npz或文件夹路径
        embedding_dim (int): embedding维度
        is_memory_mapped (bool): 是否使用内存映射（适用于超大规模数据）
        transform (callable, optional): 数据转换函数
    """
    def __init__(self, data_path, embedding_dim=512, is_memory_mapped=False, transform=None):
        self.embedding_dim = embedding_dim
        self.transform = transform
        self.is_memory_mapped = is_memory_mapped
        
        # 加载数据
        self.data = self._load_data(data_path)
        
        # 获取数据长度
        if isinstance(self.data, np.memmap):
            self.length = len(self.data)
        elif isinstance(self.data, np.ndarray):
            self.length = len(self.data)
        elif isinstance(self.data, list):
            self.length = len(self.data)
        else:
            raise TypeError(f"不支持的数据类型: {type(self.data)}")
    
    def _load_data(self, data_path):
        """
        根据路径类型加载数据
        
        Args:
            data_path (str): 数据路径
        
        Returns:
            加载的数据对象
        """
        if os.path.isdir(data_path):
            # 从文件夹加载多个文件
            return self._load_from_directory(data_path)
        elif data_path.endswith('.npy'):
            if self.is_memory_mapped:
                # 使用内存映射加载大型.npy文件
                return np.memmap(data_path, dtype='float32', mode='r')
            else:
                return np.load(data_path)
        elif data_path.endswith('.npz'):
            # 加载.npz文件
            with np.load(data_path) as data:
                # 假设数据存储在'embeddings'键下
                if 'embeddings' in data:
                    return data['embeddings']
                else:
                    # 如果没有指定键，使用第一个数组
                    keys = list(data.keys())
                    print(f"警告: 未找到'embeddings'键，使用第一个键: {keys[0]}")
                    return data[keys[0]]
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
    
    def _load_from_directory(self, directory_path):
        """
        从目录加载多个embedding文件
        
        Args:
            directory_path (str): 目录路径
        
        Returns:
            list或np.memmap: 合并的数据
        """
        all_files = []
        # 遍历目录中的所有.npy和.npz文件
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.npy', '.npz')):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            raise FileNotFoundError(f"在目录中未找到.npy或.npz文件: {directory_path}")
        
        print(f"找到 {len(all_files)} 个embedding文件")
        
        # 对于内存映射模式，我们将返回文件列表，按需加载
        if self.is_memory_mapped:
            return all_files
        
        # 否则，合并所有数据到一个numpy数组
        all_embeddings = []
        for file_path in tqdm(all_files, desc="加载embedding文件"):
            if file_path.endswith('.npy'):
                embeddings = np.load(file_path)
            elif file_path.endswith('.npz'):
                with np.load(file_path) as data:
                    keys = list(data.keys())
                    embeddings = data[keys[0]]
            
            # 验证维度
            if len(embeddings.shape) == 1:
                # 单个embedding，扩展维度
                embeddings = embeddings.reshape(1, -1)
            
            if embeddings.shape[1] != self.embedding_dim:
                print(f"警告: 文件 {file_path} 的embedding维度不匹配 ({embeddings.shape[1]} vs {self.embedding_dim})")
            
            all_embeddings.append(embeddings)
        
        # 合并所有embedding
        return np.vstack(all_embeddings)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据
        
        Args:
            idx (int): 索引
        
        Returns:
            torch.Tensor: embedding张量
        """
        if self.is_memory_mapped and isinstance(self.data, list):
            # 内存映射模式，需要确定数据在哪个文件中
            # 这里简化处理，实际应用可能需要更复杂的索引映射
            file_idx = idx % len(self.data)
            offset = idx // len(self.data)
            
            # 加载对应文件
            file_path = self.data[file_idx]
            if file_path.endswith('.npy'):
                embeddings = np.load(file_path)
            elif file_path.endswith('.npz'):
                with np.load(file_path) as data:
                    keys = list(data.keys())
                    embeddings = data[keys[0]]
            
            # 获取对应偏移量的数据（如果存在）
            if offset < len(embeddings):
                embedding = embeddings[offset]
            else:
                # 如果偏移量超出范围，使用第一个样本（实际应用中应该更智能地处理）
                embedding = embeddings[0]
        else:
            # 直接从数组或memmap获取
            embedding = self.data[idx]
        
        # 转换为numpy数组
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # 转换为torch张量
        embedding = torch.from_numpy(embedding.astype(np.float32))
        
        # 应用转换
        if self.transform:
            embedding = self.transform(embedding)
        
        return embedding
    
    def normalize(self):
        """
        对数据进行归一化处理
        """
        if isinstance(self.data, np.ndarray) and not self.is_memory_mapped:
            # 计算均值和标准差
            mean = np.mean(self.data, axis=0)
            std = np.std(self.data, axis=0) + 1e-8  # 避免除零
            
            # 归一化
            self.data = (self.data - mean) / std
            print(f"数据已归一化，均值: {np.mean(self.data, axis=0).mean():.4f}, 标准差: {np.std(self.data, axis=0).mean():.4f}")
    
    def sample(self, n_samples):
        """
        从数据集中随机采样n_samples个样本
        
        Args:
            n_samples (int): 采样数量
        
        Returns:
            EmbeddingDataset: 采样后的数据集
        """
        indices = np.random.choice(len(self), min(n_samples, len(self)), replace=False)
        
        # 创建采样后的数据
        if isinstance(self.data, np.ndarray):
            sampled_data = self.data[indices]
        else:
            sampled_data = [self.data[i] for i in indices]
        
        # 创建新的数据集实例
        sampled_dataset = EmbeddingDataset.__new__(EmbeddingDataset)
        sampled_dataset.embedding_dim = self.embedding_dim
        sampled_dataset.transform = self.transform
        sampled_dataset.is_memory_mapped = False  # 采样后的数据较小，不需要内存映射
        sampled_dataset.data = sampled_data
        sampled_dataset.length = len(indices)
        
        return sampled_dataset
    
    def save(self, save_path):
        """
        保存数据集
        
        Args:
            save_path (str): 保存路径
        """
        if isinstance(self.data, np.ndarray):
            np.save(save_path, self.data)
            print(f"数据集已保存到: {save_path}")
        else:
            print("警告: 仅支持保存numpy数组格式的数据集")

# 数据转换类示例
class NormalizeTransform:
    """
    数据归一化转换
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        else:
            # 标准化到[-1, 1]范围
            min_val = x.min()
            max_val = x.max()
            if max_val > min_val:
                return 2 * (x - min_val) / (max_val - min_val) - 1
            return x