# 数据加载器创建模块
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import EmbeddingDataset
import numpy as np

def create_dataloaders(data_path, config, is_memory_mapped=False, transform=None):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_path (str): 数据路径
        config (dict): 配置字典
        is_memory_mapped (bool): 是否使用内存映射
        transform (callable, optional): 数据转换函数
    
    Returns:
        tuple:
            - DataLoader: 训练数据加载器
            - DataLoader: 验证数据加载器
            - DataLoader: 测试数据加载器
    """
    # 加载配置
    batch_size = config['data']['batch_size']
    shuffle = config['data']['shuffle']
    num_workers = config['data']['num_workers']
    embedding_dim = config['data']['embedding_dim']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    
    # 验证分割比例
    if not np.isclose(train_split + val_split + test_split, 1.0):
        print(f"警告: 分割比例之和不为1.0，已重新归一化。")
        total = train_split + val_split + test_split
        train_split /= total
        val_split /= total
        test_split /= total
    
    # 创建数据集
    dataset = EmbeddingDataset(
        data_path=data_path,
        embedding_dim=embedding_dim,
        is_memory_mapped=is_memory_mapped,
        transform=transform
    )
    
    # 计算分割大小
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # 确保测试集不为空
    if test_size <= 0:
        test_size = 1
        val_size = max(1, val_size - 1)
        train_size = dataset_size - val_size - test_size
    
    print(f"数据集大小: {dataset_size}")
    print(f"训练集大小: {train_size} ({train_split*100:.1f}%)")
    print(f"验证集大小: {val_size} ({val_split*100:.1f}%)")
    print(f"测试集大小: {test_size} ({test_split*100:.1f}%)")
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # 如果使用GPU，加快数据传输
        drop_last=False   # 不丢弃最后一个不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_large_scale_dataloaders(data_paths, config, transform=None):
    """
    创建大规模数据的加载器，支持从多个文件或目录加载数据
    
    Args:
        data_paths (list): 数据路径列表
        config (dict): 配置字典
        transform (callable, optional): 数据转换函数
    
    Returns:
        tuple:
            - DataLoader: 训练数据加载器
            - DataLoader: 验证数据加载器
            - DataLoader: 测试数据加载器
    """
    # 对于大规模数据，我们假设每个路径已经是预分割的数据集
    # 这里提供一个简化的实现，实际应用中可能需要更复杂的处理
    
    if len(data_paths) == 3:
        # 如果提供了三个路径，分别作为训练、验证和测试集
        train_path, val_path, test_path = data_paths
        
        # 创建各个数据集
        train_dataset = EmbeddingDataset(
            data_path=train_path,
            embedding_dim=config['data']['embedding_dim'],
            is_memory_mapped=True,  # 大规模数据使用内存映射
            transform=transform
        )
        
        val_dataset = EmbeddingDataset(
            data_path=val_path,
            embedding_dim=config['data']['embedding_dim'],
            is_memory_mapped=True,
            transform=transform
        )
        
        test_dataset = EmbeddingDataset(
            data_path=test_path,
            embedding_dim=config['data']['embedding_dim'],
            is_memory_mapped=True,
            transform=transform
        )
    else:
        # 否则，使用第一个路径并进行分割
        print("警告: 未提供三个数据路径，使用第一个路径并进行随机分割")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_paths[0],
            config=config,
            is_memory_mapped=True,
            transform=transform
        )
        return train_loader, val_loader, test_loader
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def get_data_stats(dataset):
    """
    计算数据集的统计信息
    
    Args:
        dataset (EmbeddingDataset): 数据集
    
    Returns:
        dict: 统计信息字典
    """
    # 采样一些数据来计算统计信息
    sample_size = min(10000, len(dataset))  # 最多采样10000个样本
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    samples = []
    for idx in indices:
        samples.append(dataset[idx].numpy())
    
    samples = np.array(samples)
    
    stats = {
        'mean': np.mean(samples, axis=0),
        'std': np.std(samples, axis=0),
        'min': np.min(samples, axis=0),
        'max': np.max(samples, axis=0),
        'sample_size': sample_size
    }
    
    return stats