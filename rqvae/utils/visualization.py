# 可视化工具模块
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history (dict): 包含训练历史的字典，必须包含 'train_loss' 和 'val_loss' 键
        save_path (str, optional): 保存图像的路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练与验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 如果有其他指标，也绘制出来
    if 'train_reconstruction_loss' in history and 'val_reconstruction_loss' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_reconstruction_loss'], label='训练重建损失')
        plt.plot(history['val_reconstruction_loss'], label='验证重建损失')
        plt.title('重建损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_embeddings(embeddings, labels=None, method='tsne', n_samples=1000, save_path=None):
    """
    可视化嵌入向量
    
    Args:
        embeddings (np.ndarray or torch.Tensor): 嵌入向量，形状为 [n, dim]
        labels (np.ndarray or torch.Tensor, optional): 标签，用于着色
        method (str): 降维方法，支持 'tsne' 或 'pca'
        n_samples (int): 采样数量，用于加速可视化
        save_path (str, optional): 保存图像的路径
    """
    # 转换为numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # 采样以加速可视化
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_embeddings = embeddings[indices]
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            sampled_labels = labels[indices]
        else:
            sampled_labels = None
    else:
        sampled_embeddings = embeddings
        if labels is not None and isinstance(labels, torch.Tensor):
            sampled_labels = labels.detach().cpu().numpy()
        else:
            sampled_labels = labels
    
    # 降维
    if method == 'tsne':
        print(f"使用 t-SNE 进行降维，样本数: {len(sampled_embeddings)}")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(sampled_embeddings)
    elif method == 'pca':
        print(f"使用 PCA 进行降维，样本数: {len(sampled_embeddings)}")
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(sampled_embeddings)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    if sampled_labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=sampled_labels, 
                             cmap='viridis', alpha=0.7, s=10)
        plt.colorbar(scatter, label='标签')
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=10, color='blue')
    
    plt.title(f'{method.upper()} 嵌入向量可视化')
    plt.xlabel('维度 1')
    plt.ylabel('维度 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_embeddings(original, reconstructed, n_samples=100, save_path=None):
    """
    对比原始嵌入和重建嵌入
    
    Args:
        original (np.ndarray or torch.Tensor): 原始嵌入
        reconstructed (np.ndarray or torch.Tensor): 重建嵌入
        n_samples (int): 采样数量
        save_path (str, optional): 保存图像的路径
    """
    # 转换为numpy数组
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # 采样
    if len(original) > n_samples:
        indices = np.random.choice(len(original), n_samples, replace=False)
        original_sampled = original[indices]
        reconstructed_sampled = reconstructed[indices]
    else:
        original_sampled = original
        reconstructed_sampled = reconstructed
    
    # 计算余弦相似度
    original_normalized = original_sampled / np.linalg.norm(original_sampled, axis=1, keepdims=True)
    reconstructed_normalized = reconstructed_sampled / np.linalg.norm(reconstructed_sampled, axis=1, keepdims=True)
    cosine_similarities = np.sum(original_normalized * reconstructed_normalized, axis=1)
    
    # 计算MSE
    mse_values = np.mean((original_sampled - reconstructed_sampled) ** 2, axis=1)
    
    # 绘制直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(cosine_similarities, bins=50, alpha=0.7, color='blue')
    axes[0].axvline(np.mean(cosine_similarities), color='red', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(cosine_similarities):.4f}')
    axes[0].set_title('原始嵌入与重建嵌入的余弦相似度分布')
    axes[0].set_xlabel('余弦相似度')
    axes[0].set_ylabel('频次')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mse_values, bins=50, alpha=0.7, color='green')
    axes[1].axvline(np.mean(mse_values), color='red', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(mse_values):.4f}')
    axes[1].set_title('原始嵌入与重建嵌入的MSE分布')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('频次')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_codebook_usage(codebook_indices, num_codebooks, save_path=None):
    """
    可视化码本使用情况
    
    Args:
        codebook_indices (np.ndarray or torch.Tensor): 码本索引，形状为 [n, num_codebooks]
        num_codebooks (int): 码本数量
        save_path (str, optional): 保存图像的路径
    """
    # 转换为numpy数组
    if isinstance(codebook_indices, torch.Tensor):
        codebook_indices = codebook_indices.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 4 * num_codebooks))
    
    for i in range(num_codebooks):
        plt.subplot(num_codebooks, 1, i + 1)
        unique, counts = np.unique(codebook_indices[:, i], return_counts=True)
        plt.bar(unique, counts, alpha=0.7)
        plt.title(f'码本 {i+1} 的使用频率')
        plt.xlabel('码本索引')
        plt.ylabel('使用次数')
        plt.grid(True, alpha=0.3)
        
        # 计算并显示熵
        total = len(codebook_indices)
        probabilities = counts / total
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        plt.text(0.95, 0.95, f'熵: {entropy:.4f}', transform=plt.gca().transAxes, 
                 ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reconstruction_examples(original, reconstructed, n_examples=5, feature_indices=None, save_path=None):
    """
    绘制重建示例
    
    Args:
        original (np.ndarray or torch.Tensor): 原始嵌入
        reconstructed (np.ndarray or torch.Tensor): 重建嵌入
        n_examples (int): 示例数量
        feature_indices (list, optional): 要可视化的特征索引
        save_path (str, optional): 保存图像的路径
    """
    # 转换为numpy数组
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # 选择特征索引
    if feature_indices is None:
        # 如果没有指定，选择前10个特征
        feature_indices = list(range(min(10, original.shape[1])))
    
    # 选择示例
    if len(original) > n_examples:
        indices = np.random.choice(len(original), n_examples, replace=False)
    else:
        indices = list(range(len(original)))
    
    # 绘制每个示例
    plt.figure(figsize=(12, 4 * n_examples))
    
    for i, idx in enumerate(indices):
        plt.subplot(n_examples, 1, i + 1)
        
        # 绘制特征
        plt.plot(feature_indices, original[idx, feature_indices], 'b-', label='原始')
        plt.plot(feature_indices, reconstructed[idx, feature_indices], 'r--', label='重建')
        
        # 计算并显示余弦相似度
        orig_norm = original[idx] / np.linalg.norm(original[idx])
        recon_norm = reconstructed[idx] / np.linalg.norm(reconstructed[idx])
        cos_sim = np.dot(orig_norm, recon_norm)
        
        plt.title(f'示例 {i+1} (余弦相似度: {cos_sim:.4f})')
        plt.xlabel('特征索引')
        plt.ylabel('特征值')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_quantization_error(error_history, save_path=None):
    """
    绘制量化误差历史
    
    Args:
        error_history (dict): 包含量化误差历史的字典
        save_path (str, optional): 保存图像的路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制总量化误差
    if 'total_quantization_error' in error_history:
        plt.plot(error_history['total_quantization_error'], label='总量化误差')
    
    # 绘制每个码本的量化误差
    for key in error_history:
        if key.startswith('quantization_error_'):
            codebook_idx = key.split('_')[-1]
            plt.plot(error_history[key], label=f'码本 {codebook_idx} 量化误差')
    
    plt.title('量化误差历史')
    plt.xlabel('轮次')
    plt.ylabel('误差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_confusion_matrix(actual, predicted, num_classes, normalize=True, save_path=None):
    """
    创建混淆矩阵
    
    Args:
        actual (np.ndarray or torch.Tensor): 实际类别
        predicted (np.ndarray or torch.Tensor): 预测类别
        num_classes (int): 类别数量
        normalize (bool): 是否归一化
        save_path (str, optional): 保存图像的路径
    """
    # 转换为numpy数组
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    
    # 创建混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual, predicted, labels=range(num_classes))
    
    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()