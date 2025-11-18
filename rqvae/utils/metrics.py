# 评估指标计算模块
import torch
import numpy as np
from scipy import spatial

def compute_reconstruction_loss(x, x_recon, loss_type='mse'):
    """
    计算重建损失
    
    Args:
        x (torch.Tensor): 原始输入
        x_recon (torch.Tensor): 重建输出
        loss_type (str): 损失类型，支持'mse', 'mae', 'cosine'
    
    Returns:
        torch.Tensor: 损失值
    """
    if loss_type == 'mse':
        return torch.mean((x - x_recon) ** 2)
    elif loss_type == 'mae':
        return torch.mean(torch.abs(x - x_recon))
    elif loss_type == 'cosine':
        # 计算余弦相似度，然后转换为损失
        x_normalized = F.normalize(x, dim=1)
        x_recon_normalized = F.normalize(x_recon, dim=1)
        cosine_similarity = torch.sum(x_normalized * x_recon_normalized, dim=1)
        return torch.mean(1 - cosine_similarity)  # 1 - 相似度作为损失
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

def compute_accuracy(codebook_indices_true, codebook_indices_pred):
    """
    计算SID预测的准确率
    
    Args:
        codebook_indices_true (torch.Tensor): 真实的码本索引
        codebook_indices_pred (torch.Tensor): 预测的码本索引
    
    Returns:
        dict: 包含各种准确率指标的字典
    """
    # 确保输入是相同形状
    assert codebook_indices_true.shape == codebook_indices_pred.shape
    
    batch_size, num_codebooks = codebook_indices_true.shape
    
    # 计算每个码本的准确率
    codebook_accuracies = []
    for i in range(num_codebooks):
        correct = torch.sum(codebook_indices_true[:, i] == codebook_indices_pred[:, i]).item()
        accuracy = correct / batch_size
        codebook_accuracies.append(accuracy)
    
    # 计算完全匹配准确率（所有码本都正确）
    full_matches = torch.all(codebook_indices_true == codebook_indices_pred, dim=1)
    full_accuracy = torch.sum(full_matches).item() / batch_size
    
    # 计算平均码本准确率
    mean_codebook_accuracy = np.mean(codebook_accuracies)
    
    return {
        'full_accuracy': full_accuracy,
        'mean_codebook_accuracy': mean_codebook_accuracy,
        'codebook_accuracies': codebook_accuracies
    }

def compute_nearest_neighbors(embeddings, query_embeddings, k=10):
    """
    计算最近邻
    
    Args:
        embeddings (np.ndarray): 候选embedding库，形状为 [n, dim]
        query_embeddings (np.ndarray): 查询embedding，形状为 [m, dim]
        k (int): 返回前k个最近邻
    
    Returns:
        tuple:
            - np.ndarray: 距离矩阵，形状为 [m, k]
            - np.ndarray: 索引矩阵，形状为 [m, k]
    """
    # 转换为numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    
    # 归一化向量以使用余弦相似度
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    # 计算余弦相似度（注意：spatial.distance.cdist返回的是距离，不是相似度）
    distances = spatial.distance.cdist(query_embeddings, embeddings, metric='cosine')
    
    # 获取前k个最近邻
    indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, indices, axis=1)
    
    return nearest_distances, indices

def compute_mean_reciprocal_rank(indices, ground_truth_indices):
    """
    计算平均倒数排名（MRR）
    
    Args:
        indices (np.ndarray): 预测的索引矩阵，形状为 [m, k]
        ground_truth_indices (np.ndarray): 真实的索引，形状为 [m]
    
    Returns:
        float: MRR值
    """
    mrr_scores = []
    
    for i in range(len(ground_truth_indices)):
        # 查找真实索引在预测中的位置
        ranks = np.where(indices[i] == ground_truth_indices[i])[0]
        if len(ranks) > 0:
            # 排名是位置+1
            mrr_scores.append(1.0 / (ranks[0] + 1))
        else:
            mrr_scores.append(0.0)
    
    return np.mean(mrr_scores)

def compute_precision_recall(indices, ground_truth_indices, k_list=None):
    """
    计算精确率和召回率
    
    Args:
        indices (np.ndarray): 预测的索引矩阵，形状为 [m, k_max]
        ground_truth_indices (np.ndarray): 真实的索引，形状为 [m]
        k_list (list): 不同的k值列表
    
    Returns:
        dict: 包含不同k值的精确率和召回率
    """
    if k_list is None:
        k_list = [1, 5, 10]
    
    results = {}
    
    for k in k_list:
        if k > indices.shape[1]:
            print(f"警告: k={k} 大于预测的最大长度 {indices.shape[1]}")
            k = indices.shape[1]
        
        precision_scores = []
        recall_scores = []
        
        for i in range(len(ground_truth_indices)):
            # 检查前k个预测中是否包含真实索引
            top_k_indices = indices[i, :k]
            is_relevant = ground_truth_indices[i] in top_k_indices
            
            # 精确率 = 相关项目数 / 检索到的项目数
            precision = 1.0 * is_relevant / k
            
            # 召回率 = 相关项目数 / 所有相关项目数
            recall = 1.0 * is_relevant
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        results[f'precision@{k}'] = np.mean(precision_scores)
        results[f'recall@{k}'] = np.mean(recall_scores)
    
    return results

def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model (nn.Module): 要评估的模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 运行设备
    
    Returns:
        dict: 评估指标字典
    """
    import torch.nn.functional as F
    
    model.eval()
    
    total_mse_loss = 0
    total_cosine_loss = 0
    
    all_original = []
    all_reconstructed = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # 前向传播
            x_recon, _, _, _, _ = model(batch)
            
            # 计算MSE损失
            mse_loss = F.mse_loss(x_recon, batch)
            total_mse_loss += mse_loss.item() * batch.size(0)
            
            # 计算余弦损失
            cosine_sim = F.cosine_similarity(x_recon, batch, dim=1)
            cosine_loss = 1 - cosine_sim.mean()
            total_cosine_loss += cosine_loss.item() * batch.size(0)
            
            # 保存用于进一步分析
            all_original.append(batch.cpu())
            all_reconstructed.append(x_recon.cpu())
    
    # 计算平均损失
    dataset_size = len(dataloader.dataset)
    avg_mse_loss = total_mse_loss / dataset_size
    avg_cosine_loss = total_cosine_loss / dataset_size
    
    # 合并所有批次的数据
    all_original = torch.cat(all_original)
    all_reconstructed = torch.cat(all_reconstructed)
    
    # 计算其他统计信息
    # 计算归一化均方根误差 (NRMSE)
    rmse = torch.sqrt(F.mse_loss(all_reconstructed, all_original))
    data_range = all_original.max() - all_original.min()
    nrmse = rmse / data_range
    
    # 计算皮尔逊相关系数
    original_flat = all_original.view(-1).numpy()
    reconstructed_flat = all_reconstructed.view(-1).numpy()
    pearson_corr = np.corrcoef(original_flat, reconstructed_flat)[0, 1]
    
    return {
        'mse_loss': avg_mse_loss,
        'cosine_loss': avg_cosine_loss,
        'rmse': rmse.item(),
        'nrmse': nrmse.item(),
        'pearson_correlation': pearson_corr
    }

# 导入必要的模块
try:
    import torch.nn.functional as F
except ImportError:
    pass  # 延迟导入，避免循环依赖