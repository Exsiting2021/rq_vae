#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ-VAE模型评估脚本
用于评估训练好的RQVAE模型性能
"""

import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from rqvae.models import RQVAE
from rqvae.data import create_dataloaders, create_large_scale_dataloaders, EmbeddingDataset
from rqvae.utils.config import load_config
from rqvae.utils.logger import setup_logger
from rqvae.utils.metrics import (
    evaluate_model, compute_nearest_neighbors, compute_mean_reciprocal_rank,
    compute_precision_recall, compute_reconstruction_loss
)
from rqvae.utils.visualization import (
    plot_reconstruction_examples, compare_embeddings, visualize_codebook_usage,
    visualize_embeddings, create_confusion_matrix
)
from rqvae.utils.sid_extractor import SIDExtractor

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='评估RQ-VAE模型')
    parser.add_argument('--model-path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='测试数据目录路径')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载器的工作线程数')
    parser.add_argument('--large-scale', action='store_true',
                        help='使用大规模数据加载器')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化结果')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='用于可视化的样本数量')
    return parser.parse_args()

def compute_codebook_usage_stats(sids):
    """
    计算码本使用统计信息
    
    Args:
        sids (np.ndarray): SID数组，形状为 [n, num_codebooks]
    
    Returns:
        dict: 码本使用统计信息
    """
    num_codebooks = sids.shape[1]
    stats = {}
    
    for i in range(num_codebooks):
        # 统计每个码本的使用频率
        unique, counts = np.unique(sids[:, i], return_counts=True)
        
        # 计算统计信息
        usage_count = len(unique)
        total_entries = len(sids)
        coverage = usage_count / sids[:, i].max() if sids[:, i].max() > 0 else 0
        
        # 计算熵
        probabilities = counts / total_entries
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        stats[f'codebook_{i}'] = {
            'unique_entries': usage_count,
            'coverage': coverage,
            'entropy': entropy,
            'most_common': int(unique[np.argmax(counts)]),
            'most_common_count': int(counts.max()),
            'least_common': int(unique[np.argmin(counts)]),
            'least_common_count': int(counts.min())
        }
    
    # 计算总体统计信息
    stats['overall'] = {
        'total_sids': len(sids),
        'unique_sids': len(np.unique(sids, axis=0)),
        'sid_diversity': len(np.unique(sids, axis=0)) / len(sids) if len(sids) > 0 else 0
    }
    
    return stats

def compute_retrieval_metrics(embeddings, reconstructed_embeddings, sample_size=1000):
    """
    计算检索性能指标
    
    Args:
        embeddings (np.ndarray): 原始嵌入
        reconstructed_embeddings (np.ndarray): 重建嵌入
        sample_size (int): 用于评估的样本数量
    
    Returns:
        dict: 检索性能指标
    """
    # 采样以加速计算
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        query_embeddings = embeddings[indices]
        query_reconstructed = reconstructed_embeddings[indices]
    else:
        indices = np.arange(len(embeddings))
        query_embeddings = embeddings
        query_reconstructed = reconstructed_embeddings
    
    # 计算最近邻
    print(f"计算检索性能指标，查询数量: {len(query_embeddings)}, 数据库大小: {len(embeddings)}")
    
    # 使用原始嵌入作为查询，在原始嵌入数据库中搜索
    orig_distances, orig_indices = compute_nearest_neighbors(
        embeddings, query_embeddings, k=10
    )
    
    # 使用重建嵌入作为查询，在原始嵌入数据库中搜索
    recon_distances, recon_indices = compute_nearest_neighbors(
        embeddings, query_reconstructed, k=10
    )
    
    # 计算MRR
    orig_mrr = compute_mean_reciprocal_rank(orig_indices, indices)
    recon_mrr = compute_mean_reciprocal_rank(recon_indices, indices)
    
    # 计算精确率和召回率
    orig_precision_recall = compute_precision_recall(orig_indices, indices, k_list=[1, 5, 10])
    recon_precision_recall = compute_precision_recall(recon_indices, indices, k_list=[1, 5, 10])
    
    # 计算检索一致性（两个检索结果的排名差异）
    consistency_scores = []
    for i in range(len(query_embeddings)):
        # 找到查询在原始检索结果中的位置
        orig_pos = np.where(orig_indices[i] == indices[i])[0]
        if len(orig_pos) > 0:
            orig_pos = orig_pos[0]
        else:
            orig_pos = 10  # 如果不在前10名，视为位置10
        
        # 找到查询在重建检索结果中的位置
        recon_pos = np.where(recon_indices[i] == indices[i])[0]
        if len(recon_pos) > 0:
            recon_pos = recon_pos[0]
        else:
            recon_pos = 10  # 如果不在前10名，视为位置10
        
        # 计算位置差异的倒数（越小越好）
        consistency = 1.0 / (abs(orig_pos - recon_pos) + 1)
        consistency_scores.append(consistency)
    
    retrieval_metrics = {
        'original_mrr': orig_mrr,
        'reconstructed_mrr': recon_mrr,
        'mrr_ratio': recon_mrr / orig_mrr if orig_mrr > 0 else 0,
        'retrieval_consistency': np.mean(consistency_scores)
    }
    
    # 添加精确率和召回率
    for k in [1, 5, 10]:
        retrieval_metrics[f'original_precision@{k}'] = orig_precision_recall[f'precision@{k}']
        retrieval_metrics[f'reconstructed_precision@{k}'] = recon_precision_recall[f'precision@{k}']
        retrieval_metrics[f'original_recall@{k}'] = orig_precision_recall[f'recall@{k}']
        retrieval_metrics[f'reconstructed_recall@{k}'] = recon_precision_recall[f'recall@{k}']
    
    return retrieval_metrics

def main():
    """
    主评估函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'evaluation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 设置日志
    logger = setup_logger('RQVAE-Evaluator', 
                         log_file=os.path.join(output_dir, 'evaluation.log'))
    
    # 打印评估信息
    logger.info('开始评估RQ-VAE模型')
    logger.info(f'模型路径: {args.model_path}')
    logger.info(f'数据目录: {args.data_dir}')
    logger.info(f'输出目录: {output_dir}')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 创建数据加载器
    logger.info('创建数据加载器...')
    try:
        # 为评估创建配置
        eval_config = {
            'data': {
                'data_dir': args.data_dir,
                'num_workers': args.num_workers,
                'val_split': 0.0,  # 不分割验证集
                'test_split': 0.0,  # 使用所有数据进行评估
                'shuffle': False,  # 不打乱顺序
                'pin_memory': True
            }
        }
        
        if args.large_scale:
            train_loader, _, _ = create_large_scale_dataloaders(
                data_dir=eval_config['data']['data_dir'],
                batch_size=args.batch_size,
                num_workers=eval_config['data']['num_workers'],
                val_split=eval_config['data']['val_split'],
                test_split=eval_config['data']['test_split'],
                shuffle=eval_config['data']['shuffle'],
                pin_memory=eval_config['data']['pin_memory']
            )
        else:
            train_loader, _, _ = create_dataloaders(
                data_dir=eval_config['data']['data_dir'],
                batch_size=args.batch_size,
                num_workers=eval_config['data']['num_workers'],
                val_split=eval_config['data']['val_split'],
                test_split=eval_config['data']['test_split'],
                shuffle=eval_config['data']['shuffle'],
                pin_memory=eval_config['data']['pin_memory']
            )
        
        # 由于不分割，train_loader实际上包含所有数据
        data_loader = train_loader
        logger.info(f'评估数据集大小: {len(data_loader.dataset)}')
    except Exception as e:
        logger.error(f'创建数据加载器失败: {e}')
        raise
    
    # 创建SID提取器
    logger.info('加载模型并创建SID提取器...')
    try:
        extractor = SIDExtractor(args.model_path, args.config)
        model = extractor.model
        
        # 获取模型配置
        embedding_dim = model.embedding_dim
        latent_dim = model.latent_dim
        num_codebooks = model.quantizer.num_codebooks
        codebook_size = model.quantizer.codebook_size
        
        logger.info(f'模型配置: 嵌入维度={embedding_dim}, 潜在维度={latent_dim}, '
                   f'码本数量={num_codebooks}, 码本大小={codebook_size}')
    except Exception as e:
        logger.error(f'加载模型失败: {e}')
        raise
    
    # 执行评估
    logger.info('开始执行模型评估...')
    
    # 1. 计算基本重建指标
    logger.info('计算重建指标...')
    reconstruction_metrics = evaluate_model(model, data_loader, device)
    logger.info('重建指标:')
    for key, value in reconstruction_metrics.items():
        logger.info(f'  {key}: {value:.6f}')
    
    # 2. 提取所有数据的SID并计算码本使用统计
    logger.info('提取SID并计算码本使用统计...')
    
    # 收集所有嵌入和重建嵌入以进行进一步分析
    all_embeddings = []
    all_reconstructed = []
    all_sids = []
    
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            
            # 前向传播
            x_recon, _, _, codebook_indices, _ = model(batch)
            
            # 保存结果
            all_embeddings.append(batch.cpu().numpy())
            all_reconstructed.append(x_recon.cpu().numpy())
            all_sids.append(codebook_indices.cpu().numpy())
            
            # 记录进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                logger.info(f'处理批次: {batch_idx + 1}/{len(data_loader)}')
    
    # 合并所有批次
    all_embeddings = np.concatenate(all_embeddings)
    all_reconstructed = np.concatenate(all_reconstructed)
    all_sids = np.concatenate(all_sids)
    
    # 计算码本使用统计
    codebook_stats = compute_codebook_usage_stats(all_sids)
    logger.info('码本使用统计:')
    logger.info(f"  唯一SID数量: {codebook_stats['overall']['unique_sids']}")
    logger.info(f"  SID多样性: {codebook_stats['overall']['sid_diversity']:.6f}")
    
    # 3. 计算检索性能指标
    logger.info('计算检索性能指标...')
    try:
        # 限制样本大小以避免内存问题
        sample_size = min(args.sample_size, len(all_embeddings))
        retrieval_metrics = compute_retrieval_metrics(
            all_embeddings, all_reconstructed, sample_size=sample_size
        )
        
        logger.info('检索性能指标:')
        logger.info(f"  原始MRR: {retrieval_metrics['original_mrr']:.6f}")
        logger.info(f"  重建MRR: {retrieval_metrics['reconstructed_mrr']:.6f}")
        logger.info(f"  MRR比率: {retrieval_metrics['mrr_ratio']:.6f}")
        logger.info(f"  检索一致性: {retrieval_metrics['retrieval_consistency']:.6f}")
        
        for k in [1, 5, 10]:
            logger.info(f"  重建精确率@{k}: {retrieval_metrics[f'reconstructed_precision@{k}']:.6f}")
            logger.info(f"  重建召回率@{k}: {retrieval_metrics[f'reconstructed_recall@{k}']:.6f}")
    except Exception as e:
        logger.error(f'计算检索性能指标失败: {e}')
        retrieval_metrics = {}
    
    # 4. 生成可视化结果（如果启用）
    if args.visualize:
        logger.info('生成可视化结果...')
        try:
            # 采样以加速可视化
            sample_size_vis = min(1000, len(all_embeddings))
            vis_indices = np.random.choice(len(all_embeddings), sample_size_vis, replace=False)
            sampled_embeddings = all_embeddings[vis_indices]
            sampled_reconstructed = all_reconstructed[vis_indices]
            sampled_sids = all_sids[vis_indices]
            
            # 4.1 嵌入对比可视化
            compare_path = os.path.join(output_dir, 'visualizations', 'embedding_comparison.png')
            compare_embeddings(sampled_embeddings, sampled_reconstructed, 
                              n_samples=500, save_path=compare_path)
            logger.info(f'嵌入对比可视化已保存到: {compare_path}')
            
            # 4.2 重建示例
            recon_examples_path = os.path.join(output_dir, 'visualizations', 'reconstruction_examples.png')
            plot_reconstruction_examples(sampled_embeddings, sampled_reconstructed, 
                                        n_examples=5, save_path=recon_examples_path)
            logger.info(f'重建示例已保存到: {recon_examples_path}')
            
            # 4.3 码本使用情况
            codebook_usage_path = os.path.join(output_dir, 'visualizations', 'codebook_usage.png')
            visualize_codebook_usage(sampled_sids, num_codebooks, save_path=codebook_usage_path)
            logger.info(f'码本使用情况已保存到: {codebook_usage_path}')
            
            # 4.4 潜在空间可视化（如果启用）
            if latent_dim > 2:
                # 对于高维潜在空间，使用降维可视化
                # 我们需要重新计算量化后的潜在表示
                logger.info('生成潜在空间可视化...')
                
                # 采样用于可视化的批次
                vis_batch = torch.from_numpy(sampled_embeddings[:500]).float().to(device)
                with torch.no_grad():
                    _, quantized, _, _, _ = model(vis_batch)
                quantized_np = quantized.cpu().numpy()
                
                # t-SNE可视化
                tsne_path = os.path.join(output_dir, 'visualizations', 'latent_tsne.png')
                visualize_embeddings(quantized_np, method='tsne', n_samples=500, save_path=tsne_path)
                logger.info(f'潜在空间t-SNE可视化已保存到: {tsne_path}')
                
                # PCA可视化
                pca_path = os.path.join(output_dir, 'visualizations', 'latent_pca.png')
                visualize_embeddings(quantized_np, method='pca', n_samples=500, save_path=pca_path)
                logger.info(f'潜在空间PCA可视化已保存到: {pca_path}')
                
        except Exception as e:
            logger.error(f'生成可视化结果失败: {e}')
    
    # 5. 保存评估结果
    logger.info('保存评估结果...')
    
    # 合并所有指标
    all_metrics = {
        'reconstruction_metrics': reconstruction_metrics,
        'codebook_stats': codebook_stats,
        'retrieval_metrics': retrieval_metrics,
        'model_info': {
            'embedding_dim': embedding_dim,
            'latent_dim': latent_dim,
            'num_codebooks': num_codebooks,
            'codebook_size': codebook_size,
            'model_path': args.model_path
        },
        'eval_info': {
            'timestamp': timestamp,
            'dataset_size': len(all_embeddings),
            'device': str(device)
        }
    }
    
    # 保存为YAML文件
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.yaml')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_metrics, f, default_flow_style=False, allow_unicode=True)
    
    # 保存为JSON格式（更适合程序化读取）
    import json
    metrics_json_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型以支持JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(all_metrics), f, ensure_ascii=False, indent=2)
    
    logger.info(f'评估结果已保存到: {metrics_path} 和 {metrics_json_path}')
    
    # 6. 导出码本
    logger.info('导出码本...')
    codebooks_dir = os.path.join(output_dir, 'codebooks')
    extractor.export_codebooks(codebooks_dir)
    
    # 7. 保存示例SID和重建结果
    logger.info('保存示例SID和重建结果...')
    
    # 保存前100个示例的SID
    sample_sids_path = os.path.join(output_dir, 'sample_sids.npy')
    np.save(sample_sids_path, all_sids[:100])
    
    # 保存前100个示例的原始和重建嵌入
    sample_embeddings_path = os.path.join(output_dir, 'sample_embeddings.npz')
    np.savez_compressed(
        sample_embeddings_path,
        original=all_embeddings[:100],
        reconstructed=all_reconstructed[:100],
        sids=all_sids[:100]
    )
    
    logger.info('评估完成！')
    
    # 打印总结
    print("\n评估总结:")
    print(f"模型路径: {args.model_path}")
    print(f"数据集大小: {len(all_embeddings)}")
    print(f"\n重建指标:")
    print(f"  MSE损失: {reconstruction_metrics['mse_loss']:.6f}")
    print(f"  余弦损失: {reconstruction_metrics['cosine_loss']:.6f}")
    print(f"  RMSE: {reconstruction_metrics['rmse']:.6f}")
    print(f"  NRMSE: {reconstruction_metrics['nrmse']:.6f}")
    print(f"  皮尔逊相关系数: {reconstruction_metrics['pearson_correlation']:.6f}")
    
    print(f"\nSID统计:")
    print(f"  唯一SID数量: {codebook_stats['overall']['unique_sids']}")
    print(f"  SID多样性: {codebook_stats['overall']['sid_diversity']:.6f}")
    
    if retrieval_metrics:
        print(f"\n检索性能:")
        print(f"  原始MRR: {retrieval_metrics['original_mrr']:.6f}")
        print(f"  重建MRR: {retrieval_metrics['reconstructed_mrr']:.6f}")
        print(f"  MRR比率: {retrieval_metrics['mrr_ratio']:.6f}")
    
    print(f"\n结果保存在: {output_dir}")

if __name__ == '__main__':
    main()