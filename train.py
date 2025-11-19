#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ-VAE训练脚本
用于训练Residual Quantized VAE模型以提取语义标识符(SID)
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
from rqvae.trainer import Trainer
from rqvae.utils.config import load_config
from rqvae.utils.logger import setup_logger
from rqvae.utils.metrics import evaluate_model
from rqvae.utils.visualization import (
    plot_training_history, visualize_embeddings, compare_embeddings,
    visualize_codebook_usage, plot_reconstruction_examples
)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='训练RQ-VAE模型')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, 
                        help='从指定的检查点恢复训练')
    parser.add_argument('--data-dir', '--data_dir', type=str, default=None, 
                        help='数据目录路径，覆盖配置文件中的设置')
    parser.add_argument('--output-dir', '--output_dir', type=str, default='outputs', 
                        help='输出目录路径')
    parser.add_argument('--large-scale', action='store_true', 
                        help='使用大规模数据加载器')
    parser.add_argument('--load-item-ids', action='store_true', 
                        help='从数据文件中加载item ids')
    return parser.parse_args()

def main():
    """
    主训练函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置（如果命令行提供了）
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    
    # 确保训练配置包含必要的参数
    if 'training' not in config:
        config['training'] = {}
    
    # 设置默认训练参数
    default_training_params = {
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'betas': [0.9, 0.999],
        'patience': 10,
        'save_interval': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 更新配置
    for key, value in default_training_params.items():
        if key not in config['training']:
            config['training'][key] = value
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'rqvae_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 设置日志
    logger = setup_logger(
        log_dir=output_dir,  # 日志目录
        log_name='RQVAE-Trainer'  # 日志名称
    )
    
    # 打印配置信息
    logger.info('开始训练RQ-VAE模型')
    logger.info(f'输出目录: {output_dir}')
    logger.info(f'配置文件: {args.config}')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 创建数据加载器
    logger.info('创建数据加载器...')
    try:
        # 创建自定义的EmbeddingDataset以支持load_item_ids参数
        dataset = EmbeddingDataset(
            data_path=config['data']['data_dir'],
            embedding_dim=config['data']['embedding_dim'],
            is_memory_mapped=args.large_scale,
            load_item_ids=args.load_item_ids
        )
        
        # 计算分割大小
        dataset_size = len(dataset)
        train_split = config['data']['train_split']
        val_split = config['data']['val_split']
        test_split = 1.0 - train_split - val_split
        
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # 确保测试集不为空
        if test_size <= 0:
            test_size = 1
            val_size = max(1, val_size - 1)
            train_size = dataset_size - val_size - test_size
        
        # 分割数据集
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=config['data']['shuffle'],
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f'训练集大小: {len(train_loader.dataset)}')
        if val_loader is not None:
            logger.info(f'验证集大小: {len(val_loader.dataset)}')
        if test_loader is not None:
            logger.info(f'测试集大小: {len(test_loader.dataset)}')
    except Exception as e:
        logger.error(f'创建数据加载器失败: {e}')
        raise
    
    # 创建模型
    logger.info('创建RQ-VAE模型...')
    try:
        # 从配置中提取参数
        input_dim = config['data']['embedding_dim']
        latent_dim = config['model']['quantizer']['latent_dim']
        num_codebooks = config['model']['quantizer']['num_codebooks']
        codebook_size = config['model']['quantizer']['codebook_size']
        encoder_hidden_dims = config['model']['encoder']['hidden_dims']
        decoder_hidden_dims = config['model']['decoder']['hidden_dims']
        activation = config['model']['encoder']['activation']
        dropout = config['model']['encoder']['dropout']
        commitment_cost = config['model']['quantizer']['commitment_cost']
        decay = config['model']['quantizer'].get('decay', 0.99)
        
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
        
        logger.info(f'模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    except Exception as e:
        logger.error(f'创建模型失败: {e}')
        raise
    
    # 创建训练器
    logger.info('创建训练器...')
    try:
        # 更新配置
        config['training']['device'] = str(device)
        config['training']['checkpoint_dir'] = os.path.join(output_dir, 'checkpoints')
        config['training']['log_dir'] = output_dir
        
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
    except Exception as e:
        logger.error(f'创建训练器失败: {e}')
        raise
    
    # 开始训练
    logger.info('开始训练...')
    try:
        if args.resume is not None:
            logger.info(f'从检查点恢复训练: {args.resume}')
            start_epoch = trainer.load_checkpoint(args.resume)
            trainer.train(start_epoch=start_epoch)
        else:
            trainer.train(start_epoch=0)
    except KeyboardInterrupt:
        logger.info('训练被用户中断')
    except Exception as e:
        logger.error(f'训练失败: {e}')
        raise
    
    # 评估模型
    if test_loader is not None:
        logger.info('在测试集上评估模型...')
        try:
            # 加载最佳模型
            best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pt')
            if os.path.exists(best_model_path):
                trainer.load_checkpoint(best_model_path)
                logger.info(f'加载最佳模型: {best_model_path}')
            
            test_metrics = evaluate_model(model, test_loader, device)
            logger.info('测试集评估结果:')
            for key, value in test_metrics.items():
                logger.info(f'  {key}: {value:.6f}')
            
            # 保存测试结果
            with open(os.path.join(output_dir, 'test_results.yaml'), 'w', encoding='utf-8') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
                
        except Exception as e:
            logger.error(f'评估模型失败: {e}')
    
    # 可视化结果（可选）
    try:
        logger.info('生成可视化结果...')
        
        # 从测试集中获取一些样本
        sample_batch = next(iter(test_loader))
        sample_batch = sample_batch.to(device)
        
        # 进行预测
        with torch.no_grad():
            model.eval()
            x_recon, quantized, _, codebook_indices, _ = model(sample_batch)
        
        # 转换为numpy数组
        sample_batch_np = sample_batch.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()
        codebook_indices_np = codebook_indices.cpu().numpy()
        
        # 可视化原始嵌入和重建嵌入的对比
        compare_path = os.path.join(output_dir, 'visualizations', 'embedding_comparison.png')
        compare_embeddings(sample_batch_np, x_recon_np, n_samples=200, save_path=compare_path)
        logger.info(f'嵌入对比可视化已保存到: {compare_path}')
        
        # 可视化重建示例
        recon_examples_path = os.path.join(output_dir, 'visualizations', 'reconstruction_examples.png')
        plot_reconstruction_examples(sample_batch_np, x_recon_np, n_examples=5, save_path=recon_examples_path)
        logger.info(f'重建示例已保存到: {recon_examples_path}')
        
        # 可视化码本使用情况
        codebook_usage_path = os.path.join(output_dir, 'visualizations', 'codebook_usage.png')
        visualize_codebook_usage(codebook_indices_np, config['model']['quantizer']['num_codebooks'], save_path=codebook_usage_path)
        logger.info(f'码本使用情况已保存到: {codebook_usage_path}')
        
        # 可视化嵌入分布
        if config['model']['latent_dim'] > 2:
            # 对于高维潜在空间，使用降维可视化
            tsne_path = os.path.join(output_dir, 'visualizations', 'latent_tsne.png')
            visualize_embeddings(quantized.cpu().numpy(), method='tsne', n_samples=500, save_path=tsne_path)
            logger.info(f'潜在空间t-SNE可视化已保存到: {tsne_path}')
            
            pca_path = os.path.join(output_dir, 'visualizations', 'latent_pca.png')
            visualize_embeddings(quantized.cpu().numpy(), method='pca', n_samples=500, save_path=pca_path)
            logger.info(f'潜在空间PCA可视化已保存到: {pca_path}')
            
    except Exception as e:
        logger.error(f'生成可视化结果失败: {e}')
    
    logger.info('训练完成！')

if __name__ == '__main__':
    main()