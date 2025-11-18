#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ-VAE使用示例脚本
展示如何使用RQVAE模型进行训练、评估和应用
"""

import os
import sys
import numpy as np
import torch
import time

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 导入项目模块
from rqvae.models import RQVAE
from rqvae.data import EmbeddingDataset, create_dataloaders
from rqvae.trainer import Trainer, train_from_scratch, resume_training
from rqvae.utils.config import load_config, save_config
from rqvae.utils.logger import setup_logger
from rqvae.utils.sid_extractor import SIDExtractor


def create_demo_data(output_dir, num_samples=10000, embedding_dim=512):
    """
    创建演示数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        embedding_dim: 嵌入维度
        
    Returns:
        str: 数据文件路径
    """
    print(f"创建演示数据集，包含 {num_samples} 个 {embedding_dim} 维嵌入向量...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成随机嵌入向量（在实际应用中，这应该是真实的多模态嵌入）
    # 为了演示效果，我们生成一些有结构的数据
    np.random.seed(42)
    
    # 生成一些聚类中心
    num_clusters = 10
    centers = np.random.randn(num_clusters, embedding_dim)
    
    # 围绕聚类中心生成样本
    embeddings = []
    for center in centers:
        # 每个聚类生成1000个样本，添加一些噪声
        cluster_samples = center + 0.5 * np.random.randn(num_samples // num_clusters, embedding_dim)
        embeddings.append(cluster_samples)
    
    # 合并所有样本
    embeddings = np.vstack(embeddings)
    
    # 如果样本数量不足，添加随机样本
    if len(embeddings) < num_samples:
        additional_samples = np.random.randn(num_samples - len(embeddings), embedding_dim)
        embeddings = np.vstack([embeddings, additional_samples])
    
    # 归一化到单位球面上（这通常是多模态嵌入的常见处理）
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    embeddings = embeddings / norms
    
    # 保存数据
    data_path = os.path.join(output_dir, 'demo_embeddings.npy')
    np.save(data_path, embeddings)
    
    print(f"演示数据集已保存到: {data_path}")
    print(f"数据集形状: {embeddings.shape}")
    return data_path


def train_example_model(data_path, config_path, output_dir):
    """
    训练示例模型
    
    Args:
        data_path: 数据文件路径
        config_path: 配置文件路径
        output_dir: 输出目录
        
    Returns:
        str: 模型保存路径
    """
    print("\n开始训练示例模型...")
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建数据加载器
    data_dir = os.path.dirname(data_path)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        val_split=0.1,
        test_split=0.1,
        shuffle=True
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 初始化模型
    model = RQVAE(
        embedding_dim=config['data']['embedding_dim'],
        latent_dim=config['model']['latent_dim'],
        encoder_hidden_dims=config['model']['encoder_hidden_dims'],
        decoder_hidden_dims=config['model']['decoder_hidden_dims'],
        num_codebooks=config['model']['num_codebooks'],
        codebook_size=config['model']['codebook_size'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir
    )
    
    # 开始训练
    try:
        trainer.train()
        print("训练完成！")
    except KeyboardInterrupt:
        print("训练被中断")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        raise
    
    # 返回最佳模型路径
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"最佳模型已保存到: {best_model_path}")
        return best_model_path
    else:
        print("未找到最佳模型，返回最后保存的模型")
        return os.path.join(output_dir, 'model_latest.pth')


def extract_sids_example(model_path, data_path, output_dir):
    """
    提取SID示例
    
    Args:
        model_path: 模型路径
        data_path: 数据文件路径
        output_dir: 输出目录
        
    Returns:
        str: SID文件路径
    """
    print("\n开始提取SID...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    embeddings = np.load(data_path)
    print(f"加载了 {len(embeddings)} 个嵌入向量")
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path)
    
    # 提取SID
    start_time = time.time()
    sids = extractor.extract_sids(embeddings, batch_size=1024)
    end_time = time.time()
    
    print(f"SID提取完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"SID形状: {sids.shape}")
    
    # 保存SID
    sids_path = os.path.join(output_dir, 'extracted_sids.npy')
    np.save(sids_path, sids)
    print(f"SID已保存到: {sids_path}")
    
    # 显示一些SID示例
    print("\nSID示例:")
    for i in range(min(5, len(sids))):
        print(f"  样本 {i}: {sids[i]}")
    
    return sids_path


def reconstruct_embeddings_example(model_path, sids_path, output_dir):
    """
    从SID重建嵌入向量示例
    
    Args:
        model_path: 模型路径
        sids_path: SID文件路径
        output_dir: 输出目录
        
    Returns:
        str: 重建嵌入向量文件路径
    """
    print("\n从SID重建嵌入向量...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载SID
    sids = np.load(sids_path)
    print(f"加载了 {len(sids)} 个SID")
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path)
    
    # 重建嵌入向量
    start_time = time.time()
    reconstructed_embeddings = extractor.reconstruct_from_sids(sids, batch_size=1024)
    end_time = time.time()
    
    print(f"嵌入向量重建完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"重建嵌入向量形状: {reconstructed_embeddings.shape}")
    
    # 保存重建嵌入向量
    recon_path = os.path.join(output_dir, 'reconstructed_embeddings.npy')
    np.save(recon_path, reconstructed_embeddings)
    print(f"重建嵌入向量已保存到: {recon_path}")
    
    return recon_path


def compute_similarity_example(model_path, data_path, output_dir):
    """
    计算相似度示例
    
    Args:
        model_path: 模型路径
        data_path: 数据文件路径
        output_dir: 输出目录
    """
    print("\n计算SID相似度示例...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    embeddings = np.load(data_path)
    print(f"加载了 {len(embeddings)} 个嵌入向量")
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path)
    
    # 提取所有SID作为数据库
    print("提取所有SID作为数据库...")
    all_sids = extractor.extract_sids(embeddings, batch_size=1024)
    
    # 选择几个查询向量
    query_indices = [0, 100, 200]
    query_embeddings = embeddings[query_indices]
    
    print(f"\n对 {len(query_embeddings)} 个查询向量进行相似度搜索:")
    
    for i, query_emb in enumerate(query_embeddings):
        # 提取查询向量的SID
        query_sid = extractor.extract_sids(np.array([query_emb]))[0]
        
        print(f"\n查询向量 {i} (索引 {query_indices[i]}):")
        print(f"  查询SID: {query_sid}")
        
        # 计算与数据库中所有SID的相似度
        similarities = extractor.compute_sid_similarity(query_sid, all_sids)
        
        # 获取前5个最相似的结果
        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        print(f"  前 {top_k} 个最相似的结果:")
        for j, (idx, sim) in enumerate(zip(top_indices, top_similarities)):
            print(f"    排名 {j+1}: 索引={idx}, 相似度={sim:.6f}, SID={all_sids[idx]}")
    
    # 演示批量处理
    print("\n批量处理示例:")
    batch_size = 10
    batch_embeddings = embeddings[:batch_size]
    batch_sids = extractor.extract_sids(batch_embeddings)
    
    print(f"处理了 {batch_size} 个嵌入向量，生成的SID形状: {batch_sids.shape}")


def generate_new_embeddings_example(model_path, output_dir, num_samples=10):
    """
    生成新的嵌入向量示例（用于推荐系统）
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        num_samples: 生成的样本数量
    """
    print("\n生成新的嵌入向量示例（用于推荐系统）...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path)
    
    # 获取模型配置
    num_codebooks = extractor.model.quantizer.num_codebooks
    codebook_size = extractor.model.quantizer.codebook_size
    
    print(f"模型配置: {num_codebooks} 个码本，每个码本大小为 {codebook_size}")
    
    # 方法1: 随机生成SID
    print("\n方法1: 随机生成SID...")
    np.random.seed(42)
    random_sids = np.random.randint(0, codebook_size, size=(num_samples, num_codebooks))
    
    # 从随机SID重建嵌入向量
    random_embeddings = extractor.reconstruct_from_sids(random_sids)
    print(f"生成了 {len(random_embeddings)} 个随机嵌入向量")
    
    # 方法2: 通过插值生成SID
    print("\n方法2: 通过插值生成SID（更适合推荐系统）...")
    
    # 选择几个基础SID进行插值
    base_sids = np.random.randint(0, codebook_size, size=(3, num_codebooks))
    interpolated_sids = []
    
    for i in range(num_samples):
        # 随机选择两个基础SID
        idx1, idx2 = np.random.choice(3, 2, replace=False)
        sid1, sid2 = base_sids[idx1], base_sids[idx2]
        
        # 在两个SID之间进行插值
        # 对于每个码本，我们随机选择其中一个码本中的索引
        interpolated_sid = []
        for j in range(num_codebooks):
            # 随机选择使用哪个码本的索引
            if np.random.random() > 0.5:
                interpolated_sid.append(sid1[j])
            else:
                interpolated_sid.append(sid2[j])
        interpolated_sids.append(interpolated_sid)
    
    interpolated_sids = np.array(interpolated_sids)
    
    # 从插值生成的SID重建嵌入向量
    interpolated_embeddings = extractor.reconstruct_from_sids(interpolated_sids)
    print(f"生成了 {len(interpolated_embeddings)} 个插值嵌入向量")
    
    # 保存生成的嵌入向量
    gen_path = os.path.join(output_dir, 'generated_embeddings.npz')
    np.savez_compressed(
        gen_path,
        random=random_embeddings,
        interpolated=interpolated_embeddings,
        random_sids=random_sids,
        interpolated_sids=interpolated_sids
    )
    
    print(f"生成的嵌入向量已保存到: {gen_path}")
    print("\n这些生成的嵌入向量可以用于推荐系统中的内容生成或多样化推荐")


def main():
    """
    主函数
    """
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'examples', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 创建演示数据
    data_path = create_demo_data(
        output_dir=os.path.join(output_dir, 'data'),
        num_samples=10000,
        embedding_dim=512
    )
    
    # 步骤2: 创建简化的配置文件
    print("\n创建简化的配置文件...")
    config = {
        'data': {
            'embedding_dim': 512
        },
        'model': {
            'latent_dim': 128,
            'encoder_hidden_dims': [256],
            'decoder_hidden_dims': [256],
            'num_codebooks': 4,
            'codebook_size': 256,
            'dropout': 0.1,
            'activation': 'relu'
        },
        'training': {
            'batch_size': 128,
            'num_workers': 4,
            'lr': 1e-4,
            'epochs': 10,
            'patience': 3,
            'kl_weight': 0.001,
            'codebook_weight': 1.0,
            'commitment_weight': 0.25,
            'ema_decay': 0.99
        }
    }
    
    config_path = os.path.join(output_dir, 'config.yaml')
    save_config(config, config_path)
    print(f"配置文件已保存到: {config_path}")
    
    # 注意: 在实际应用中，我们应该训练模型，但这会耗费时间
    # 由于这只是演示，我们可以跳过训练步骤，使用注释掉的代码来模拟训练
    # 或者用户可以根据需要取消注释并运行训练
    
    # 步骤3（可选）: 训练模型
    # model_path = train_example_model(data_path, config_path, os.path.join(output_dir, 'model'))
    
    # 模拟一个模型路径（在实际应用中，这应该是训练好的模型路径）
    print("\n注意: 由于时间限制，跳过了模型训练步骤")
    print("在实际应用中，您需要先训练模型，然后使用训练好的模型进行以下操作")
    print("请使用 'python train.py --config examples/output/config.yaml --data-dir examples/output/data' 来训练模型")
    
    # 这里我们模拟一个模型路径，在实际使用时请替换为真实的模型路径
    model_path = "path/to/trained/model.pth"
    
    # 以下步骤需要一个训练好的模型，在实际使用时请取消注释
    
    # 步骤4: 提取SID
    # sids_path = extract_sids_example(model_path, data_path, os.path.join(output_dir, 'sids'))
    
    # 步骤5: 从SID重建嵌入向量
    # if 'sids_path' in locals():
    #     recon_path = reconstruct_embeddings_example(model_path, sids_path, os.path.join(output_dir, 'reconstructed'))
    
    # 步骤6: 计算相似度
    # compute_similarity_example(model_path, data_path, os.path.join(output_dir, 'similarity'))
    
    # 步骤7: 生成新的嵌入向量（用于推荐系统）
    # generate_new_embeddings_example(model_path, os.path.join(output_dir, 'generated'))
    
    print("\n示例脚本运行完成！")
    print("请按照注释说明修改脚本，以使用您自己训练好的模型")
    print("\n完整工作流程:")
    print("1. 准备您的嵌入向量数据（GLMv4提取的多模态嵌入）")
    print("2. 训练RQVAE模型: python train.py --config config.yaml --data-dir your_data_dir")
    print("3. 评估模型: python evaluate.py --model-path your_model_path --data-dir your_test_data")
    print("4. 提取SID: python sid_tools.py extract --model-path your_model_path --input your_embeddings --output your_sids.npy")
    print("5. 从SID重建: python sid_tools.py reconstruct --model-path your_model_path --input your_sids.npy --output reconstructed_embeddings.npy")
    print("6. 计算相似度: python sid_tools.py similarity --model-path your_model_path --query query_embeddings --database database_embeddings --output similarity_results.json")


if __name__ == '__main__':
    main()