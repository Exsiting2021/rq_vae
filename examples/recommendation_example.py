#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用RQ-VAE SID的推荐系统示例
展示如何使用提取的SID实现内容推荐功能
"""

import os
import sys
import numpy as np
import json
import time
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 导入项目模块
from rqvae.utils.sid_extractor import SIDExtractor


def create_demo_recommendation_data(output_dir, num_items=10000, embedding_dim=512):
    """
    创建演示推荐系统数据集
    
    Args:
        output_dir: 输出目录
        num_items: 物品数量
        embedding_dim: 嵌入维度
        
    Returns:
        tuple: (物品嵌入路径, 物品元数据路径)
    """
    print(f"创建演示推荐系统数据集，包含 {num_items} 个物品...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成物品嵌入（模拟GLMv4提取的多模态嵌入）
    np.random.seed(42)
    
    # 创建一些类别，每个类别有不同的特征
    categories = ['电子产品', '服装', '食品', '书籍', '家居用品']
    category_centers = {}
    for i, category in enumerate(categories):
        # 为每个类别创建一个中心嵌入
        center = np.random.randn(embedding_dim)
        # 确保每个类别的中心有足够的区分度
        center[i * 100] = 5.0  # 在不同维度上设置强信号
        category_centers[category] = center
    
    # 生成物品数据
    items = []
    item_embeddings = []
    
    for i in range(num_items):
        # 随机选择一个类别
        category = np.random.choice(categories)
        center = category_centers[category]
        
        # 生成围绕类别中心的嵌入
        embedding = center + 0.5 * np.random.randn(embedding_dim)
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # 创建物品信息
        item = {
            'item_id': f'item_{i}',
            'title': f'{category}示例 {i}',
            'category': category,
            'price': round(np.random.uniform(10.0, 1000.0), 2),
            'rating': round(np.random.uniform(3.0, 5.0), 1),
            'popularity': np.random.randint(1, 1000)
        }
        
        items.append(item)
        item_embeddings.append(embedding)
    
    # 转换为numpy数组
    item_embeddings = np.array(item_embeddings)
    
    # 保存嵌入
    embeddings_path = os.path.join(output_dir, 'item_embeddings.npy')
    np.save(embeddings_path, item_embeddings)
    
    # 保存物品元数据
    metadata_path = os.path.join(output_dir, 'item_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"物品嵌入已保存到: {embeddings_path}")
    print(f"物品元数据已保存到: {metadata_path}")
    print(f"生成了 {len(items)} 个物品，类别分布:")
    
    # 统计类别分布
    from collections import Counter
    category_counts = Counter(item['category'] for item in items)
    for category, count in category_counts.items():
        print(f"  {category}: {count} ({count/num_items*100:.1f}%)")
    
    return embeddings_path, metadata_path


def build_sid_index(embeddings_path, model_path, output_dir):
    """
    构建SID索引
    
    Args:
        embeddings_path: 嵌入向量路径
        model_path: 模型路径
        output_dir: 输出目录
        
    Returns:
        tuple: (SID索引路径, 物品ID到索引的映射路径)
    """
    print("\n构建SID索引...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载嵌入向量
    embeddings = np.load(embeddings_path)
    num_items = len(embeddings)
    print(f"加载了 {num_items} 个物品的嵌入向量")
    
    # 创建SID提取器
    extractor = SIDExtractor(model_path)
    
    # 提取SID
    print("提取物品的SID...")
    start_time = time.time()
    sids = extractor.extract_sids(embeddings, batch_size=1024)
    end_time = time.time()
    
    print(f"SID提取完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"SID形状: {sids.shape}")
    
    # 构建简单的倒排索引（码本ID -> 物品索引列表）
    print("构建SID倒排索引...")
    inverted_index = defaultdict(list)
    
    # 对于每个码本，建立码本索引到物品的映射
    for item_idx in range(num_items):
        for codebook_idx in range(sids.shape[1]):
            code_value = sids[item_idx, codebook_idx]
            # 键格式: (码本索引, 码本值)
            key = (codebook_idx, code_value)
            inverted_index[key].append(item_idx)
    
    # 保存SID数组
    sids_path = os.path.join(output_dir, 'item_sids.npy')
    np.save(sids_path, sids)
    
    # 保存倒排索引
    index_path = os.path.join(output_dir, 'sid_inverted_index.json')
    # 转换为可JSON序列化的格式
    serializable_index = {}
    for (codebook, value), items in inverted_index.items():
        key = f"{codebook}:{value}"
        serializable_index[key] = items
    
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_index, f)
    
    print(f"SID数组已保存到: {sids_path}")
    print(f"倒排索引已保存到: {index_path}")
    print(f"倒排索引包含 {len(inverted_index)} 个条目")
    
    return sids_path, index_path


def load_recommendation_data(embeddings_path, metadata_path, sids_path=None):
    """
    加载推荐系统数据
    
    Args:
        embeddings_path: 嵌入向量路径
        metadata_path: 元数据路径
        sids_path: SID路径（可选）
        
    Returns:
        tuple: (嵌入向量, 元数据, SID（如果提供）)
    """
    print("\n加载推荐系统数据...")
    
    # 加载嵌入向量
    embeddings = np.load(embeddings_path)
    print(f"加载了 {len(embeddings)} 个物品的嵌入向量")
    
    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"加载了 {len(metadata)} 个物品的元数据")
    
    # 加载SID（如果提供）
    sids = None
    if sids_path and os.path.exists(sids_path):
        sids = np.load(sids_path)
        print(f"加载了 {len(sids)} 个物品的SID")
    
    return embeddings, metadata, sids


def simple_recommendation(query_item_idx, sids, k=10, alpha=0.7):
    """
    简单的基于SID的推荐算法
    
    Args:
        query_item_idx: 查询物品的索引
        sids: 所有物品的SID
        k: 返回的推荐数量
        alpha: 权重衰减因子（随着码本位置的增加，权重衰减）
        
    Returns:
        list: 推荐的物品索引列表
    """
    # 获取查询物品的SID
    query_sid = sids[query_item_idx]
    num_codebooks = len(query_sid)
    
    # 计算相似度分数
    scores = np.zeros(len(sids))
    
    # 对每个码本赋予不同的权重
    for codebook_idx in range(num_codebooks):
        # 计算权重（第一个码本权重最高，最后一个最低）
        weight = alpha ** (num_codebooks - codebook_idx - 1)
        
        # 计算该码本上的匹配
        mask = sids[:, codebook_idx] == query_sid[codebook_idx]
        scores[mask] += weight
    
    # 排除查询物品本身
    scores[query_item_idx] = -1
    
    # 获取前k个最高分的物品索引
    top_indices = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_indices]
    
    return top_indices, top_scores


def inverted_index_recommendation(query_item_idx, sids, inverted_index, k=10, alpha=0.7):
    """
    使用倒排索引进行更高效的推荐
    
    Args:
        query_item_idx: 查询物品的索引
        sids: 所有物品的SID
        inverted_index: 倒排索引
        k: 返回的推荐数量
        alpha: 权重衰减因子
        
    Returns:
        list: 推荐的物品索引列表
    """
    # 获取查询物品的SID
    query_sid = sids[query_item_idx]
    num_codebooks = len(query_sid)
    
    # 使用倒排索引快速找到候选物品
    candidate_scores = defaultdict(float)
    
    # 对每个码本查找候选物品
    for codebook_idx in range(num_codebooks):
        weight = alpha ** (num_codebooks - codebook_idx - 1)
        code_value = query_sid[codebook_idx]
        
        # 从倒排索引获取候选物品
        key = f"{codebook_idx}:{code_value}"
        candidates = inverted_index.get(key, [])
        
        # 为每个候选物品增加分数
        for item_idx in candidates:
            if item_idx != query_item_idx:  # 排除查询物品本身
                candidate_scores[item_idx] += weight
    
    # 排序候选物品
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 返回前k个结果
    top_indices = [idx for idx, _ in sorted_candidates[:k]]
    top_scores = [score for _, score in sorted_candidates[:k]]
    
    return np.array(top_indices), np.array(top_scores)


def hybrid_recommendation(query_item_idx, embeddings, sids, inverted_index=None, 
                          k=10, alpha=0.7, sid_weight=0.7):
    """
    混合推荐算法：结合SID相似度和原始嵌入相似度
    
    Args:
        query_item_idx: 查询物品的索引
        embeddings: 所有物品的嵌入向量
        sids: 所有物品的SID
        inverted_index: 倒排索引（可选）
        k: 返回的推荐数量
        alpha: SID权重衰减因子
        sid_weight: SID相似度在混合推荐中的权重（0-1之间）
        
    Returns:
        list: 推荐的物品索引列表
    """
    # 获取查询物品的嵌入
    query_embedding = embeddings[query_item_idx]
    
    # 计算原始嵌入的余弦相似度
    cosine_scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )
    
    # 计算SID相似度
    if inverted_index is not None:
        sid_indices, sid_scores = inverted_index_recommendation(
            query_item_idx, sids, inverted_index, k=len(sids), alpha=alpha
        )
    else:
        sid_indices, sid_scores = simple_recommendation(
            query_item_idx, sids, k=len(sids), alpha=alpha
        )
    
    # 归一化SID分数
    max_sid_score = np.max(sid_scores)
    if max_sid_score > 0:
        normalized_sid_scores = sid_scores / max_sid_score
    else:
        normalized_sid_scores = sid_scores
    
    # 创建完整的SID分数数组
    full_sid_scores = np.zeros(len(embeddings))
    full_sid_scores[sid_indices] = normalized_sid_scores
    
    # 混合分数
    hybrid_scores = sid_weight * full_sid_scores + (1 - sid_weight) * cosine_scores
    
    # 排除查询物品本身
    hybrid_scores[query_item_idx] = -1
    
    # 获取前k个最高分的物品索引
    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    top_scores = hybrid_scores[top_indices]
    
    return top_indices, top_scores


def generate_user_profile(user_history, sids, alpha=0.7):
    """
    根据用户历史生成用户偏好SID
    
    Args:
        user_history: 用户交互过的物品索引列表
        sids: 所有物品的SID
        alpha: 码本权重衰减因子
        
    Returns:
        dict: 用户偏好配置文件
    """
    if not user_history:
        return None
    
    num_codebooks = sids.shape[1]
    
    # 统计每个码本位置上的值分布
    codebook_preferences = []
    
    for codebook_idx in range(num_codebooks):
        # 计算该码本位置的权重
        weight = alpha ** (num_codebooks - codebook_idx - 1)
        
        # 统计该码本位置上每个值的出现次数
        value_counts = defaultdict(int)
        for item_idx in user_history:
            value = sids[item_idx, codebook_idx]
            value_counts[value] += weight
        
        codebook_preferences.append(value_counts)
    
    return codebook_preferences


def user_based_recommendation(user_profile, sids, k=10, min_matches=1):
    """
    基于用户偏好的推荐
    
    Args:
        user_profile: 用户偏好配置文件
        sids: 所有物品的SID
        k: 返回的推荐数量
        min_matches: 最小匹配次数
        
    Returns:
        list: 推荐的物品索引列表
    """
    if not user_profile:
        return [], []
    
    # 计算每个物品与用户偏好的匹配分数
    scores = np.zeros(len(sids))
    
    for item_idx in range(len(sids)):
        item_sid = sids[item_idx]
        match_count = 0
        score = 0.0
        
        # 检查每个码本位置
        for codebook_idx in range(len(item_sid)):
            value = item_sid[codebook_idx]
            
            # 检查是否在用户偏好中
            if value in user_profile[codebook_idx]:
                match_count += 1
                score += user_profile[codebook_idx][value]
        
        # 只有达到最小匹配次数的物品才考虑
        if match_count >= min_matches:
            scores[item_idx] = score
    
    # 获取前k个最高分的物品索引
    top_indices = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_indices]
    
    return top_indices, top_scores


def display_recommendations(recommended_indices, scores, metadata, query_item=None):
    """
    显示推荐结果
    
    Args:
        recommended_indices: 推荐的物品索引列表
        scores: 对应的分数
        metadata: 物品元数据
        query_item: 查询物品（可选）
    """
    print("\n推荐结果:")
    if query_item:
        print(f"基于物品 '{query_item['title']}' ({query_item['category']}) 的推荐:")
    
    for i, (idx, score) in enumerate(zip(recommended_indices, scores)):
        item = metadata[idx]
        print(f"{i+1}. {item['title']}")
        print(f"   类别: {item['category']}")
        print(f"   价格: ¥{item['price']}")
        print(f"   评分: {item['rating']}★")
        print(f"   热度: {item['popularity']}")
        print(f"   推荐分数: {score:.4f}")
        print()

def main():
    """
    主函数
    """
    # 创建输出目录
    output_dir = os.path.join(project_root, 'examples', 'recommendation_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 创建演示数据
    embeddings_path, metadata_path = create_demo_recommendation_data(
        output_dir=os.path.join(output_dir, 'data'),
        num_items=10000,
        embedding_dim=512
    )
    
    # 注意: 在实际应用中，我们需要一个训练好的RQVAE模型
    # 由于这只是演示，我们提供一个模拟的模型路径
    print("\n注意: 此示例需要一个训练好的RQVAE模型")
    print("在实际应用中，请使用您自己训练的模型")
    
    # 模拟模型路径（请替换为实际路径）
    model_path = "path/to/trained/rqvae_model.pth"
    
    # 以下步骤需要一个训练好的模型，在实际使用时请取消注释
    
    # 步骤2: 构建SID索引
    # sids_path, index_path = build_sid_index(
    #     embeddings_path=embeddings_path,
    #     model_path=model_path,
    #     output_dir=os.path.join(output_dir, 'index')
    # )
    
    # 为了演示，我们生成一些随机SID
    print("\n为演示目的，生成随机SID...")
    embeddings = np.load(embeddings_path)
    num_items = len(embeddings)
    num_codebooks = 4
    codebook_size = 256
    
    # 生成随机SID
    np.random.seed(42)
    sids = np.random.randint(0, codebook_size, size=(num_items, num_codebooks))
    sids_path = os.path.join(output_dir, 'index', 'item_sids.npy')
    os.makedirs(os.path.dirname(sids_path), exist_ok=True)
    np.save(sids_path, sids)
    
    # 构建简单的倒排索引
    inverted_index = defaultdict(list)
    for item_idx in range(num_items):
        for codebook_idx in range(num_codebooks):
            code_value = sids[item_idx, codebook_idx]
            key = f"{codebook_idx}:{code_value}"
            inverted_index[key].append(item_idx)
    
    index_path = os.path.join(output_dir, 'index', 'sid_inverted_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(dict(inverted_index), f)
    
    print(f"生成的随机SID已保存到: {sids_path}")
    print(f"倒排索引已保存到: {index_path}")
    
    # 步骤3: 加载数据
    embeddings, metadata, sids = load_recommendation_data(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        sids_path=sids_path
    )
    
    # 加载倒排索引
    with open(index_path, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)
    
    # 步骤4: 基于物品的推荐示例
    print("\n=== 基于物品的推荐示例 ===")
    
    # 选择一些查询物品
    query_indices = [0, 100, 1000]
    
    for query_idx in query_indices:
        query_item = metadata[query_idx]
        print(f"\n查询物品: {query_item['title']} (类别: {query_item['category']})")
        
        # 使用简单的SID推荐
        print("\n1. 基于SID的简单推荐:")
        simple_indices, simple_scores = simple_recommendation(query_idx, sids, k=5)
        display_recommendations(simple_indices, simple_scores, metadata)
        
        # 使用倒排索引推荐（更高效）
        print("\n2. 使用倒排索引的SID推荐:")
        inverted_indices, inverted_scores = inverted_index_recommendation(
            query_idx, sids, inverted_index, k=5
        )
        display_recommendations(inverted_indices, inverted_scores, metadata)
        
        # 使用混合推荐
        print("\n3. 混合推荐 (SID + 原始嵌入):")
        hybrid_indices, hybrid_scores = hybrid_recommendation(
            query_idx, embeddings, sids, inverted_index, k=5, sid_weight=0.7
        )
        display_recommendations(hybrid_indices, hybrid_scores, metadata)
    
    # 步骤5: 基于用户的推荐示例
    print("\n=== 基于用户的推荐示例 ===")
    
    # 创建用户历史（例如，用户交互过的物品）
    user_history = [0, 10, 20, 30]  # 假设用户浏览了这些物品
    print("用户历史交互物品:")
    for item_idx in user_history:
        item = metadata[item_idx]
        print(f"  - {item['title']} (类别: {item['category']})")
    
    # 生成用户偏好配置文件
    user_profile = generate_user_profile(user_history, sids)
    
    # 基于用户偏好的推荐
    user_indices, user_scores = user_based_recommendation(user_profile, sids, k=10)
    print("\n基于用户偏好的推荐:")
    display_recommendations(user_indices, user_scores, metadata)
    
    # 步骤6: 推荐系统性能分析
    print("\n=== 推荐系统性能分析 ===")
    
    # 评估推荐速度
    print("\n评估不同推荐方法的速度:")
    
    # 简单SID推荐
    start_time = time.time()
    for _ in range(10):  # 测试10次
        simple_recommendation(0, sids, k=10)
    simple_time = (time.time() - start_time) / 10
    print(f"简单SID推荐平均耗时: {simple_time*1000:.2f} 毫秒/查询")
    
    # 倒排索引推荐
    start_time = time.time()
    for _ in range(10):
        inverted_index_recommendation(0, sids, inverted_index, k=10)
    inverted_time = (time.time() - start_time) / 10
    print(f"倒排索引推荐平均耗时: {inverted_time*1000:.2f} 毫秒/查询")
    print(f"倒排索引提速: {simple_time/inverted_time:.2f}x")
    
    # 混合推荐
    start_time = time.time()
    for _ in range(10):
        hybrid_recommendation(0, embeddings, sids, inverted_index, k=10)
    hybrid_time = (time.time() - start_time) / 10
    print(f"混合推荐平均耗时: {hybrid_time*1000:.2f} 毫秒/查询")
    
    # 步骤7: 推荐结果多样性分析
    print("\n推荐结果多样性分析:")
    
    for query_idx in query_indices[:2]:  # 只分析前两个查询
        query_item = metadata[query_idx]
        print(f"\n查询: {query_item['title']} ({query_item['category']})")
        
        # 获取混合推荐结果
        hybrid_indices, _ = hybrid_recommendation(
            query_idx, embeddings, sids, inverted_index, k=20, sid_weight=0.7
        )
        
        # 分析类别分布
        categories = [metadata[idx]['category'] for idx in hybrid_indices]
        category_counts = Counter(categories)
        total = len(categories)
        
        print("推荐结果类别分布:")
        for category, count in category_counts.items():
            percentage = count / total * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # 计算类别多样性分数（熵）
        import math
        entropy = 0.0
        for count in category_counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        
        max_entropy = math.log2(len(set(metadata[idx]['category'] for idx in range(len(metadata)))))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"类别多样性分数 (归一化熵): {normalized_entropy:.4f}")
    
    print("\n推荐系统示例运行完成！")
    print("\n在实际应用中使用时的建议:")
    print("1. 使用真实训练好的RQVAE模型替换示例中的模拟模型")
    print("2. 使用您的真实多模态嵌入数据替换示例数据")
    print("3. 根据您的业务需求调整推荐算法参数")
    print("4. 考虑添加更多业务规则和过滤条件")
    print("5. 实现A/B测试框架来评估不同推荐策略的效果")
    print("6. 对于大规模系统，考虑使用更高效的索引结构如Faiss")


if __name__ == '__main__':
    from collections import Counter
    main()