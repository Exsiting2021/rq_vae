#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ-VAE SID工具
用于从嵌入向量提取SID或从SID重建嵌入向量
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from rqvae.utils.sid_extractor import SIDExtractor
from rqvae.utils.config import load_config
from rqvae.utils.logger import setup_logger


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='RQ-VAE SID工具')
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='命令: extract(提取SID), reconstruct(重建嵌入), similarity(计算SID相似度)')
    
    # extract命令 - 从嵌入向量提取SID
    extract_parser = subparsers.add_parser('extract', help='从嵌入向量提取SID')
    extract_parser.add_argument('--model-path', type=str, required=True,
                              help='训练好的模型路径')
    extract_parser.add_argument('--input', type=str, required=True,
                              help='输入嵌入向量文件(.npy或.npz)或目录')
    extract_parser.add_argument('--output', type=str, required=True,
                              help='输出SID文件(.npy)')
    extract_parser.add_argument('--config', type=str, default=None,
                              help='配置文件路径')
    extract_parser.add_argument('--batch-size', type=int, default=1024,
                              help='处理批次大小')
    extract_parser.add_argument('--save-json', action='store_true',
                              help='同时保存为JSON格式')
    
    # reconstruct命令 - 从SID重建嵌入向量
    reconstruct_parser = subparsers.add_parser('reconstruct', help='从SID重建嵌入向量')
    reconstruct_parser.add_argument('--model-path', type=str, required=True,
                                  help='训练好的模型路径')
    reconstruct_parser.add_argument('--input', type=str, required=True,
                                  help='输入SID文件(.npy或.json)')
    reconstruct_parser.add_argument('--output', type=str, required=True,
                                  help='输出嵌入向量文件(.npy或.npz)')
    reconstruct_parser.add_argument('--config', type=str, default=None,
                                  help='配置文件路径')
    reconstruct_parser.add_argument('--batch-size', type=int, default=1024,
                                  help='处理批次大小')
    
    # similarity命令 - 计算SID相似度
    similarity_parser = subparsers.add_parser('similarity', help='计算SID相似度')
    similarity_parser.add_argument('--model-path', type=str, required=True,
                                 help='训练好的模型路径')
    similarity_parser.add_argument('--query', type=str, required=True,
                                 help='查询SID文件(.npy或.json)或嵌入向量文件(.npy或.npz)')
    similarity_parser.add_argument('--database', type=str, required=True,
                                 help='数据库SID文件(.npy或.json)或嵌入向量文件(.npy或.npz)')
    similarity_parser.add_argument('--output', type=str, required=True,
                                 help='输出相似度结果文件(.json)')
    similarity_parser.add_argument('--config', type=str, default=None,
                                 help='配置文件路径')
    similarity_parser.add_argument('--k', type=int, default=10,
                                 help='返回前k个最相似的结果')
    similarity_parser.add_argument('--batch-size', type=int, default=1024,
                                 help='处理批次大小')
    
    return parser.parse_args()


def load_embeddings(file_path):
    """
    加载嵌入向量
    
    Args:
        file_path: 嵌入向量文件路径(.npy或.npz)
    
    Returns:
        np.ndarray: 嵌入向量数组
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        with np.load(file_path) as data:
            # 尝试不同的键
            for key in ['embeddings', 'embedding', 'vectors', 'vector', 'arr_0', 'data']:
                if key in data:
                    return data[key]
            # 如果找不到匹配的键，返回第一个数组
            return list(data.values())[0]
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def load_sids(file_path):
    """
    加载SID
    
    Args:
        file_path: SID文件路径(.npy或.json)
    
    Returns:
        np.ndarray: SID数组
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 支持不同格式的JSON
            if isinstance(data, list):
                # 直接是SID列表
                return np.array(data)
            elif isinstance(data, dict):
                # 尝试不同的键
                for key in ['sids', 'codes', 'indices', 'data']:
                    if key in data:
                        return np.array(data[key])
                raise ValueError(f"JSON文件中找不到SID数据")
            else:
                raise ValueError(f"JSON文件格式不支持")
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def load_input_data(file_path, is_sid=False):
    """
    加载输入数据
    
    Args:
        file_path: 文件路径或目录路径
        is_sid: 是否为SID数据
    
    Returns:
        np.ndarray: 数据数组
    """
    # 检查是否是目录
    if os.path.isdir(file_path):
        # 遍历目录下所有支持的文件
        all_data = []
        supported_extensions = ['.npy', '.npz'] if not is_sid else ['.npy', '.json']
        
        for root, _, files in os.walk(file_path):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path_full = os.path.join(root, file)
                    try:
                        if is_sid:
                            data = load_sids(file_path_full)
                        else:
                            data = load_embeddings(file_path_full)
                        all_data.append(data)
                    except Exception as e:
                        print(f"警告: 无法加载文件 {file_path_full}: {e}")
        
        if not all_data:
            raise ValueError(f"目录 {file_path} 中没有找到支持的文件")
        
        # 合并所有数据
        return np.concatenate(all_data, axis=0)
    else:
        # 单个文件
        if is_sid:
            return load_sids(file_path)
        else:
            return load_embeddings(file_path)


def ensure_dir(file_path):
    """
    确保输出目录存在
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def extract_sids(args):
    """
    从嵌入向量提取SID
    """
    print(f"从嵌入向量提取SID...")
    print(f"模型路径: {args.model_path}")
    print(f"输入路径: {args.input}")
    print(f"输出路径: {args.output}")
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 加载数据
    print(f"加载嵌入向量数据...")
    embeddings = load_input_data(args.input, is_sid=False)
    print(f"加载了 {len(embeddings)} 个嵌入向量")
    
    # 创建SID提取器
    print(f"初始化SID提取器...")
    extractor = SIDExtractor(args.model_path, args.config)
    
    # 提取SID
    print(f"提取SID...")
    sids = extractor.extract_sids(embeddings, batch_size=args.batch_size)
    
    # 保存结果
    print(f"保存SID到 {args.output}...")
    np.save(args.output, sids)
    
    # 保存为JSON格式（如果需要）
    if args.save_json:
        json_path = args.output.replace('.npy', '.json')
        # 将numpy数组转换为Python列表以便JSON序列化
        sids_list = sids.tolist()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'sids': sids_list}, f, ensure_ascii=False, indent=2)
        print(f"保存JSON格式到 {json_path}")
    
    print(f"SID提取完成，共提取了 {len(sids)} 个SID")


def reconstruct_embeddings(args):
    """
    从SID重建嵌入向量
    """
    print(f"从SID重建嵌入向量...")
    print(f"模型路径: {args.model_path}")
    print(f"输入路径: {args.input}")
    print(f"输出路径: {args.output}")
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 加载数据
    print(f"加载SID数据...")
    sids = load_input_data(args.input, is_sid=True)
    print(f"加载了 {len(sids)} 个SID")
    
    # 创建SID提取器
    print(f"初始化SID提取器...")
    extractor = SIDExtractor(args.model_path, args.config)
    
    # 重建嵌入向量
    print(f"重建嵌入向量...")
    embeddings = extractor.reconstruct_from_sids(sids, batch_size=args.batch_size)
    
    # 保存结果
    print(f"保存嵌入向量到 {args.output}...")
    if args.output.endswith('.npz'):
        np.savez_compressed(args.output, embeddings=embeddings)
    else:
        np.save(args.output, embeddings)
    
    print(f"嵌入向量重建完成，共重建了 {len(embeddings)} 个嵌入向量")


def compute_similarity(args):
    """
    计算SID相似度
    """
    print(f"计算SID相似度...")
    print(f"模型路径: {args.model_path}")
    print(f"查询数据路径: {args.query}")
    print(f"数据库路径: {args.database}")
    print(f"输出路径: {args.output}")
    print(f"返回前 {args.k} 个结果")
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 创建SID提取器
    print(f"初始化SID提取器...")
    extractor = SIDExtractor(args.model_path, args.config)
    
    # 加载查询数据
    print(f"加载查询数据...")
    # 自动检测文件类型并加载
    if args.query.endswith(('.json', '.npy')) and (
        args.query.endswith('.json') or 
        (args.query.endswith('.npy') and 
         (os.path.basename(args.query).lower().find('sid') >= 0 or 
          os.path.basename(args.query).lower().find('code') >= 0))):
        query_sids = load_input_data(args.query, is_sid=True)
        print(f"加载了 {len(query_sids)} 个查询SID")
    else:
        # 假设是嵌入向量
        query_embeddings = load_input_data(args.query, is_sid=False)
        print(f"加载了 {len(query_embeddings)} 个查询嵌入向量")
        # 提取SID
        print(f"从查询嵌入向量提取SID...")
        query_sids = extractor.extract_sids(query_embeddings, batch_size=args.batch_size)
    
    # 加载数据库数据
    print(f"加载数据库数据...")
    # 自动检测文件类型并加载
    if args.database.endswith(('.json', '.npy')) and (
        args.database.endswith('.json') or 
        (args.database.endswith('.npy') and 
         (os.path.basename(args.database).lower().find('sid') >= 0 or 
          os.path.basename(args.database).lower().find('code') >= 0))):
        database_sids = load_input_data(args.database, is_sid=True)
        print(f"加载了 {len(database_sids)} 个数据库SID")
    else:
        # 假设是嵌入向量
        database_embeddings = load_input_data(args.database, is_sid=False)
        print(f"加载了 {len(database_embeddings)} 个数据库嵌入向量")
        # 提取SID
        print(f"从数据库嵌入向量提取SID...")
        database_sids = extractor.extract_sids(database_embeddings, batch_size=args.batch_size)
    
    # 计算相似度
    print(f"计算相似度...")
    results = []
    
    for i, query_sid in enumerate(tqdm(query_sids)):
        # 计算与数据库中所有SID的相似度
        similarities = extractor.compute_sid_similarity(query_sid, database_sids)
        
        # 获取前k个最相似的结果
        top_k_indices = np.argsort(similarities)[::-1][:args.k]
        top_k_similarities = similarities[top_k_indices]
        top_k_sids = database_sids[top_k_indices]
        
        # 构建结果
        result = {
            'query_index': i,
            'query_sid': query_sid.tolist(),
            'top_k': [
                {
                    'index': int(idx),
                    'sid': sid.tolist(),
                    'similarity': float(sim)
                }
                for idx, sid, sim in zip(top_k_indices, top_k_sids, top_k_similarities)
            ]
        }
        results.append(result)
    
    # 保存结果
    print(f"保存相似度结果到 {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"相似度计算完成，共处理了 {len(query_sids)} 个查询")


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 检查命令
    if args.command == 'extract':
        extract_sids(args)
    elif args.command == 'reconstruct':
        reconstruct_embeddings(args)
    elif args.command == 'similarity':
        compute_similarity(args)
    else:
        print("请指定命令: extract, reconstruct 或 similarity")
        sys.exit(1)


if __name__ == '__main__':
    main()