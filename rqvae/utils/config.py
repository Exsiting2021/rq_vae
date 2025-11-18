# 配置文件加载和保存工具
import yaml
import os

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path (str): 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config, config_path):
    """
    保存配置到YAML文件
    
    Args:
        config (dict): 配置字典
        config_path (str): 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"配置已保存到: {config_path}")