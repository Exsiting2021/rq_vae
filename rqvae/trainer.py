# RQ-VAE训练器模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import numpy as np
from .utils.logger import setup_logger
from .utils.metrics import compute_reconstruction_loss

class Trainer:
    """
    RQ-VAE训练器类，负责模型的训练、验证和保存
    
    Args:
        model (nn.Module): RQVAE模型
        config (dict): 配置字典
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
    """
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设置设备
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 设置损失函数
        self.reconstruction_loss_fn = nn.MSELoss()
        
        # 设置早停参数
        self.patience = config['training']['patience']
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # 创建检查点目录
        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = setup_logger(config['training']['log_dir'], 'rqvae_trainer')
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"模型参数数量: {self._count_parameters()}")
    
    def _create_optimizer(self):
        """
        创建优化器
        
        Returns:
            torch.optim.Optimizer: 优化器实例
        """
        learning_rate = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        betas = self.config['training']['betas']
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """
        创建学习率调度器
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: 调度器实例
        """
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return scheduler
    
    def _count_parameters(self):
        """
        计算模型参数数量
        
        Returns:
            int: 参数数量
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        num_batches = 0
        
        # 创建进度条
        progress_bar = tqdm(self.train_loader, desc="训练", leave=False)
        
        for batch in progress_bar:
            # 将数据移到设备上
            batch = batch.to(self.device)
            
            # 前向传播
            x_recon, quantization_loss, z, z_quantized, _ = self.model(batch)
            
            # 计算重建损失
            recon_loss = self.reconstruction_loss_fn(x_recon, batch)
            
            # 总损失 = 重建损失 + 量化损失
            loss = recon_loss + quantization_loss
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新统计信息，处理loss可能已经是浮点数的情况
            total_loss += loss.item() if hasattr(loss, 'item') else loss
            total_recon_loss += recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss
            total_quant_loss += quantization_loss.item() if hasattr(quantization_loss, 'item') else quantization_loss
            num_batches += 1
            
            # 更新进度条，处理可能是浮点数的情况
            progress_bar.set_postfix({
                'loss': loss.item() if hasattr(loss, 'item') else loss,
                'recon_loss': recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss,
                'quant_loss': quantization_loss.item() if hasattr(quantization_loss, 'item') else quantization_loss
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_quant_loss = total_quant_loss / num_batches
        
        self.logger.info(
            f"训练 - 总损失: {avg_loss:.6f}, "
            f"重建损失: {avg_recon_loss:.6f}, "
            f"量化损失: {avg_quant_loss:.6f}"
        )
        
        return avg_loss
    
    def validate_epoch(self):
        """
        验证一个epoch
        
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            # 创建进度条
            progress_bar = tqdm(self.val_loader, desc="验证", leave=False)
            
            for batch in progress_bar:
                # 将数据移到设备上
                batch = batch.to(self.device)
                
                # 前向传播
                x_recon, quantization_loss, z, z_quantized, _ = self.model(batch)
                
                # 计算重建损失
                recon_loss = self.reconstruction_loss_fn(x_recon, batch)
                
                # 总损失 = 重建损失 + 量化损失
                loss = recon_loss + quantization_loss
                
                # 更新统计信息，处理loss可能已经是浮点数的情况
                total_loss += loss.item() if hasattr(loss, 'item') else loss
                total_recon_loss += recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss
                total_quant_loss += quantization_loss.item() if hasattr(quantization_loss, 'item') else quantization_loss
                num_batches += 1
                
                # 更新进度条，处理可能是浮点数的情况
                progress_bar.set_postfix({
                    'loss': loss.item() if hasattr(loss, 'item') else loss,
                    'recon_loss': recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss,
                    'quant_loss': quantization_loss.item() if hasattr(quantization_loss, 'item') else quantization_loss
                })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_quant_loss = total_quant_loss / num_batches
        
        self.logger.info(
            f"验证 - 总损失: {avg_loss:.6f}, "
            f"重建损失: {avg_recon_loss:.6f}, "
            f"量化损失: {avg_quant_loss:.6f}"
        )
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存模型检查点
        
        Args:
            epoch (int): 当前epoch
            is_best (bool): 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存常规检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存到: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"最佳模型已保存到: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载模型检查点
        
        Args:
            checkpoint_path (str): 检查点路径
        
        Returns:
            int: 加载的epoch数
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载最佳验证损失
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        epoch = checkpoint['epoch']
        self.logger.info(f"已加载检查点，epoch: {epoch}")
        
        return epoch
    
    def train(self, start_epoch=0):
        """
        训练模型
        
        Args:
            start_epoch (int): 起始epoch
        """
        num_epochs = self.config['training']['num_epochs']
        save_interval = self.config['training']['save_interval']
        
        self.logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 验证一个epoch
            val_loss = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存检查点
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_checkpoint(epoch + 1, is_best=True)
            else:
                self.early_stop_counter += 1
                self.logger.info(f"早停计数器: {self.early_stop_counter}/{self.patience}")
                
                if self.early_stop_counter >= self.patience:
                    self.logger.info("触发早停，停止训练")
                    break
        
        self.logger.info("训练完成！")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")

# 训练工具函数
def train_from_scratch(model, config, train_loader, val_loader):
    """
    从头开始训练模型
    
    Args:
        model (nn.Module): 模型实例
        config (dict): 配置字典
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
    """
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train(start_epoch=0)
    return trainer

def resume_training(model, config, train_loader, val_loader, checkpoint_path):
    """
    从检查点恢复训练
    
    Args:
        model (nn.Module): 模型实例
        config (dict): 配置字典
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        checkpoint_path (str): 检查点路径
    """
    trainer = Trainer(model, config, train_loader, val_loader)
    start_epoch = trainer.load_checkpoint(checkpoint_path)
    trainer.train(start_epoch=start_epoch)
    return trainer

def get_latest_checkpoint(checkpoint_dir):
    """
    获取最新的检查点文件
    
    Args:
        checkpoint_dir (str): 检查点目录
    
    Returns:
        str: 最新检查点的路径
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 获取所有检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if not checkpoint_files:
        return None
    
    # 按epoch号排序
    def get_epoch(file_name):
        return int(file_name.split('_')[2].split('.')[0])
    
    checkpoint_files.sort(key=get_epoch, reverse=True)
    
    return os.path.join(checkpoint_dir, checkpoint_files[0])