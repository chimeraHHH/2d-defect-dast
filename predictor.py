import os
import sys
import torch

# 确保项目根目录在系统路径中，以便能够正确导入相对路径下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.arch import CrystalTransformer
from utils.preprocess import CrystalPreprocessor

class FormationEnergyPredictor:
    """
    平台级预测封装类。
    在初始化时只加载一次模型架构和权重，并提供方法进行多次重复预测。
    """
    def __init__(self, 
                 model_path='models/formation_energy_model.pth', 
                 feature_path='atom_features.pth',
                 device=None):
        """
        初始化预测器。
        
        Args:
            model_path (str): 模型权重文件路径。
            feature_path (str): 原子特征张量文件路径。
            device (str, optional): 运行设备 ('cpu' 或 'cuda')。如果为 None 则自动检测。
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 1. 初始化数据预处理器
        self.preprocessor = CrystalPreprocessor(feature_path=feature_path, device=self.device)
        
        # 2. 加载模型 Checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件未找到: {model_path}")
            
        ckpt = torch.load(model_path, map_location=self.device)
        config = ckpt.get('config', {})
        
        # 从配置中提取模型架构参数
        hidden_dim = config.get('hidden_dim', 64)
        n_local = config.get('n_local', 2)
        n_global = config.get('n_global', 1)
        
        # 3. 初始化模型架构
        self.model = CrystalTransformer(
            atom_fea_len=9, 
            hidden_dim=hidden_dim, 
            n_local_layers=n_local, 
            n_global_layers=n_global
        ).to(self.device)
        
        # 4. 加载模型权重
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # 处理可能存在的 DataParallel 'module.' 前缀
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval() # 切换到评估模式
        
        # 5. 加载 Normalizer (用于反归一化)
        self.normalizer = ckpt.get('normalizer', None)
        if self.normalizer is not None:
            self.mean = self.normalizer['mean'].to(self.device)
            self.std = self.normalizer['std'].to(self.device)
        else:
            self.mean = torch.tensor(0.0, device=self.device)
            self.std = torch.tensor(1.0, device=self.device)
        
        print(f"Predictor initialized successfully on {self.device}.")

    def predict(self, structure):
        """
        对单一晶体结构进行形成能预测。
        
        Args:
            structure: ase.Atoms 或 pymatgen Structure 对象。
            
        Returns:
            float: 预测的形成能数值。
        """
        # 预处理数据 (转为图和序列)
        batch_dict = self.preprocessor.process(structure)
        
        # 推理预测
        with torch.no_grad():
            output = self.model(batch_dict)
            
            # 预测结果反归一化，与训练端保持一致
            if self.normalizer is not None:
                output = output * self.std + self.mean
            
        return output.item()
        
    def predict_many(self, structures):
        """
        对多个晶体结构进行批量预测（串行处理）。
        
        Args:
            structures: 包含多个 ase.Atoms 或 pymatgen Structure 对象的列表。
            
        Returns:
            list[float]: 预测结果列表。
        """
        return [self.predict(struct) for struct in structures]

if __name__ == "__main__":
    # ===== 快速测试平台预测类 =====
    from ase.build import bulk
    
    # 获取绝对路径，以适配直接运行此脚本
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_p = os.path.join(base_dir, 'models', 'formation_energy_model.pth')
    feat_p = os.path.join(base_dir, 'atom_features.pth')
    
    # 实例化预测器（仅加载一次）
    print("正在加载预测引擎...")
    predictor = FormationEnergyPredictor(model_path=model_p, feature_path=feat_p)
    
    # 构建多个待测结构进行多次预测
    print("\n构建测试晶体结构...")
    si_fcc = bulk('Si', 'fcc', a=5.43)
    si_bcc = bulk('Si', 'bcc', a=3.5)
    cu_fcc = bulk('Cu', 'fcc', a=3.6)
    
    # 执行多次预测
    print("\n执行连续预测...")
    print(f"Si (FCC) 形成能: {predictor.predict(si_fcc):.4f}")
    print(f"Si (BCC) 形成能: {predictor.predict(si_bcc):.4f}")
    print(f"Cu (FCC) 形成能: {predictor.predict(cu_fcc):.4f}")
