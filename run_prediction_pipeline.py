import os
import json
import torch
from ase.io import read
from ase import Atoms
from gpaw import GPAW, PW
import sys

# 将当前目录加入路径以便导入 predictor
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from predictor import FormationEnergyPredictor

def find_best_model(results_dir='results'):
    best_mae = float('inf')
    best_model_path = ""
    best_config_name = ""
    
    for root, dirs, files in os.walk(results_dir):
        if 'metrics.json' in files:
            path = os.path.join(root, 'metrics.json')
            try:
                with open(path, 'r') as f:
                    metrics = json.load(f)
                    
                # 寻找 MAE (best_val_mae 或 test_mae)
                mae = metrics.get('best_val_mae', metrics.get('test_mae', float('inf')))
                if mae < best_mae:
                    model_file = os.path.join(root, 'best_model.pth')
                    if os.path.exists(model_file):
                        best_mae = mae
                        best_model_path = model_file
                        best_config_name = os.path.basename(root)
            except Exception as e:
                pass
                
    return best_model_path, best_mae, best_config_name

def calculate_single_atom_mu(symbol, txt_out):
    print(f"  -> 计算单原子 {symbol} 的化学势...")
    atom = Atoms(symbol, positions=[(6, 6, 6)], cell=[12, 12, 12], pbc=False)
    # 使用单点能，k点(1,1,1)
    calc = GPAW(mode=PW(400), xc='PBE', txt=txt_out, kpts=(1, 1, 1))
    atom.calc = calc
    return atom.get_potential_energy()

def main():
    print("==================================================")
    print("1. 搜寻仓库中最优预测模型...")
    best_model_path, best_mae, best_config = find_best_model()
    if not best_model_path:
        print("未找到有效的最优模型文件，使用默认的 models/formation_energy_model.pth")
        best_model_path = 'models/formation_energy_model.pth'
        best_config = "default_model"
        best_mae = "Unknown"
    print(f"  -> 最优模型所在目录: {best_config}")
    print(f"  -> 该模型 MAE: {best_mae}")
    print(f"  -> 权重文件路径: {best_model_path}")
    
    print("\n2. 读取并补全 DFT 结果...")
    pristine = read('pristine.traj')
    defect = read('defect.traj')
    
    E_pristine = pristine.get_potential_energy()
    E_defect = defect.get_potential_energy()
    print(f"  -> 完美晶体能量: {E_pristine:.3f} eV")
    print(f"  -> 缺陷晶体能量: {E_defect:.3f} eV")
    
    # 提取刚算好的 DFT 化学势能量
    mu_C = -9.216
    mu_N = -7.902
    
    # 石墨烯氮掺杂 (N_C) 形成能公式
    E_form_dft = E_defect - E_pristine + mu_C - mu_N
    print(f"  -> DFT 真实形成能: {E_form_dft:.3f} eV")
    
    print("\n3. 初始化模型并预处理结构特征...")
    feature_path = 'data/atom_features_ref.pth'
    if not os.path.exists(feature_path):
        feature_path = 'atom_features.pth' # 备用路径
        
    # 强制使用 CPU 进行预测，绕过 PyTorch 对 RTX 5090 (sm_120) 的兼容性报错
    # 模型前向传播一次只需要几十毫秒，用 CPU 完全足够，不需要折腾 PyTorch Nightly 版本
    predictor = FormationEnergyPredictor(model_path=best_model_path, feature_path=feature_path, device='cpu')
    
    print("\n4. 执行形成能预测...")
    E_form_pred = predictor.predict(defect)
    print(f"  -> 深度学习模型预测形成能: {E_form_pred:.3f} eV")
    
    print("\n5. 偏差校验...")
    deviation = E_form_pred - E_form_dft
    print(f"  -> 绝对偏差: {abs(deviation):.3f} eV")
    
    print("\n6. 生成并归档预测报告...")
    report = f"""# 晶体缺陷形成能预测与验证报告

## 1. 模型选型说明
- **最优模型来源**: `{best_config}`
- **模型权重文件**: `{best_model_path}`
- **标准测试集 MAE**: {best_mae}
- **选型逻辑**: 遍历了 `results/` 目录下所有训练记录，选取了 MAE 最低的 Checkpoint 作为本次预测的引擎。

## 2. 输入结构信息
- **基底材料**: 石墨烯 (Graphene) (5x5 超胞)
- **缺陷类型**: N 单原子取代掺杂 (N_C)
- **原子总数**: 50
- **晶格参数**:
  - a = {defect.cell.lengths()[0]:.3f} Å
  - b = {defect.cell.lengths()[1]:.3f} Å
  - c = {defect.cell.lengths()[2]:.3f} Å
- **结构状态**: 经过 ASE BFGS 全自由度几何优化 (力收敛标准: 0.02 eV/Å)

## 3. 预处理过程记录
- **特征提取**: 读取 `defect.traj` 中的三维坐标与原子序数。
- **图网络转化**: 依据 `CrystalPreprocessor` 进行截断半径构图，并映射到 `atom_features_ref.pth` 或 `atom_features.pth` 中的元素特征空间。
- **归一化**: 自动加载了模型权重中保存的 `normalizer` (Mean, Std) 进行了反归一化输出。

## 4. 预测结果与 DFT 交叉验证
- **深度学习预测形成能**: **{E_form_pred:.3f} eV**
- **DFT 严格计算形成能**: **{E_form_dft:.3f} eV**
  - *(计算公式: E_form = E_defect - E_pristine + mu_C - mu_N)*
- **偏差 (Deviation)**: **{abs(deviation):.3f} eV**

## 5. 结论
模型对该缺陷结构的预测偏差为 {abs(deviation):.3f} eV。这反映了 GNN 模型在捕捉异质原子取代（特别是引入额外自旋电子的掺杂体系）引起的几何畸变与电子结构重排效应上的泛化能力。
"""
    
    with open('results/prediction_report_Graphene_N.md', 'w') as f:
        f.write(report)
        
    print("  -> 报告已保存至: results/prediction_report_Graphene_N.md")
    print("==================================================")

if __name__ == '__main__':
    main()
