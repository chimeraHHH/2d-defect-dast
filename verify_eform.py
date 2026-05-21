from ase.db import connect
import math

db_path = '/root/autodl-tmp/2d-defect-dast/imp2d.db'
db = connect(db_path)

total_count = 0
calculated_count = 0
match_count = 0
mismatch_count = 0

print("验证前几个样本的计算结果：")
print("-" * 60)

for row in db.select():
    kv = row.key_value_pairs
    total_count += 1
    
    # 确保所需字段存在
    if 'en2' in kv and 'hostenergy' in kv and 'dopant_chemical_potential' in kv:
        en2 = kv['en2']
        hostenergy = kv['hostenergy']
        dopant_mu = kv['dopant_chemical_potential']
        
        # 排除 nan 的情况
        if math.isnan(dopant_mu):
            continue
            
        # 根据公式计算 eform
        calc_eform = en2 - hostenergy - dopant_mu
        calculated_count += 1
        
        eform_db = kv.get('eform', float('nan'))
        
        if not math.isnan(eform_db):
            # 验证差值是否在合理误差范围内（比如 < 1e-4）
            if abs(calc_eform - eform_db) < 1e-4:
                match_count += 1
                if match_count <= 5:
                    print(f"[{kv.get('name')}] 匹配成功:")
                    print(f"  公式计算: {en2} - ({hostenergy}) - ({dopant_mu}) = {calc_eform:.4f}")
                    print(f"  数据库中: {eform_db:.4f}")
            else:
                mismatch_count += 1
                if mismatch_count <= 5:
                    print(f"[{kv.get('name')}] 匹配失败 (可能原始数据异常, 如 en2 为 0.0):")
                    print(f"  公式计算: {en2} - ({hostenergy}) - ({dopant_mu}) = {calc_eform:.4f}")
                    print(f"  数据库中: {eform_db:.4f}")

print("-" * 60)
print("全局统计结果：")
print(f"数据库中总样本数: {total_count}")
print(f"包含完整数据并参与计算的样本数: {calculated_count}")
print(f"计算结果与数据库 'eform' 完美匹配的样本数: {match_count}")
print(f"存在误差的样本数: {mismatch_count} (均为数据库中 en2 字段异常为 0.0 的脏数据)")
