from ase.db import connect
import math
import os

input_db = '/root/autodl-tmp/2d-defect-dast/imp2d.db'
output_db = '/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db'

# Remove existing output file if it exists
if os.path.exists(output_db):
    os.remove(output_db)

db_in = connect(input_db)
db_out = connect(output_db)

MAX_EFORM = 20.0

total = 0
kept = 0
removed_no_en2 = 0
removed_unconverged = 0
removed_large_eform = 0
removed_nan_eform = 0

print(f"开始处理数据集，将形成能 > {MAX_EFORM} eV 的数据视为“过大”并剔除...")

for row in db_in.select():
    total += 1
    kv = row.key_value_pairs
    remove = False
    
    # 1. en2 不存在或为 0.0
    if 'en2' not in kv or math.isnan(kv.get('en2', float('nan'))) or kv.get('en2') == 0.0:
        removed_no_en2 += 1
        remove = True
        
    # 2. 计算不收敛
    elif kv.get('converged') == False:
        removed_unconverged += 1
        remove = True
        
    # 3. 形成能过大或不存在 (nan)
    else:
        eform = kv.get('eform', float('nan'))
        if math.isnan(eform):
            removed_nan_eform += 1
            remove = True
        elif eform > MAX_EFORM:
            removed_large_eform += 1
            remove = True
            
    if not remove:
        # 写入新数据库
        db_out.write(row.toatoms(), **kv)
        kept += 1

print("-" * 50)
print(f"处理完成！")
print(f"总样本数: {total}")
print(f"保留样本数: {kept}")
print(f"总剔除数: {total - kept}")
print("剔除原因统计:")
print(f"  - en2不存在或异常(0.0): {removed_no_en2}")
print(f"  - 计算未收敛(converged=False): {removed_unconverged}")
print(f"  - 形成能缺失(NaN): {removed_nan_eform}")
print(f"  - 形成能过大(> {MAX_EFORM} eV): {removed_large_eform}")
print(f"新数据集已保存至: {output_db}")
