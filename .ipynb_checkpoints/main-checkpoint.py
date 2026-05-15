from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from predictor import FormationEnergyPredictor
import os
from ase.io import read
from ase import Atoms
import pickle
import shutil
import json
import asyncio

app = FastAPI()

# 允许跨域请求，以便前端可以访问后端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（在生产环境中应该限制）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 在程序启动时就加载模型（避免重复加载）
# 使用基于当前文件的绝对路径，适配当前项目结构
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "formation_energy_model.pth")
feature_path = os.path.join(base_dir, "atom_features.pth")

predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path)

# 提供静态前端文件服务
frontend_dir = os.path.join(base_dir, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    if os.path.exists(os.path.join(frontend_dir, "index.html")):
        return FileResponse(os.path.join(frontend_dir, "index.html"))
    return {"message": "二维材料缺陷预测平台后端已启动"}

@app.post("/predict")
async def predict_formation_energy(file: UploadFile = File(...)):
    # 2. 接收并保存上传的文件
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 3. 使用 ase 读取晶体结构文件，并调用写好的预测器
        structure = read(temp_path)
        result = predictor.predict(structure) 
        
        # 4. 返回结果
        return {
            "filename": file.filename,
            "formation_energy": float(result),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # 5. 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict_batch_stream")
async def predict_batch_stream(file_path: str = Form(None), file: UploadFile = File(None)):
    async def generate():
        temp_path = None
        try:
            if file_path and os.path.exists(file_path):
                target_path = file_path
            elif file:
                temp_path = f"temp_batch_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                target_path = temp_path
            else:
                yield json.dumps({"status": "error", "message": "未提供文件或文件路径"}) + "\n"
                return

            with open(target_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                yield json.dumps({"status": "error", "message": "不支持的数据格式，需要List[dict]"}) + "\n"
                return
                
            total = len(data)
            yield json.dumps({"status": "start", "total": total}) + "\n"
            
            for i, item in enumerate(data):
                try:
                    if all(k in item for k in ['numbers', 'positions', 'cell', 'pbc']):
                        atoms = Atoms(numbers=item['numbers'], positions=item['positions'], cell=item['cell'], pbc=item['pbc'])
                    else:
                        raise ValueError("缺少必要的结构信息 (numbers, positions, cell, pbc)")
                        
                    res = predictor.predict(atoms)
                    
                    # yield progress
                    yield json.dumps({
                        "status": "progress", 
                        "current": i + 1, 
                        "total": total, 
                        "result": {
                            "id": str(item.get('id', item.get('unique_id', i))),
                            "formula": atoms.get_chemical_formula(),
                            "formation_energy": float(res)
                        }
                    }) + "\n"
                except Exception as e:
                    yield json.dumps({
                        "status": "progress_error", 
                        "current": i + 1, 
                        "message": str(e)
                    }) + "\n"
                
                # yield control to event loop so it streams
                await asyncio.sleep(0.001)
                
            yield json.dumps({"status": "done"}) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": f"处理批次数据时出错: {str(e)}"}) + "\n"
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
