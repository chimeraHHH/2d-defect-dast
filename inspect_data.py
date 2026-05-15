import pickle

with open('data/processed/cleaned_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

with open('inspect_data.txt', 'w') as f:
    f.write(f"Total items: {len(data)}\n")
    f.write(f"Keys: {data[0].keys()}\n")
    for i in range(min(5, len(data))):
        f.write(f"Formula {i}: {data[i].get('formula', '')}\n")
        f.write(f"Defect type {i}: {data[i].get('defect_type', '')}\n")
