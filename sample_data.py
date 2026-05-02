import pickle
import random
import os
import sys

def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    input_file_1 = '/Users/wuleyan/Desktop/2D_Platform_有前端和后端的/final_dataset.pkl'
    input_file_2 = '/Users/wuleyan/Desktop/beifen/WLY/final_dataset.pkl'
    
    output_file = f'/Users/wuleyan/Desktop/2D_Platform_有前端和后端的/sample_{n_samples}_dataset.pkl'
    
    input_file = None
    if os.path.exists(input_file_1):
        input_file = input_file_1
    elif os.path.exists(input_file_2):
        input_file = input_file_2
    else:
        # Fallback empty logic if neither exists just to test
        return

    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return

    sampled_data = None
    if isinstance(data, list):
        n_actual = min(n_samples, len(data))
        sampled_data = random.sample(data, n_actual)
    elif hasattr(data, 'sample'):
        n_actual = min(n_samples, len(data))
        sampled_data = data.sample(n=n_actual)
    elif isinstance(data, dict):
        n_actual = min(n_samples, len(data))
        keys = random.sample(list(data.keys()), n_actual)
        sampled_data = {k: data[k] for k in keys}

    if sampled_data is not None:
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(sampled_data, f)
            with open(f'/Users/wuleyan/Desktop/2D_Platform_有前端和后端的/success_{n_samples}.txt', 'w') as f:
                f.write('success')
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
