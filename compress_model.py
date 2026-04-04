import joblib
import os
import time

def compress_model(input_path="model/model.pkl", output_path="model/model_compressed.pkl"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return False
    
    print(f"Loading model from {input_path}...")
    start_time = time.time()
    try:
        model = joblib.load(input_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    print(f"Compressing and saving to {output_path} (level 3)...")
    try:
        # compress=3 is usually a good balance for size and speed
        joblib.dump(model, output_path, compress=3)
    except Exception as e:
        print(f"Error saving compressed model: {e}")
        return False
    
    end_time = time.time()
    
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print("-" * 30)
    print(f"Original Size: {original_size:.2f} MB")
    print(f"Compressed Size: {compressed_size:.2f} MB")
    print(f"Reduction: {(1 - compressed_size/original_size) * 100:.1f}%")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print("-" * 30)
    
    if compressed_size < 100:
        print("✅ SUCCESS! The file is now under the 100MB limit.")
        # Replace the original with the compressed one
        os.remove(input_path)
        os.rename(output_path, input_path)
        print(f"Refreshed {input_path} with compressed version.")
        return True
    else:
        print("❌ WARNING: The file is still over 100MB. Try increasing compress level.")
        return False

if __name__ == "__main__":
    compress_model()
