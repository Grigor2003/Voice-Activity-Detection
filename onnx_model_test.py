from other.data.datasets import MSDWildDataset

model_path = "buffer/silero_vad.onnx"
dataset_path = "datasets/MSDWild"

data = MSDWildDataset(dataset_path)

model = load_model(model_path)
results = process_dataset(dataset_path, model)
print(results)
print("Processing complete.")
