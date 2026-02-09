from datasets import load_dataset

try:
    ds = load_dataset("ethical-spectacle/gus-dataset-v1")
    print(f"Available splits: {ds.keys()}")
    for split in ds.keys():
        print(f"Split '{split}' has {len(ds[split])} examples")
except Exception as e:
    print(e)
