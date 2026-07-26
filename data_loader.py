from datasets import load_dataset

ds = load_dataset("gretelai/synthetic_text_to_sql")

train = ds['train']
test_val = ds['test']

split_data = test_val.train_test_split(test_size=0.5, seed=42)

validation = split_data['train']
test = split_data['test']

train.to_parquet('datasets/train.parquet')
validation.to_parquet('datasets/validation.parquet')
test.to_parquet('datasets/test.parquet')