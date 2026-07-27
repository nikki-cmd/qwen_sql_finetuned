TRAIN_PATH = "datasets/train.parquet"
VALIDATION_SET = "datasets/validation_mini.parquet"

TXT_LOG_FILE = "results/validation_responses.txt"
TIMING_LOG_FILE = "results/validation_timing.log"
PARQUET_OUTPUT = "results/validation_results.parquet"
MAX_NEW_TOKENS = 100
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
ADAPTER_PATH = "models/checkpoint-1200" 

system_instruction = """You are a strict SQL generator. Convert the following natural language request into a valid SQL query.
Here is story of table creation:\n"""

extra_system_instruction = """ You must output ONLY the raw SQL code block without other text or explaining. 
Do not include any conversational text, explanations, notes, markdown formatting (do not use ```sql), or comments. 
If you violate these rules, the system will fail.
"""