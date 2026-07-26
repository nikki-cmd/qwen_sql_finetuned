TRAIN_PATH = "datasets/train.parquet"
VALIDATION_SET = "datasets/validation_mini.parquet"

TXT_LOG_FILE = "validation_responses.txt"
TIMING_LOG_FILE = "validation_timing.log"
PARQUET_OUTPUT = "validation_results.parquet"
MAX_NEW_TOKENS = 100
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

system_instruction = """You are a strict SQL generator. Convert the following natural language request into a valid SQL query.
Here is story of table creation:\n"""

extra_system_instruction = """ You must output ONLY the raw SQL code block without other text or explaining. 
Do not include any conversational text, explanations, notes, markdown formatting (do not use ```sql), or comments. 
If you violate these rules, the system will fail.
"""