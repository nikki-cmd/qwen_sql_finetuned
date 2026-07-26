import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from config import VALIDATION_SET, system_instruction, extra_system_instruction, MODEL_NAME, TXT_LOG_FILE, TIMING_LOG_FILE, MAX_NEW_TOKENS, PARQUET_OUTPUT
from transformers import AutoTokenizer, AutoModelForCausalLM

validation_set = pd.read_parquet(VALIDATION_SET)
print(f"Loaded validation set: {len(validation_set)} rows")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
model.eval()


open(TXT_LOG_FILE, "w", encoding="utf-8").close()
open(TIMING_LOG_FILE, "w", encoding="utf-8").close()

results = []
total_start = time.time()

with open(TXT_LOG_FILE, "a", encoding="utf-8") as txt_log, \
     open(TIMING_LOG_FILE, "a", encoding="utf-8") as time_log:
    time_log.write("id\tgeneration_time_sec\ntask_id\ttime\n")

    for row_idx in tqdm(range(len(validation_set)), desc="Validation"):
        row = validation_set.iloc[row_idx]

        row_id = row.get("id", row_idx) if "id" in validation_set.columns else row_idx
        
        system_prompt = system_instruction + "\n" + extra_system_instruction
        
        user_prompt = f"Context:\n{row['sql_context']}\n\nQuestion:\n{row['sql_prompt']}\n\nMy Code Answer Will Be:"
        messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt}
		]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     repetition_penalty=1.3,
                                     no_repeat_ngram_size=4,
                                     num_beams=1,
                                     pad_token_id=tokenizer.eos_token_id,)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        generation_time = end_time - start_time

        input_len = inputs.input_ids.shape[-1]
        response = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        
        print(response)
        txt_log.write(f"=== ID: {row_id} ===\n")
        txt_log.write(f"--- PROMPT ---\n{user_prompt}\n")
        txt_log.write(f"--- RESPONSE ---\n{response}\n")
        txt_log.write(f"--- TIME: {generation_time:.4f}s ---\n")
        txt_log.write("=" * 60 + "\n")
        txt_log.flush()

        time_log.write(f"{row_id}\t{generation_time:.4f}\n")
        time_log.flush()

        results.append(
            {
                "id": row_id,
                "prompt": user_prompt,
                "response": response,
                "generation_time_sec": generation_time,
            }
        )

total_time = time.perf_counter() - total_start

results_df = pd.DataFrame(results)
results_df.to_parquet(PARQUET_OUTPUT, index=False, engine="pyarrow")  



print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
print(f"Total rows processed : {len(results_df)}")
print(f"Total time           : {total_time:.2f}s")
print(f"Mean generation time : {results_df['generation_time_sec'].mean():.4f}s")
print(f"Median generation    : {results_df['generation_time_sec'].median():.4f}s")
print(f"Min / Max time       : {results_df['generation_time_sec'].min():.4f}s / "
      f"{results_df['generation_time_sec'].max():.4f}s")
print(f"\nFiles saved:")
print(f"  - {TXT_LOG_FILE}")
print(f"  - {TIMING_LOG_FILE}")
print(f"  - {PARQUET_OUTPUT}")