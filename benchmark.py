import pandas as pd
import sqlite3
import math
import re
from tqdm import tqdm

results = pd.read_parquet("results_bigger_Qwen/validation_results.parquet") 
gt_set = pd.read_parquet("datasets/validation_mini.parquet") 
merged_df = pd.merge(results, gt_set, on='id', how='inner')


def clean_sql(query: str) -> str:
    if not isinstance(query, str):
        return ""
    query = re.sub(r'^```sql\s*', '', query, flags=re.IGNORECASE | re.MULTILINE)
    query = re.sub(r'^```\s*', '', query, flags=re.MULTILINE)
    query = re.sub(r'```$', '', query, flags=re.MULTILINE)
    if ';' in query:
        query = query.split(';')[0] + ';'
    return query

def normalize_result(result):
    if result is None:
        return None
    try:
        return sorted([tuple(r) for r in result])
    except TypeError:
        return [tuple(r) for r in result]

def compare_results(model_res, gt_res):
    if model_res is None or gt_res is None:
        return False
    
    model_res = normalize_result(model_res)
    gt_res = normalize_result(gt_res)
    
    if len(model_res) != len(gt_res):
        return False
        
    for m_row, g_row in zip(model_res, gt_res):
        if len(m_row) != len(g_row):
            return False
            
        for m_val, g_val in zip(m_row, g_row):
            if m_val is None and g_val is None:
                continue
            if m_val is None or g_val is None:
                return False
                
            if isinstance(m_val, float) and isinstance(g_val, float):
                if not math.isclose(m_val, g_val, rel_tol=1e-5, abs_tol=1e-5):
                    return False
            elif m_val != g_val:
                return False
    return True

def evaluate_row(row):
    sql_context = row['sql_context']
    model_sql = clean_sql(row['response'])
    gt_sql = row['sql']

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    try:
        cursor.executescript(sql_context)
        
        cursor.execute(gt_sql)
        gt_result = cursor.fetchall()
        
        cursor.execute(model_sql)
        model_result = cursor.fetchall()
        
        is_correct = compare_results(model_result, gt_result)
        error_type = "None" if is_correct else "Result Mismatch"
        
    except sqlite3.Error as e:
        is_correct = False
        error_msg = str(e).lower()
        if "no such table" in error_msg or "no such column" in error_msg:
            error_type = "Schema Error (Table/Column)"
        elif "syntax error" in error_msg:
            error_type = "Syntax Error"
        else:
            error_type = f"SQLite Error: {str(e)[:50]}"
    except Exception as e:
        is_correct = False
        error_type = f"Unexpected Error: {str(e)[:50]}"
    finally:
        conn.close()
        
    return is_correct, error_type

correct_count = 0
error_counts = {}

for i, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Evaluating"):
    is_correct, error_type = evaluate_row(row)
    
    merged_df.at[i, 'is_correct'] = is_correct
    merged_df.at[i, 'error_type'] = error_type
    
    if is_correct:
        correct_count += 1
    else:
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

total = len(merged_df)
accuracy = (correct_count / total) * 100 if total > 0 else 0

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total:       {total}")
print(f"True answers:      {correct_count} ({accuracy:.2f}%)")
print(f"Wrong answers:   {total - correct_count}")
print("-" * 60)
print("Errors distribution:")
for err_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {err_type}: {count} ({(count/total)*100:.1f}%)")
print("="*60)

output_path = "validation_results_with_eval.parquet"
merged_df.to_parquet(output_path, index=False)
print(f"\nResults in: {output_path}")