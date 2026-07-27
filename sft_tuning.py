import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint  # Для поиска последнего чекпоинта
from trl import SFTTrainer
from datasets import Dataset 
from config import (
    TRAIN_PATH, system_instruction, extra_system_instruction, 
    MODEL_NAME, TXT_LOG_FILE, TIMING_LOG_FILE, MAX_NEW_TOKENS
)
from peft import LoraConfig


def format_dataset(example):
    system_prompt = system_instruction + "\n" + extra_system_instruction
    user_prompt = f"Context:\n{example['sql_context']}\n\nQuestion:\n{example['sql_prompt']}"
    assistant_prompt = example['sql']
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]
    }


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


quantization_config = None
if torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True     
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    quantization_config=quantization_config
)

lora_config = LoraConfig(
    r=16,                          # Ранг матриц адаптера (8, 16 или 32)
    lora_alpha=32,                 # Масштабирующий коэффициент
    lora_dropout=0.05,             # Вероятность dropout для слоев LoRA
    bias="none",                   # Не обучать bias-параметры
    task_type="CAUSAL_LM",         # Тип задачи
    # Для моделей Qwen целевые модули обычно такие:
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)



raw_dataset = Dataset.from_parquet(TRAIN_PATH)
formatted_dataset = raw_dataset.map(format_dataset, remove_columns=raw_dataset.column_names)


eval_sample = formatted_dataset.select(range(min(3, len(formatted_dataset))))


class LogModelPredictionsCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, log_file):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.log_file = log_file

    def on_step_end(self, args, state, control, model=None, **kwargs):
        
        if state.global_step % args.save_steps == 0 and state.global_step > 0:
            print(f"\n[Step {state.global_step}] Generating test SQL...")
            model.eval()
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*20} Learning step {state.global_step} {'='*20}\n")
                
                for i, example in enumerate(self.eval_dataset):
                    
                    prompt_messages = example['messages'][:-1] 
                    prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs = self.tokenizer(prompt_text, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=MAX_NEW_TOKENS,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            do_sample=False 
                        )
                    
                    
                    input_len = inputs.input_ids.shape[1]
                    generated_sql = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                    target_sql = example['messages'][-1]['content']
                    
                    f.write(f"Тест {i+1}:\n")
                    f.write(f"Ожидалось: {target_sql}\n")
                    f.write(f"Модель выдала: {generated_sql.strip()}\n")
                    f.write(f"{'-'*40}\n")
                    
            model.train()

log_callback = LogModelPredictionsCallback(eval_sample, tokenizer, TXT_LOG_FILE)


training_args = TrainingArguments(
    output_dir="./sql_fine_tuned_checkpoints", 
    per_device_train_batch_size=1,             
    gradient_accumulation_steps=4,             
    logging_steps=10,                          
    save_steps=25,                                                    
    num_train_epochs=3,                        
    learning_rate=2e-4,                        
    bf16=torch.cuda.is_available(),            
    logging_dir="./logs",                      
    report_to=["tensorboard"],                 
    remove_unused_columns=False,
    gradient_checkpointing=True             
)

last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"Found last checkpoint: {last_checkpoint}")
    else:
        print("Checkpoint not found")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
    callbacks=[log_callback],
    peft_config=lora_config,       
)

trainer.train(resume_from_checkpoint=last_checkpoint)


trainer.save_model("./sql_fine_tuned_final")
tokenizer.save_pretrained("./sql_fine_tuned_final")
print("Done.")
