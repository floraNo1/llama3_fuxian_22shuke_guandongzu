import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from typing import Dict, List

# 模型配置
hidden_size = 256
intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128

config = AutoConfig.for_model(
    model_type='llama',
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_attention_heads=16,
    num_hidden_layers=4,
    num_key_value_heads=8
)

# 分词器
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_config(
    config,
    torch_dtype=torch.float32
).to(device)

# 打印模型的每一层及其参数大小
def print_model_parameters(model):
    print('Layer Name & Parameters')
    print('----------------------------')
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f'{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}')
    print('----------------------------')
    print(f'Total Parameters: {total_params} ({total_params / 1000000:.1f} M)')

print_model_parameters(model)

# Kaiming 初始化
def kaiming_initialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)

kaiming_initialization(model)

# 数据处理函数
def process_func(examples: Dict[str, List]) -> Dict[str, List]:
    max_token = 2048
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)
    input_ids_list = encoded_texts['input_ids']
    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))
    return {
        'input_ids': new_input_ids_list,
        'attention_mask': new_attn_mask_list
    }

if __name__ == '__main__':
    dataset_name_or_path = 'noanabeshima/TinyStoriesV2'
    ds_train = load_dataset(dataset_name_or_path, split='train[:10%]')
    ds_val = load_dataset(dataset_name_or_path, split='validation')

    print(ds_train)
    print(ds_val)

    num_proc = 8
    ds_train = ds_train.shuffle()
    ds_train = ds_train.map(process_func, batched=True, num_proc=num_proc, remove_columns=ds_train.column_names, desc='Running tokenizer on train_set:')
    ds_val = ds_val.map(process_func, batched=True, num_proc=num_proc, remove_columns=ds_val.column_names, desc='Running tokenizer on val_set:')

    print(ds_train)
    print(ds_val)

    # 设置 DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='saves',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_steps=1000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        num_train_epochs=2,
        save_steps=1000,
        save_total_limit=2,
        seed=3407
    )

    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train()
