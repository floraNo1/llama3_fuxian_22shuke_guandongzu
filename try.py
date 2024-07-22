import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from typing import Dict, List

# 配置模型参数
hidden_size = 256
intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128

# 创建模型配置
config = AutoConfig.for_model(
    model_type='llama',  # 模型类型
    hidden_size=hidden_size,  # 模型的隐藏层大小
    intermediate_size=intermediate_size,  # 前馈层的中间大小
    num_attention_heads=16,  # 注意力头的数量gg
    num_hidden_layers=4,  # 隐藏层的数量
    num_key_value_heads=8  # 键值头的数量
)

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')

# 确定使用的设备（如果有GPU则使用GPU，否则使用CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用指定配置初始化模型
model = AutoModelForCausalLM.from_config(
    config,
    torch_dtype=torch.float32
).to(device)  # 将模型移动到指定设备


# 用于模型参数的Kaiming初始化的函数
def kaiming_initialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # 对权重应用初始化
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:  # 将偏置设置为零
            torch.nn.init.constant_(param, 0)

# 对模型应用Kaiming初始化
kaiming_initialization(model)

# 数据处理函数，用于准备训练数据集
def process_func(examples: Dict[str, List]) -> Dict[str, List]:
    max_token = 2048  # 最大token长度
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)  # 对输入文本进行分词
    input_ids_list = encoded_texts['input_ids']
    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        # 截断并添加EOS token
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))  # 创建注意力掩码
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
    ds_train = ds_train.shuffle()  # 打乱训练数据集
    ds_train = ds_train.map(process_func, batched=True, num_proc=num_proc, remove_columns=ds_train.column_names, desc='对训练集进行分词处理:')
    ds_val = ds_val.map(process_func, batched=True, num_proc=num_proc, remove_columns=ds_val.column_names, desc='对验证集进行分词处理:')

    print(ds_train)
    print(ds_val)

    # 设置数据收集器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='/param_/',  # 输出目录
        overwrite_output_dir=True,  # 是否覆盖输出目录
        do_train=True,  # 是否进行训练
        do_eval=True,  # 是否进行评估
        eval_steps=1000,  # 评估间隔步数
        per_device_train_batch_size=4,  # 每个设备的训练批次大小
        gradient_accumulation_steps=1,  # 梯度累积步数
        learning_rate=1e-4,  # 学习率
        lr_scheduler_type='cosine',  # 学习率调度器类型
        bf16=torch.cuda.is_bf16_supported(),  # 是否使用BF16
        fp16=not torch.cuda.is_bf16_supported(),  # 是否使用FP16
        logging_steps=50,  # 日志记录步数
        num_train_epochs=2,  # 训练轮数
        save_steps=1000,  # 保存模型间隔步数
        save_total_limit=2,  # 最大保存模型数量
        seed=3407  # 随机种子
    )

    # 创建Trainer实例
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
