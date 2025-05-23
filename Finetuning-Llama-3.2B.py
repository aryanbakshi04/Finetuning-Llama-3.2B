# -*- coding: utf-8 -*-
"""Finetuning Llama 3.2B
"""

!pip install unsloth transformers trl

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

model, tokenizer=FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)

# using Qlora
model=FastLanguageModel.get_peft_model(
    model,r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

tokenizer=get_chat_template(tokenizer,chat_template="llama-3.1")

!pip install datasets
from datasets import load_dataset

dataset = load_dataset("mlabonne/FineTome-100k", split="train")

dataset=standardize_sharegpt(dataset)

dataset

dataset[0]

dataset=dataset.map(
    lambda examples:{
        "text":[
            tokenizer.apply_chat_template(convo,tokenize=False)
            for convo in examples["conversations"]
        ]
    },
    batched=True
)

dataset

dataset[0]

trainer=SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="ouputs"
    ),
)

trainer.train()

model.save_pretrained("finetuned_model")

inference_model, inference_tokenizer= FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True
)

text_prompts=[
    "what are the key priniciples of a successful career?"
]
for prompt in text_prompts:
  formatted_prompt=inference_tokenizer.apply_chat_template([{
      "role":"user",
      "content":prompt
  }],
      tokenize=False
)
  model_inputs=inference_tokenizer(formatted_prompt,return_tensors="pt").to("cuda")
  generated_ids=inference_model.generate(
      **model_inputs,
      max_new_tokens=512,
      temperature=0.7,
      do_sample=True,
      pad_token_id=inference_tokenizer.pad_token_id
  )

  response=inference_tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
  print(response)

