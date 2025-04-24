# Finetuning LLaMA 3.2B Instruct with Unsloth

This project demonstrates how to fine-tune the **LLaMA 3.2B Instruct** model using the **Unsloth** framework and **QLoRA** for memory-efficient adaptation. The training data is sourced from the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset, formatted in ShareGPT style.

## Features

- Uses **Unsloth** for fast and efficient fine-tuning
- Utilizes **QLoRA** for 4-bit quantization
- Adopts **SFTTrainer** from Hugging Face's `trl` library
- Compatible with **Colab GPU environments**
- Saves and reloads fine-tuned models for inference

---

## Dependencies

Install all required packages:

```bash
pip install unsloth transformers trl datasets
```

---

## File Structure

- `finetuning_llama_3_2b.py`: Main training and inference script

---

## Steps Performed

### 1. Load Pretrained Model
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
```

### 2. Apply QLoRA (PEFT)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

### 3. Load and Format Dataset
```python
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
```

### 4. Format Data with Chat Template
```python
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

### 5. Train the Model
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    ...
)
trainer.train()
```

### 6. Save and Load the Model for Inference
```python
model.save_pretrained("finetuned_model")
```

---

## Inference Example

After training:

```python
inference_model, inference_tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True
)
```

Then generate responses to prompts like:

```python
"what are the key principles of a successful career?"
```

## Output

- Model is saved at `./finetuned_model`
- Trained weights are ready for downstream use

---
