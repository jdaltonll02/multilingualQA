import torch
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments


def load_model(model_name: str) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.tie_word_embeddings = False
    return model


def build_training_args(
    *,
    output_dir: str,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    label_smoothing_factor: float,
    optim: str,
    predict_with_generate: bool,
    generation_max_length: int,
    eval_strategy: str,
    save_strategy: str,
    load_best_model_at_end: bool,
    metric_for_best_model: str,
    fp16: bool,
    bf16: bool,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    dataloader_num_workers: int,
    logging_steps: int,
    save_total_limit: int,
    report_to: str,
    seed: int,
) -> Seq2SeqTrainingArguments:
    has_gpu = torch.cuda.is_available()
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor,
        optim=optim,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        fp16=fp16 and has_gpu,
        bf16=bf16 and has_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=has_gpu,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        seed=seed,
        data_seed=seed,
    )
