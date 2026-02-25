import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 のみ使用 (RTX 2080 Ti)

import torch
from datasets import load_from_disk
from transformers import (
    AutoProcessor, PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig

def main():
    model_id = "google/paligemma2-3b-pt-224"
    output_dir = "./paligemma-eggplant-lora"

    # 1. データセットの読み込み (trainとvalを使う)
    dataset_dict = load_from_disk("./eggplant_dataset")
    train_ds = dataset_dict["train"]
    val_ds = dataset_dict["val"]

    print(f"Training on {len(train_ds)} samples, Validating on {len(val_ds)} samples.")

    # 2. プロセッサの読み込み
    processor = AutoProcessor.from_pretrained(model_id)

    # 3. 4bit量子化 + モデルの読み込み
    #    RTX 2080 Ti は Turing アーキテクチャなので bf16 非対応 → compute_dtype を float16 にする
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # RTX 2080 Ti 用
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},  # 単一GPU明示指定 (Acceleratorとの衝突回避)
    )

    # 4. Vision Encoder を完全に凍結する
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    # 5. LoRA を言語モデル部分にのみ適用する
    #    modules_to_save は不要。target_modules で言語モデル内のLinear層だけを指定。
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=None,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. gradient checkpointing を安全に有効化
    #    enable_input_require_grads() を呼ぶことで、vision encoder 出力から
    #    言語モデルへの受け渡し時に requires_grad=True が付与され、
    #    gradient checkpointing が正常に動作するようになる
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 7. データコレーター (バッチ作成処理)の定義
    def collate_fn(examples):
        # PaliGemma の推奨に従い、プロンプトの先頭に <image> トークンを付与する
        texts = ["<image>" + ex["prompt"] for ex in examples]
        labels = [ex["label"] for ex in examples]
        images = [ex["image"].convert("RGB") for ex in examples]

        # suffix に正解ラベルを渡すと、自動で labels のマスク (-100) 処理が行われる
        batch = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
        )
        return batch

    # 8. TrainingArguments の設定
    #    RTX 2080 Ti (11GB VRAM) に合わせた省メモリ設定
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,   # 実質バッチサイズ 1x8=8
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-4,
        num_train_epochs=3,
        fp16=True,                       # RTX 2080 Ti は fp16 を使う (bf16 非対応)
        optim="paged_adamw_8bit",        # Optimizer のステートを 8bit にして VRAM 節約
        gradient_checkpointing=True,     # 中間アクティベーションを再計算してメモリ削減
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=4,       # データ読み込みをCPUが並列処理し、GPUの待ち時間を削減
        dataloader_pin_memory=True,     # GPUへの転送を高速化
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # 9. 学習実行
    print("=== Start Training ===")
    trainer.train()

    # 10. 最終モデルの保存
    final_save_path = f"{output_dir}-final"
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"=== Saved Model to {final_save_path} ===")

if __name__ == "__main__":
    main()
