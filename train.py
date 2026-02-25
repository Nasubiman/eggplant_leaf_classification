import torch
import os
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

    # 3. 4bit量子化設定 (VRAM 12GB環境用)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # モデルの読み込み
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" # 自動的にGPUへ割り当て
    )
    
    # Vision Encoderなど不必要なパラメータの学習を防ぎ、LoRAを言語モデル部分のアテンション等に適用
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. データコレーター (バッチ作成処理)の定義
    def collate_fn(examples):
        texts = [ex["prompt"] for ex in examples]
        labels = [ex["label"] for ex in examples]
        images = [ex["image"].convert("RGB") for ex in examples]
        
        # suffixに正解ラベル（生成させたい文字列）のみを渡すことで、Trainerが勝手にinput・target・loss計算用のマスク(-100)をしてくれます
        batch = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest"
        )
        return batch

    # 5. TrainingArgumentsの設定 (12GB VRAMに合わせて極力細かく制御)
    # 実際にOOM (Out Of Memory) が出る場合は、batch_sizeを下げ、accumulationを上げてください。
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",  # 各エポック終わりにValidationを実行
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        num_train_epochs=3, # まずは3エポック
        bf16=True,  # RTX 30/40シリーズ等での高速・安定化
        optim="paged_adamw_8bit", # Optimizerも8bit化してメモリ節約
        remove_unused_columns=False, # image等の独自カラムが消されないように必須
        dataloader_pin_memory=False, 
        report_to="none" # wandb等を使わない場合はnone
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn
    )

    # 6. 学習実行
    print("=== Start Training ===")
    trainer.train()
    
    # 7. 最終モデルの保存
    final_save_path = f"{output_dir}-final"
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"=== Saved Model to {final_save_path} ===")

if __name__ == "__main__":
    main()
