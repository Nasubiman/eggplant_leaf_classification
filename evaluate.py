import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import json
import time
from datasets import load_from_disk
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, LogitsProcessor
from peft import PeftModel
from tqdm import tqdm


class ClassConstrainedLogitsProcessor(LogitsProcessor):
    """推論時に、指定されたクラスIDのトークンだけを出力可能にする。
    それ以外のトークンの logits を -inf に設定することで、
    モデルが 0〜6 以外の値を絶対に出力しないように強制する。
    """
    def __init__(self, allowed_token_ids, eos_token_id):
        self.allowed_token_ids = allowed_token_ids
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        # 最初のトークンでは、許可されたクラスIDのみ出力可能にする
        # 2トークン目以降は EOS のみ許可 (数字1文字だけ出力して終了させる)
        generated_len = input_ids.shape[-1]

        mask = torch.full_like(scores, float("-inf"))

        if generated_len == input_ids.shape[-1]:
            # まだ1トークンも生成していない → クラスID用トークンを許可
            for tid in self.allowed_token_ids:
                mask[:, tid] = scores[:, tid]
        
        # EOS も常に許可 (生成を止められるように)
        mask[:, self.eos_token_id] = scores[:, self.eos_token_id]

        return mask


def main():
    model_id = "google/paligemma2-3b-pt-224"
    lora_path = "./paligemma-eggplant-lora-final"  # 学習後の保存先

    # 1. データセットの読み込み (testを使う)
    dataset_dict = load_from_disk("./eggplant_dataset")
    test_ds = dataset_dict["test"]

    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    # 逆マッピング (ID -> 疾患名)の作成
    id_to_class = {v: k for k, v in class_mapping.items()}
    num_classes = len(class_mapping)

    # 2. プロセッサとモデルの読み込み
    print("Loading models (this might take a while)...")
    processor = AutoProcessor.from_pretrained(model_id)

    # RTX 2080 Ti は bf16 非対応 → float16 を使用
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    # LoRAアダプタの統合
    model = PeftModel.from_pretrained(base_model, lora_path).eval()

    # 3. Constrained Decoding 用の準備
    #    "0", "1", ..., "6" に対応するトークンIDを取得する
    tokenizer = processor.tokenizer
    allowed_token_ids = []
    for i in range(num_classes):
        token_id = tokenizer.encode(str(i), add_special_tokens=False)
        allowed_token_ids.append(token_id[0])
        print(f"  Class {i} ({id_to_class[str(i)]}): token_id = {token_id[0]}")

    eos_token_id = tokenizer.eos_token_id
    constrained_processor = ClassConstrainedLogitsProcessor(allowed_token_ids, eos_token_id)
    print(f"Constrained Decoding: allowed tokens = {allowed_token_ids}, EOS = {eos_token_id}")

    # 4. 推論の実行 (1件ずつ)
    correct = 0
    total = len(test_ds)
    print(f"\n=== テストデータ {total}件で推論開始 ===")

    start_time = time.time()

    for i, ex in enumerate(tqdm(test_ds)):
        prompt = "<image>" + ex["prompt"]
        true_label = ex["label"]
        image = ex["image"].convert("RGB")

        # 前処理
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 推論 (Constrained Decoding で 0〜6 のみ出力)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,  # 数字1文字 + EOS のみ
                do_sample=False,
                logits_processor=[constrained_processor],
            )

        # 入力プロンプト部分のトークンをスキップしてデコード
        generated_text = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

        # 評価
        if generated_text == true_label:
            correct += 1

        # 最初の10件だけ詳細をプリント
        if i < 10:
            pred_class = id_to_class.get(generated_text, "Unknown")
            true_class = id_to_class.get(true_label, "Unknown")
            mark = "✅" if generated_text == true_label else "❌"
            print(f"[{mark}] Pred: {generated_text}({pred_class}) | True: {true_label}({true_class})")

    # 5. 結果の計算
    acc = correct / total * 100
    print(f"\n=== 最終結果 ===")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
