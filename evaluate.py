import torch
import json
import time
from datasets import load_from_disk
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm

def main():
    model_id = "google/paligemma2-3b-pt-224"
    lora_path = "./paligemma-eggplant-lora-final" # 学習後の保存先
    
    # 1. データセットの読み込み (testを使う)
    dataset_dict = load_from_disk("./eggplant_dataset")
    test_ds = dataset_dict["test"]
    
    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    # 逆マッピング (ID -> 疾患名)の作成
    id_to_class = {v: k for k, v in class_mapping.items()}

    # 2. プロセッサとモデルの読み込み
    print("Loading models (this might take a while)...")
    processor = AutoProcessor.from_pretrained(model_id) # トークナイザー等

    # ベースモデルはbfloat16で読み込む (量子化せずに高速推論)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # LoRAアダプタの統合
    model = PeftModel.from_pretrained(base_model, lora_path).eval()

    # 3. 推論の実行 (1件ずつ)
    correct = 0
    total = len(test_ds)
    print(f"\n=== テストデータ {total}件で推論開始 ===")
    
    start_time = time.time()
    
    for i, ex in enumerate(tqdm(test_ds)):
        prompt = ex["prompt"]
        true_label = ex["label"]
        image = ex["image"].convert("RGB")
        
        # 前処理
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5, # クラス番号だけが出力されるはずなので短くてOK
                do_sample=False
            )
        
        # 入力プロンプト部分のトークンをスキップしてデコード
        generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        # 評価
        if generated_text == true_label:
            correct += 1
            
        # 最初の10件だけ詳細をプリント
        if i < 10:
            pred_class = id_to_class.get(generated_text, "Unknown")
            true_class = id_to_class.get(true_label, "Unknown")
            mark = "✅" if generated_text == true_label else "❌"
            print(f"[{mark}] Pred: {generated_text}({pred_class}) | True: {true_label}({true_class})")

    # 4. 結果の計算
    acc = correct / total * 100
    print(f"\n=== 最終結果 ===")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
