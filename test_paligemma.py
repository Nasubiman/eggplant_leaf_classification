import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import json
import time

# 1. パスとモデルの設定
model_id = "google/paligemma2-3b-pt-224"

# config.json から画像パスを読み込む
with open("config.json", "r") as f:
    config = json.load(f)
image_path = config["test_image_path"]

print(f"1. Loading {model_id}...")
start_time = time.time()

# 2. bfloat16 でモデルとプロセッサの読み込み (VRAM節約と速度向上のため)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(model_id)

# 12GB VRAMでPaliGemma-3B(約30億パラメータ)を動かすため、自動オフロードを許可するdevice_mapを使用、かつbfloat16
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
).eval()  # 推論モード

print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds on {device}!")

# 3. 画像の読み込みとプロンプトの作成
print("2. Preparing image and prompt...")
# pre-trainedモデル（ファインチューニング前）なので、一般的なプロンプトでどのような出力が出るか確認
prompt = "question: what is the health status of this eggplant leaf in the image?"
image = Image.open(image_path).convert("RGB")

print(f"Image Path: {image_path}")

# 4. 前処理と推論
print("3. Processing inputs and running inference...")
# processorを使用して、テキストと画像をモデルの入力形式に変換
inputs = processor(text=prompt, images=image, return_tensors="pt")

# デバイスへの転送
# floating-pointのテンソルのみdtypeを指定し、integer型のinput_ids等はto(device)のみ行う
inputs = {k: v.to(dtype=dtype, device=device) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

# 推論実行
infer_start = time.time()
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False  # 確定的（Deterministic）な生成にする
    )

print(f"Inference completed in {time.time() - infer_start:.2f} seconds.")

# 5. 後処理
# promptに含まれる文字もモデル出力に含まれるため、元の長さより後ろを取得する
# またbatch_decodeを使用して文字列に戻す
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
predicted_text = generated_texts[0][len(prompt):].strip()

print("\n=== Inference Result ===")
print(f"Prompt: {prompt}")
print(f"Output: {predicted_text}")
print("========================")
