import os
import json
from datasets import Dataset, DatasetDict, Image

def main():
    # 1. config.jsonからデータセットのパスを取得
    with open("config.json", "r") as f:
        config = json.load(f)
    
    dataset_dir = config["dataset_dir"]
    
    # 2. クラスフォルダをスキャンしてIDを割り当てる
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_id = {name: str(i) for i, name in enumerate(class_names)}
    
    print("=== クラスとIDのマッピング ===")
    for name, cid in class_to_id.items():
        print(f"  ID {cid}: {name}")
        
    # 推論時に表示を元に戻すために、マッピングを保存しておく
    with open("class_mapping.json", "w") as f:
        json.dump(class_to_id, f, indent=2, ensure_ascii=False)
        
    data = {"image": [], "prompt": [], "label": []}
    
    # 3. 全ての画像パスを収集
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                
                # PaliGemmaに分類させるための短い固定プロンプト
                prompt = "disease type"
                
                data["image"].append(img_path)
                data["prompt"].append(prompt)
                data["label"].append(class_to_id[class_name])  # "0", "1" などの文字列データ
                
    # 4. Hugging Face Datasetオブジェクトの作成
    dataset = Dataset.from_dict(data)
    
    # 'image'カラムをImage型にキャッシュ指定（訓練時に自動でPIL Imageとして読み込まれます）
    dataset = dataset.cast_column("image", Image())
    
    # 5. 学習用(train), 検証用(val), テスト用(test) に分割する (例: 70%, 15%, 15%)
    # まず trainと一時的なtest（合わせて30%）に分ける
    train_test = dataset.train_test_split(test_size=0.3, seed=42)
    
    # 次に一時的なtestを半分（それぞれ15%）に分けて val と test にする
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    # 辞書型にまとめる
    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "val": val_test["train"],
        "test": val_test["test"]
    })
    
    # 6. ディスクに保存
    save_path = "./eggplant_dataset"
    dataset_dict.save_to_disk(save_path)
    
    print(f"\n=== データセット作成完了 ===")
    print(f"保存先: {save_path}")
    print(f"Train件数: {len(dataset_dict['train'])} 件")
    print(f"Val件数:   {len(dataset_dict['val'])} 件")
    print(f"Test件数:  {len(dataset_dict['test'])} 件")

if __name__ == "__main__":
    main()
