import fiftyone as fo
import fiftyone.zoo as foz

# 僅從 COCO 驗證集中，快速下載並載入包含 "dog" 且結構清晰的 50 張圖片
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    classes=["fox"],
    max_samples=10
)

# 啟動 FiftyOne 網頁端 WebUI（它會在你本地開一個瀏覽器視窗）
session = fo.launch_app(dataset)
session.wait()