# SADis 實驗操作說明

SADis 是一個基於 SDXL + IP-Adapter 的三路參考圖生成實驗。核心目標是把輸入拆成三個 stream：

- `color`: 控制整體色彩與色調。
- `geometry`: 控制輪廓、姿態、構圖與形狀。
- `texture`: 控制表面紋理、筆觸與材質細節。

生成時，文字 prompt 會先經過 SDXL text encoder；三張參考圖會經過 CLIP image encoder 與 IP-Adapter projection，接著把 `color / geometry / texture` tokens 串到 text tokens 後面，送進修改過的 SDXL pipeline 做 cross-attention。

## 專案結構

| 路徑 | 用途 |
| --- | --- |
| `infer_style_plus_color_texture.py` | 單次 SADis 生成入口，使用 `ip_adapter/` 與 `pipeline_stable_diffusion_xl.py`。 |
| `run_infer_style_plus_all_modes.py` | 連續跑 `color_only`、`color_texture`、`color_geometry`、`color_texture_geometry` 四種輸入組合。 |
| `run_dog_tuning_experiments.py` | 批次 sweep 入口，會生成圖片、baseline、metrics 與 summary。 |
| `compute_single_image_metrics.py` | 對單張輸出計算 color / geometry / texture 相關指標。 |
| `texture_geometry_color_prompts.json` | 從 InstantStyle 複製過來的 prompt 清單；需先轉成 SADis prompt map 才能餵給 `--prompt-file`。 |
| `ip_adapter/` | SADis 主要三路 stream 與 attention routing 實作。 |
| `ip_adapter_equal_attention/` | equal-attention 版本。 |
| `ip_adapter_equal_attention_no_orthogonal/` | equal-attention 且不做 orthogonal decouple 的 ablation 版本。 |
| `assets/color/` | color reference 圖片。 |
| `assets/geometry/` | geometry reference 圖片。 |
| `assets/texture/` | texture reference 圖片。 |
| `results/` | 實驗輸出目錄。 |
| `docs/` | 方法分析與架構說明。 |

## 環境準備

建議在 PowerShell 裡從 repo 根目錄執行：

```powershell
cd C:\Users\tianachen\SADis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

核心套件：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install diffusers transformers safetensors accelerate einops pillow pillow-avif-plugin opencv-python numpy scipy tqdm lpips pytorch-msssim
```

如果要跑 Gemini texture 描述：

```powershell
python -m pip install google-genai
$env:GEMINI_API_KEY="你的 API key"
```

模型檔案預期放在：

```text
models/image_encoder/
sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
```

SDXL base model 使用 `stabilityai/stable-diffusion-xl-base-1.0`，會由 Hugging Face / diffusers 自動讀取或下載。

## 單次生成

最直接的跑法是使用預設參考圖與 prompt：

```powershell
python infer_style_plus_color_texture.py
```

輸出會存到程式內的 `save_dir`，目前預設是：

```text
results/disentangled/18.4.8/repro_20260504
```

要換三張參考圖與輸出資料夾，使用環境變數：

```powershell
$env:SADIS_INPUT_COLOR="assets/color/color05.jpg"
$env:SADIS_INPUT_GEOMETRY="assets/geometry/animal08.jpeg"
$env:SADIS_INPUT_TEXTURE="assets/texture/artwork_3.jpg"
$env:SADIS_SAVE_DIR="results/manual/animal08_color05_artwork3"
python infer_style_plus_color_texture.py
```

要關掉某一路 stream，把它設成 `None`：

```powershell
$env:SADIS_INPUT_GEOMETRY="None"
python infer_style_plus_color_texture.py
```

equal-attention 版本可以直接用 CLI 傳入：

```powershell
python infer_style_plus_color_texture_equal_attention.py `
  --input-color assets/color/color05.jpg `
  --input-geometry assets/geometry/face05.jpeg `
  --input-texture assets/texture/artwork_3.jpg `
  --save-dir results/manual/equal_attention_face05
```

## 四種 stream 組合

要比較不同條件怎麼接上模型，可以用：

```powershell
python run_infer_style_plus_all_modes.py `
  --color assets/color/color05.jpg `
  --geometry assets/geometry/animal08.jpeg `
  --texture assets/texture/artwork_3.jpg `
  --save-root results/all_modes/animal08_color05_artwork3
```

這會依序產生：

- `color_only`: 只接 color。
- `color_texture_legacy`: 接 color + texture。
- `color_geometry`: 接 color + geometry。
- `color_texture_geometry`: 三路都接。

先確認會跑哪些設定但不真的生成：

```powershell
python run_infer_style_plus_all_modes.py --dry-run
```

## 批次 sweep 實驗

批次實驗會從 `assets/geometry`、`assets/color`、`assets/texture` 組合 triple，生成圖片後自動算 metrics。

先跑小樣本確認環境：

```powershell
python run_dog_tuning_experiments.py `
  --geometry-categories animal `
  --limit 2 `
  --output-root results/smoke_animal `
  --skip-cross-geometry-lpips
```

正式跑某一類：

```powershell
python run_dog_tuning_experiments.py `
  --geometry-categories animal face house seen `
  --pairing cartesian `
  --output-root results/sadis_main_sweep `
  --guidance-scale 9.0 `
  --color-scale 1.6 `
  --geometry-scale 1.4 `
  --texture-scale 1.4 `
  --texture-color-decouple 0.10 `
  --geometry-color-decouple 0.35 `
  --color-to-geometry-decouple 0.25 `
  --punish-weight 0.0003 `
  --skip-cross-geometry-lpips
```

輸出結構：

```text
results/sadis_main_sweep/
  images/        # 每個 triple + prompt 的生成圖
  baselines/     # SDXL baseline，用於 residual / improvement metrics
  debug/         # geometry preprocess 與 canny debug 圖
  metrics.csv    # 每張圖的指標
  summary.json   # 整體平均指標
```

如果圖片已經生成，只想重算 metrics：

```powershell
python run_dog_tuning_experiments.py `
  --geometry-categories animal face house seen `
  --output-root results/sadis_main_sweep `
  --metrics-only `
  --skip-cross-geometry-lpips
```

## Prompt 檔案串接

`run_dog_tuning_experiments.py --prompt-file` 需要的是 JSON object，格式是 `geometry_id` 或 triple key 對到 prompt：

```json
{
  "animal01": [
    "A single dog on the ground, frontal view, soft pastel lines.",
    "A single cat on the ground, frontal view, soft pastel lines."
  ],
  "face01": "A portrait of a man with a beard"
}
```

使用方式：

```powershell
python run_dog_tuning_experiments.py `
  --prompt-file prompts/sadis_prompt_map.json `
  --geometry-categories animal face `
  --output-root results/with_prompt_map
```

注意：`texture_geometry_color_prompts.json` 目前是從 InstantStyle 複製過來的結構：

```json
{
  "geometry_prompts": [
    {
      "geometry_id": "face01",
      "geometry_image": "assets copy/geometry/face01.jpeg",
      "prompt": "Portrait of a man with a beard..."
    }
  ]
}
```

這不能直接丟給 `--prompt-file`。要先轉成 SADis 的 mapping 格式，也就是把同一個 `geometry_id` 底下的 prompt 收成 list。

可以用下面的 PowerShell 片段轉檔：

```powershell
New-Item -ItemType Directory -Force prompts | Out-Null
@'
import json
from collections import defaultdict
from pathlib import Path

src = Path("texture_geometry_color_prompts.json")
dst = Path("prompts/sadis_prompt_map.json")

data = json.loads(src.read_text(encoding="utf-8-sig"))
prompt_map = defaultdict(list)

for row in data.get("geometry_prompts", []):
    geometry_id = row.get("geometry_id")
    prompt = row.get("prompt")
    if geometry_id and prompt:
        prompt_map[geometry_id].append(prompt)

dst.write_text(
    json.dumps(prompt_map, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"saved {dst}")
'@ | python -
```

## 單張 metrics

如果你只想評估某一張輸出圖：

```powershell
python compute_single_image_metrics.py `
  --image "results/manual/animal08_color05_artwork3/實際輸出檔名.png" `
  --geometry assets/geometry/animal08.jpeg `
  --color assets/color/color05.jpg `
  --texture assets/texture/artwork_3.jpg `
  --prompt "A single dog on the ground" `
  --output-dir results/single_image_metrics `
  --save-debug
```

主要指標可以這樣讀：

- `color_histogram_distance`: 越低越接近 color reference。
- `clip_geometry_sketch`: 越高代表 output canny 與 geometry canny 越接近。
- `clip_texture_gray`: 越高代表灰階材質/紋理越接近 texture reference。
- `geometry_improvement_ratio`: 相對 SDXL baseline 的幾何改善比例，越高越好。

## 串聯流程

完整實驗通常照這條線跑：

```text
1. 準備 assets
   color reference -> assets/color/
   geometry reference -> assets/geometry/
   texture reference -> assets/texture/

2. 準備 prompt map
   把 texture_geometry_color_prompts.json 轉成 SADis prompt-file 格式。

3. 先跑 smoke test
   用 --limit 2 確認模型路徑、CUDA、輸出資料夾都正常。

4. 跑 ablation / sweep
   base: run_dog_tuning_experiments.py
   equal attention: run_dog_tuning_experiments_equal_attention.py
   no orthogonal: run_dog_tuning_experiments_equal_attention_no_orthogonal.py

5. 看結果
   先看 results/.../images，再看 metrics.csv 與 summary.json。

6. 做 figure
   使用 make_*_grid_figure.py 或 make_teaser_disentanglement_figure.py 整理成論文圖。
```

