# SADis 與 SADisBase 方法差異分析

本文比較目前 `C:\Users\tianachen\SADis` 版本與 `C:\Users\tianachen\SADisbase` 版本的相同點與差異。這裡的 `SADisBase` 指本機 `SADisbase` 資料夾內的實作，不是只看 README 或原始論文描述。

核心結論：

```text
SADisBase 是 color-texture disentanglement baseline：
    color RGB token - color gray token + texture gray token -> 單一路 IP token

目前 SADis 是 tri-stream geometry-aware extension：
    color stream + geometry stream + texture stream -> 三路 IP token
    再透過 layer-aware / phase-aware attention routing 動態混合
```

簡單說，兩者同樣是 training-free、SDXL + IP-Adapter Plus XL + WCT 的風格化方法；最大的不同是目前 SADis 把 geometry 從「評估用 reference / prompt baseline」提升成「實際參與生成的第三路 image condition」。

## 1. 專案定位

| 項目 | SADisBase | 目前 SADis |
| --- | --- | --- |
| 方法核心 | color-texture disentanglement | color-geometry-texture tri-stream disentanglement |
| 主要 reference | color + texture | color + geometry + texture |
| geometry 用途 | 多半用於實驗分組、baseline prompt、metrics | 真正進入 IP-Adapter generation |
| IP token 形式 | 單一路 fused appearance token | 三路獨立 token：color / geometry / texture |
| attention routing | 原始 IP-Adapter style，選定 target block 後單路注入 | layer-aware + phase-aware 三路混合 |
| diffusion step 策略 | pipeline 有 schedule 參數痕跡，但 base attention processor 不吃這些值 | denoising loop 每步更新 stream phase 與 step factor |
| 主要問題意識 | color 與 texture 解耦 | color / geometry / texture 三者互相污染與主體重複 |

## 2. 相同之處

### 2.1 都是 training-free

兩者都不訓練新的 diffusion model 或 adapter。共同使用：

- `stabilityai/stable-diffusion-xl-base-1.0`
- `models/image_encoder`
- `sdxl_models/ip-adapter-plus_sdxl_vit-h.bin`
- CLIP Vision encoder
- IP-Adapter Plus XL Resampler
- 修改過的 SDXL pipeline
- WCT latent color guidance

兩者方法都屬於 inference-time manipulation：透過 reference preprocessing、CLIP token arithmetic、attention processor、latent WCT 來達到解耦控制。

### 2.2 都使用 CLIP Vision hidden states

兩邊的 `IPAdapterPlusXL.get_image_embeds()` 都使用：

```python
self.image_encoder(..., output_hidden_states=True).hidden_states[-2]
```

這代表兩者都不是只用 CLIP pooled image embedding，而是用 patch-level hidden states。這點讓 texture 與局部視覺資訊能保留得更多，也是 SADis 能做 color/texture transfer 的基礎。

### 2.3 都保留原始 SADis color-texture 公式精神

SADisBase 的核心公式是：

```text
F = color_scale * F_color
  - color_sub_scale * F_color_gray
  + texture_scale * F_texture_gray
```

目前 SADis 雖然已經改成三路 stream，但 color stream 仍保留同樣精神：

```text
color_stream = color_scale * color_token
             - substract_scale * color_gray_token
```

所以目前 SADis 不是完全推翻 SADisBase，而是把原本 fused appearance token 拆開，並把 geometry 加成第三路。

### 2.4 都有 texture gray punishment

兩者都會從 texture grayscale image 建立 pure-gray image，然後把 texture token 與 pure-gray token concat，再呼叫：

```python
punish_weight_module(...)
```

目的都是抑制 texture token 中太接近平坦灰階或低資訊的方向，讓 texture 更偏向材質與局部 pattern。

差別在於：

- SADisBase 的 texture punishment 直接作用在最終 fused token 裡的 texture component。
- 目前 SADis 的 texture punishment 作用在獨立 texture stream 上，之後才進三路 attention routing。

### 2.5 都有 WCT latent guidance

兩者的 `pipeline_stable_diffusion_xl.py` 都有 `wct_transform_with_zt_add_noise()`，大致流程相同：

```text
color ref -> VAE latent at timestep
current latent + color latent -> whiten_and_color()
add small timestep-scaled noise
blend back into denoising latent
```

用途也相同：補強 color reference 的全域 latent statistics。

主要差異是預設強度：

| 參數 | SADisBase | 目前 SADis |
| --- | ---: | ---: |
| `wct_guidance` | `0.5` | `0.28` |
| `wct_starts_step_ratio` | `0.2` | `0.45` |
| `wct_ends_step_ratio` | `0.3` | `0.55` |
| `wctnoise_add_scale` | `0.01` | `0.004` |

SADisBase 的 WCT 比較早、比較強；目前 SADis 把 WCT 放到中期且較弱，避免它干擾 geometry 建立與 texture 收尾。

## 3. 主要差異

### 3.1 Reference 數量與角色不同

SADisBase 入口 `infer_style_plus_color_texture.py` 只有兩張主要 reference：

```python
color_dir = r'assets/color/1.png'
texture_dir = r'assets/texture/001.jpg'
```

目前 SADis 入口有三張：

```python
INPUT_COLOR = r"assets/color/color05.jpg"
INPUT_GEOMETRY = r"assets/geometry/face05.webp"
INPUT_TEXTURE = r"assets/texture/artwork_3.jpg"
```

差異不是多一張圖而已，而是 reference 的語義被重新分工：

| Reference | SADisBase | 目前 SADis |
| --- | --- | --- |
| Color | palette / appearance | palette stream |
| Texture | texture + style | texture stream |
| Geometry | 不進主要 generation | geometry stream |

在 SADisBase 的 batch experiment 中，`AssetTriple` 也有 geometry 欄位，但 `generate_one()` 只把 color 和 texture 傳進 `ip_model.generate()`；geometry 用於 metrics、baseline prompt 或跨 geometry 評估，不是生成條件。

目前 SADis 的 `generate_one()` 會做：

```python
geometry_ref_img = preprocess_geometry_for_rules(...)
ip_model.generate(..., geometry_ref_img=geometry_ref_img, ...)
```

這是本質差異。

### 3.2 Token 組織方式不同

SADisBase 把 color、color-gray、texture-gray 先在 CLIP token space 合成一組 token：

```text
clip_image_embeds =
    color_scale * clr_clip_image_embeds
  - subscale * clr_txr_clip_image_embeds
  + texture_scale * txr_clip_image_embeds2
```

然後只投影成一組 IP tokens：

```text
image_prompt_embeds = Resampler(clip_image_embeds)
```

若 `num_tokens=16`，最終 IP tokens 是：

```text
16 tokens
```

目前 SADis 則是：

```text
color_p = Resampler(color_stream)
geo_p   = Resampler(geometry_stream)
tex_p   = Resampler(texture_stream)

image_prompt_embeds = concat([color_p, geo_p, tex_p])
```

若每路 `num_tokens=16`，最終 IP tokens 是：

```text
16 color tokens + 16 geometry tokens + 16 texture tokens = 48 tokens
```

這個差異導致後續 attention processor 必須完全不同。

### 3.3 解耦方式不同

SADisBase 的主要解耦是單一線性公式：

```text
color RGB - color gray + texture gray
```

也就是：

- 用 color gray 扣掉 color image 的 structure / luminance component。
- 用 texture gray 加入 texture pattern。
- 但 color、texture 最後會混成同一個 token stream。

目前 SADis 使用 projection-based multi-direction disentanglement：

```text
geometry 扣 color
texture 扣 geometry
geometry 扣 texture
color 扣 geometry
color 扣 texture
texture 扣 color
```

核心 helper：

```text
proj_res(A, B, scale) = A - scale * proj_B(A)
```

並且每次處理後做 energy restoration：

```text
processed = processed * ||original|| / ||processed||
```

這代表目前 SADis 的解耦目標更細：

- color stream 不攜帶 geometry / texture。
- geometry stream 不攜帶 color / texture。
- texture stream 不攜帶 color / geometry。

SADisBase 只處理 color-texture disentanglement；目前 SADis 處理的是三方 disentanglement。

### 3.4 Geometry 前處理是目前 SADis 新增的核心

SADisBase 沒有主 generation 用的 geometry preprocessing path。

目前 SADis 新增：

- `_augment_geometry_feature_rgb()`
- `_homogenize_geometry_color()`
- `_extract_geometry_canny_edges()`
- `_filter_small_edge_components()`
- `_filter_short_edge_components()`
- `_keep_major_edge_components()`
- `preprocess_geometry_for_rules()`

geometry preprocessing 的目的：

```text
保留輪廓、pose、layout
降低色彩、材質、背景碎線
讓 geometry ref 能作為 CLIP Vision image reference
```

這和 ControlNet 的 hard edge conditioning 不同。目前 SADis 是把 preprocessing 後的 geometry RGB image 丟進 CLIP Vision，再走 IP-Adapter attention。

### 3.5 Attention processor 不同

SADisBase 的 `IPAttnProcessor2_0` 是單一路 IP attention：

```text
encoder_hidden_states -> split [text, ip_hidden_states]
text attention -> hidden_states
ip attention over ip_hidden_states -> hidden_states += scale * ip_hidden_states
```

它只知道：

```text
num_tokens = 16
```

不理解 color / geometry / texture 三路。

目前 SADis 的 `IPAttnProcessor2_0` 則會：

```text
encoder_hidden_states -> split [text, all_ip_tokens]
all_ip_tokens -> split [clr_ip, geo_ip, tex_ip]
phase mix -> active_ip_hidden
ip attention over active_ip_hidden
hidden_states += current_ip_scale * ip_res
```

它知道：

```text
stream_token_sizes = [16, 16, 16]
total_ip_tokens = 48
```

並支援：

- `set_stream_step_scales()`
- `set_stream_phase()`
- `get_phase_mix_table()`
- per-layer `stream_scales`
- down-block geometry low-pass
- down-block geometry entropy gate

這是目前 SADis 相對 SADisBase 最大的架構改動。

### 3.6 Target blocks 不同

SADisBase 預設：

```python
target_blocks = ["up_blocks.0.attentions.1"]
```

這代表它主要把 IP-Adapter style signal 放在特定 up block，偏外觀渲染。

目前 SADis 預設：

```python
TARGET_BLOCKS = ["down_blocks", "mid_block", "up_blocks"]
```

也就是幾乎所有 cross-attention blocks 都會裝上 IP-Adapter processor，然後由 layer-aware / phase-aware routing 決定每個 block 實際使用哪一路 stream。

直覺差異：

```text
SADisBase: 選少數 style block，注入 fused style token
目前 SADis: 擴到 down/mid/up，全 U-Net 分層控制 color/geometry/texture
```

### 3.7 Denoising step schedule 的有效性不同

SADisBase 的 `pipeline_stable_diffusion_xl.py` 內有：

```python
geometry_step_factor = 1.0 if step_ratio <= geometry_early_end else geometry_late_factor
texture_step_factor = texture_early_factor if step_ratio < texture_late_start else 1.0
self.set_ip_stream_step_scales([1.0, geometry_step_factor, texture_step_factor])
```

但 SADisBase 的 `IPAttnProcessor2_0` 沒有 `set_stream_step_scales()`，也沒有三路 stream 概念。因此這條呼叫在 base IP attention processor 上基本不產生作用。它像是後續 tri-stream 版本的前置痕跡或未完成接口。

目前 SADis 則完整打通：

```python
self.set_ip_stream_step_scales([color_step_factor, geometry_step_factor, texture_step_factor])
self.set_ip_stream_phase(phase_name)
```

而目前 SADis 的 attention processor 真的有：

```python
def set_stream_step_scales(...)
def set_stream_phase(...)
```

所以 step schedule 在目前 SADis 中是實際生效的。

### 3.8 Phase-aware routing 是目前 SADis 新增

SADisBase 沒有 phase mix table。

目前 SADis 有：

```python
get_base_phase_mix_table(save_name)
get_phase_mix_table(save_name)
```

並依據 block 名稱與 denoising phase 回傳：

```text
(color_weight, geometry_weight, texture_weight)
```

策略是：

- down front：geometry 幾乎完全主導。
- mid：geometry 逐步降低，color/texture 進場。
- up back：geometry 歸零，texture/color 收尾。
- `up_blocks.0.attentions.1.attn2.processor`：全程只用 color + texture，不用 geometry。

SADisBase 則是單一路 IP token 全程由同一個 `scale` 加入，不會根據 phase 切換 stream。

### 3.9 CA mask 支援不同

SADisBase 的 attention processor 保留 `ca_mask`：

```python
if self.ca_mask:
    # use IP attention map to build mask
    # preserve foreground/material region
```

這可以用 attention map 做 foreground preservation，原本註解也說 material transfer 可設 `ca_mask=True`。

目前 SADis 的新版 `IPAttnProcessor2_0` 仍保留 `ca_mask` 參數入口，但主邏輯已聚焦在三路 stream routing，沒有沿用 SADisBase 那段 foreground CA mask replacement。換句話說：

```text
SADisBase: 有 foreground CA mask path
目前 SADis: 重點改成 tri-stream routing，CA mask 不是主要功能
```

這是目前 SADis 相對 base 可能少掉的一個功能。

### 3.10 Runtime robustness 不同

目前 SADis 比 SADisBase 多了許多 runtime guard：

- reference path 可用環境變數覆蓋。
- 支援關閉任一路 reference。
- 支援 pure SDXL mode。
- `_require_reference_file()` 檢查輸入。
- `ensure_supported_cuda_runtime()` 檢查 CUDA runtime。
- attention projection 有 non-finite sanitize。
- projection 發生 CUDA / cuBLAS 問題時可 fallback fp32。

SADisBase 比較像實驗腳本，直接假設路徑與 CUDA device 存在。

## 4. Pipeline 對照

### 4.1 SADisBase generation path

```text
color image RGB
    -> CLIP Vision hidden states -> F_color

color image gray
    -> CLIP Vision hidden states -> F_color_gray

texture image gray
    -> CLIP Vision hidden states -> F_texture_gray
    -> optional gray punishment

F = color_scale * F_color
  - substract_scale * F_color_gray
  + texture_scale * F_texture_gray

F -> Resampler -> 16 IP tokens
text tokens + 16 IP tokens
    -> selected IP attention blocks
    -> SDXL denoising
    -> optional WCT
```

### 4.2 目前 SADis generation path

```text
color image RGB
    -> CLIP Vision hidden states -> color token
color image gray
    -> CLIP Vision hidden states -> color-gray token

geometry image
    -> geometry preprocessing
    -> CLIP Vision hidden states -> geometry token

texture image gray
    -> CLIP Vision hidden states -> texture token
    -> optional gray punishment

color / geometry / texture tokens
    -> projection-based mutual decoupling
    -> energy restoration

color stream    -> Resampler -> 16 color IP tokens
geometry stream -> Resampler -> 16 geometry IP tokens
texture stream  -> Resampler -> 16 texture IP tokens

text tokens + 48 IP tokens
    -> split into [color, geometry, texture]
    -> layer-aware + phase-aware active IP token
    -> SDXL denoising with per-step stream schedule
    -> optional WCT
```

## 5. 實驗與 Metrics 差異

### 5.1 SADisBase 實驗

`run_baseline_tuning_experiments.py` 使用 `AssetTriple`：

```python
AssetTriple(key, geometry, color, texture)
```

但 generation 只傳：

```python
clr_ref_img=color_ref_img
clr_texture_ref_img=color_image_gray
texture_ref_img=texture_image_gray
```

geometry 主要用於：

- geometry baseline prompt。
- canny / sketch metric。
- cross-geometry diversity。
- geometry residual metrics。

因此 SADisBase 的 geometry metrics 更像在問：

```text
在沒有直接 geometry conditioning 的情況下，
output 是否跟某個 geometry target 有相似輪廓？
```

### 5.2 目前 SADis 實驗

`run_dog_tuning_experiments.py` 也使用 geometry/color/texture triple，但 generation 真的傳入：

```python
geometry_ref_img=preprocess_geometry_for_rules(str(triple.geometry))
```

所以 metrics 問的是：

```text
在 geometry stream 真的參與生成時，
output 是否更貼近 geometry reference？
```

這讓目前 SADis 的 geometry metric 更直接、更有解釋力。

### 5.3 Metrics 內容擴充

兩者都有：

- CLIP text similarity
- CLIP image-image similarity
- color histogram distance
- Canny / sketch geometry comparison
- geometry residual metrics

目前 SADis 版本更進一步加入或整理：

- geometry category handling
- image cache
- MS-SSIM geometry
- edge recall geometry
- LPIPS memory handling / skip options
- final output brightness / saturation / contrast control
- richer negative prompt for duplicate subject 與 desaturation

## 6. Failure Mode 角度的差異

### 6.1 SADisBase 的主要 failure modes

SADisBase 常見問題比較像：

- color 和 texture 沒有完全解耦。
- texture 太弱或太灰。
- texture reference content 被帶入。
- color reference 的 structure 沒被扣乾淨。
- 只有 style block 注入時，layout 控制有限。

因為 SADisBase 沒有真正 geometry condition，所以它很難精準控制 pose / silhouette / composition。

### 6.2 目前 SADis 的主要 failure modes

目前 SADis 新增 geometry 後，控制能力更強，但也多出新問題：

- geometry semantic leakage：geometry reference 的物件身份被帶入。
- duplicate subject：geometry 太早/太強，可能鎖出第二主體。
- texture 被過度解耦：projection subtraction + punishment 疊加後 texture 變弱。
- texture content leakage：texture 太早進場或 geometry subtraction 不夠，texture subject 被畫出來。
- block/phase 參數耦合：問題不一定能靠單一 `texture_scale` 或 `geometry_scale` 解決。

也就是說，目前 SADis 用更複雜的 routing 換取更高控制力，但調參空間也變大。

## 7. 方法演化關係

可以把兩者看成以下演化：

```text
SADisBase
  |
  | 1. 原始 color-texture disentanglement:
  |    color - gray_color + gray_texture
  |
  | 2. 保留 WCT latent color guidance
  |
  v
目前 SADis
  |
  | 3. 新增 geometry reference preprocessing
  | 4. 將 fused appearance token 拆成三路 stream
  | 5. 加入 projection-based mutual disentanglement
  | 6. 將 IP attention 改成三路 token split
  | 7. 加入 layer-aware stream scales
  | 8. 加入 phase-aware attention routing
  | 9. 在 denoising loop 每步更新 stream phase / step factor
  |
  v
Tri-Stream SADis
```

因此目前 SADis 不是「換一組參數的 SADisBase」，而是架構上已經從 fused-token baseline 變成 multi-stream routing method。

## 8. 哪些 base 設計被保留

目前 SADis 保留了 SADisBase 的幾個重要設計：

| Base 設計 | 是否保留 | 備註 |
| --- | --- | --- |
| SDXL base | 保留 | 同樣使用 SDXL base 1.0 |
| IP-Adapter Plus XL | 保留 | 仍使用同一套權重與 Resampler |
| CLIP hidden_states[-2] | 保留 | patch-level image tokens |
| color RGB - color gray | 部分保留 | 變成 color stream 內部公式 |
| texture grayscale | 保留 | 目前 texture still gray |
| pure gray punishment | 保留 | 但作用在獨立 texture stream |
| WCT latent guidance | 保留 | 強度與 window 改得更保守 |
| target block idea | 保留但擴展 | 從單一 style block 擴到 down/mid/up |
| ControlNet stylization script | 保留 | 但主方法重點轉向 tri-stream IP routing |

## 9. 哪些 base 設計被替換

| Base 設計 | 目前 SADis 替換為 |
| --- | --- |
| 單一 fused appearance token | color / geometry / texture 三路 token |
| `num_tokens=16` 單路 IP token | `16 * 3 = 48` 三路 IP token |
| 單一路 IP attention | phase-aware active stream attention |
| 只選 `up_blocks.0.attentions.1` | down/mid/up 全域 target blocks |
| 無 geometry generation condition | geometry preprocess + geometry stream |
| 單純 color-texture subtraction | 多方向 projection residual |
| 固定 IP scale | layer base scale × phase mix × step factor |
| base CA mask material preservation | tri-stream routing 為主 |

## 10. 最重要的實作對照

| 功能 | SADisBase | 目前 SADis |
| --- | --- | --- |
| 推論入口 | `SADisbase/infer_style_plus_color_texture.py` | `SADis/infer_style_plus_color_texture.py` |
| IP-Adapter class | `SADisbase/ip_adapter/ip_adapter.py::IPAdapterPlusXL` | `SADis/ip_adapter/ip_adapter.py::IPAdapterPlusXL` |
| image token 建立 | fused color-texture token | tri-stream token |
| attention processor | single IP token processor | multi-stream IP token processor |
| phase mix | 無 | `get_phase_mix_table()` |
| per-layer scale | 無 | `get_layer_stream_scales()` |
| step routing | pipeline 有 no-op-like hook | pipeline + processor 完整生效 |
| geometry preprocess | 無主 generation path | `preprocess_geometry_for_rules()` |
| batch experiment | `run_baseline_tuning_experiments.py` | `run_dog_tuning_experiments.py` |

## 11. 一句話比較

SADisBase：

```text
用 color reference 與 texture reference 在 CLIP token space 做線性組合，
得到單一路 style IP token，再透過少數 attention blocks 注入 SDXL。
```

目前 SADis：

```text
把 color、geometry、texture 分成三路 reference stream，
先在 CLIP token space 互相扣除污染方向，再於 SDXL 每個 denoising phase
依照 U-Net block 角色動態選擇三路 stream 的混合比例。
```

更精簡地說：

```text
SADisBase = fused style token
目前 SADis = disentangled tri-stream routing
```

## 12. 方法貢獻角度

若要在報告中說明「我的方法相對 SADisBase 的貢獻」，可以寫成：

1. **Geometry-aware extension**：新增 geometry reference stream，使模型能直接受 pose / silhouette / layout 影響，而不是只用 prompt 或 evaluation geometry。
2. **Tri-stream tokenization**：將原始 SADis 的 fused appearance token 拆為 color、geometry、texture 三組 IP tokens。
3. **Projection-based mutual disentanglement**：用 projection residual 在 CLIP Vision token space 中降低 stream 間污染。
4. **Layer-aware stream scaling**：根據 down/mid/up blocks 的功能差異，給 geometry、color、texture 不同基礎權重。
5. **Phase-aware attention routing**：根據 denoising front/mid/back phase 動態切換三路 stream 的使用比例。
6. **Safer late-stage rendering**：後期關閉或降低 geometry，讓 color/texture 負責外觀與細節，降低 geometry 殘影和重複主體。
7. **More direct geometry evaluation**：generation 中真的使用 geometry condition，因此 geometry metrics 的解釋更直接。

## 13. 需要誠實說明的限制

目前 SADis 相對 SADisBase 不是所有方面都只有優點，也有代價：

- 參數更多，調參空間更大。
- geometry stream 可能帶來 subject leakage。
- 強 geometry 容易造成 duplicate subject。
- texture 經過多重 decouple 後可能變弱。
- base 的 CA mask foreground preservation 沒有成為目前主路徑。
- 三路 48 tokens 比 base 16 tokens 更吃 attention 計算與顯存。

因此可以把目前方法描述成：

```text
更可控，但更需要 block/phase-aware 調參。
```

## 14. 總結

SADisBase 和目前 SADis 的共同基礎是 SADis 的 color-texture disentanglement、IP-Adapter Plus XL、CLIP Vision patch tokens、WCT latent color guidance。

最大差異是：

```text
SADisBase 把 color 與 texture 混成單一路 appearance token；
目前 SADis 把 color、geometry、texture 拆成三路 token，並在 attention block 與 denoising step 上動態路由。
```

如果用一句話概括：

```text
SADisBase 解的是「顏色和材質如何分開」；
目前 SADis 解的是「顏色、形狀、材質如何分開，並且在正確的 U-Net block 與 diffusion step 發揮作用」。
```
