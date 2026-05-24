# SADis 方法細節分析

本文整理目前 `SADis` 專案中實際採用的方法。重點不是只描述原始論文版本，而是以目前程式碼為準，分析你現在這版「color / geometry / texture 三路解耦」方法的完整資料流、token 流、attention routing、denoising schedule、WCT guidance、實驗與評估設計。

核心實作位置：

- 入口推論：`infer_style_plus_color_texture.py`
- 三路 IP-Adapter：`ip_adapter/ip_adapter.py`
- 三路 attention processor：`ip_adapter/attention_processor.py`
- 修改過的 SDXL pipeline：`pipeline_stable_diffusion_xl.py`
- 批次實驗與 metrics：`run_dog_tuning_experiments.py`
- WCT latent transform：`util/wct.py`
- ControlNet 版本入口：`infer_style_controlnet_color_texture.py`

## 1. 方法總覽

目前方法可以視為在原始 SADis color-texture disentanglement 上，加上一個 geometry stream，形成三路 reference control：

| Stream | 來源 | 目標 | 主要控制內容 |
| --- | --- | --- | --- |
| Color stream | color reference RGB | 色彩、palette、整體氛圍 | 顏色分布、色調、飽和度 |
| Geometry stream | geometry reference preprocess 後的 RGB | 結構、輪廓、姿勢、構圖 | 主體形狀、邊界、pose、layout |
| Texture stream | texture reference grayscale | 表面材質、筆觸、明暗紋理 | 毛流、pattern、局部 texture |

整體流程如下：

```text
color ref       -> CLIP Vision hidden states -> color token purification   -> Resampler -> color IP tokens
geometry ref    -> geometry preprocess       -> CLIP Vision hidden states -> geometry token purification -> Resampler -> geometry IP tokens
texture ref     -> grayscale / optional high-pass -> CLIP Vision hidden states -> texture token purification -> Resampler -> texture IP tokens

text prompt -> SDXL text encoders -> text tokens

[text tokens, color IP tokens, geometry IP tokens, texture IP tokens]
    -> modified cross-attention processor
    -> phase-aware / layer-aware stream mixing
    -> SDXL denoising loop
    -> optional WCT latent color guidance
    -> final image enhancement
```

方法的主要設計哲學是：

1. 不訓練新模型，沿用 SDXL base 與 IP-Adapter Plus XL 權重。
2. 把 reference image 的作用拆成三種功能，不讓同一張 style image 同時控制顏色、紋理、構圖。
3. 在 CLIP Vision token space 做 projection subtraction，降低 stream 之間的互相污染。
4. 在 cross-attention 中依照 U-Net layer 與 denoising phase 動態混合三路 token。
5. 讓 geometry 早期強、後期退場；讓 color 穩定全程存在；讓 texture 在後期補表面細節。

## 2. 入口推論腳本

主要入口是 `infer_style_plus_color_texture.py`。

### 2.1 Reference 輸入

目前預設三張 reference：

```python
INPUT_COLOR = r"assets/color/color05.jpg"
INPUT_GEOMETRY = r"assets/geometry/face05.webp"
INPUT_TEXTURE = r"assets/texture/artwork_3.jpg"
```

也支援環境變數覆蓋：

| 環境變數 | 功能 |
| --- | --- |
| `SADIS_INPUT_COLOR` | 覆蓋 color reference；設成 `None` 可關閉 |
| `SADIS_INPUT_GEOMETRY` | 覆蓋 geometry reference；設成 `None` 可關閉 |
| `SADIS_INPUT_TEXTURE` | 覆蓋 texture reference；設成 `None` 可關閉 |
| `SADIS_SAVE_DIR` | 覆蓋輸出資料夾 |
| `SADIS_GEOMETRY_EDGE_MIN_COMPONENT_LENGTH` | 覆蓋 geometry edge component 長度過濾門檻 |

如果三張 reference 全部關閉，程式會進入 pure SDXL mode，不使用 IP-Adapter image condition。

### 2.2 主要推論參數

目前入口腳本的主要預設值：

| 參數 | 目前值 | 意義 |
| --- | ---: | --- |
| `steps` | `50` | diffusion steps |
| `seed` | `42` | 隨機種子 |
| `guidance_scale` | `9.0` | CFG guidance |
| `scale` | `1.0` | IP-Adapter 全域 scale |
| `color_scale` | `1.6` | color stream 強度 |
| `substract_scale` | `1.0` | 從 color RGB token 扣除 color-gray token 的強度 |
| `texture_scale` | `1.4` | texture stream 強度 |
| `geometry_scale` | `1.4` | geometry stream 強度 |
| `geometry_sub_scale` | `0.80` | texture/geometry 互扣預設強度 |
| `texture_color_decouple` | `0.10` | texture 扣 color 方向的強度 |
| `geometry_color_decouple` | `0.35` | geometry 扣 color 方向的強度 |
| `color_to_geometry_decouple` | `0.25` | color 扣 geometry 方向的強度 |
| `punish_weight` | `0.0003` | texture gray punishment 強度 |
| `punish_type` | `soft-weight` | punishment 方法 |

目前 schedule 設定：

| 參數 | 目前值 | 意義 |
| --- | --- | --- |
| `front_end_ratio` | `0.40` | 前期 phase 結束比例 |
| `mid_end_ratio` | `0.70` | 中期 phase 結束比例 |
| `color_stage_factors` | `(1.0, 1.0, 1.0)` | color 全程穩定 |
| `geometry_stage_factors` | `(1.0, 1.0, 0.0)` | geometry 後期退場 |
| `texture_stage_factors` | `(1.0, 1.0, 1.25)` | texture 後期增強 |

WCT 設定：

| 參數 | 目前值 | 意義 |
| --- | ---: | --- |
| `wct_guidance` | `0.28` | WCT latent guidance 混合比例 |
| `wct_starts_step_ratio` | `0.45` | WCT 開始 step 比例 |
| `wct_ends_step_ratio` | `0.55` | WCT 結束 step 比例 |
| `wctnoise_add_scale` | `0.004` | WCT 後加回噪聲的比例 |

### 2.3 Negative prompt 設計

`DEFAULT_NEGATIVE_PROMPT` 針對以下 failure modes：

- 文字、水印、logo、低品質。
- reference copy：避免直接複製 reference image 的 composition 或物件。
- duplicate subject：避免 multiple children、extra person、twins、crowd。
- 不想要的建築語義：church、cathedral、chapel、mosque。
- 色彩退化：dull、desaturated、greyish、monochromatic、flat lighting。

這個 negative prompt 很重要，因為三路 reference 尤其 texture / geometry 仍可能攜帶 subject semantics。若 prompt 沒有單主體約束，模型容易把 reference 的 subject 或 composition 畫進輸出。

## 3. Geometry Reference 前處理

geometry 前處理在 `preprocess_geometry_for_rules()` 中實作，入口與 batch script 有近似版本。

### 3.1 前處理目標

geometry stream 希望提供「形狀與輪廓」，但不要提供太多顏色與材質。因此前處理做了：

1. 色彩 homogenize：降低 chroma 差異，讓 geometry 圖不要帶入原圖色彩。
2. luma normalization：把亮度推到固定 mean 附近。
3. 邊緣抽取：用 Canny 取出輪廓。
4. component filtering：去掉小碎邊、短邊、保留主要邊緣。
5. morphology：close 與 dilate，讓輪廓更連續。
6. edge fusion：把邊緣 mask 融回 homogenized RGB。

### 3.2 Geometry feature augmentation

`_augment_geometry_feature_rgb()` 先對 geometry RGB 做輕微 sharpening 與 deterministic noise：

```text
rgb -> Gaussian blur -> unsharp mask -> add small Gaussian noise
```

目前參數：

| 參數 | 值 |
| --- | ---: |
| `GEOMETRY_FEATURE_SHARPEN_AMOUNT` | `0.28` |
| `GEOMETRY_FEATURE_SHARPEN_SIGMA` | `1.10` |
| `GEOMETRY_FEATURE_NOISE_STD` | `2.5` |

noise seed 由 geometry path 的 CRC32 決定，所以同一路徑會得到穩定結果。這可以讓 CLIP Vision token 對邊界與局部差異更敏感，但不引入每次不同的隨機擾動。

### 3.3 Canny edge extraction

`_extract_geometry_canny_edges()` 流程：

```text
RGB -> grayscale -> CLAHE -> bilateral filter -> adaptive Canny threshold
```

Canny threshold 不是固定值，而是根據 blurred gray median：

```text
lower = (1 - canny_sigma) * median
upper = (1 + canny_sigma) * median
```

目前 `canny_sigma` 預設 `0.16` 或 `0.18`，代表邊緣門檻相對貼近影像自身對比。

### 3.4 Edge component filtering

前處理用三層規則清理邊緣：

| 函式 | 作用 |
| --- | --- |
| `_filter_small_edge_components()` | 移除 area 太小的連通元件 |
| `_filter_short_edge_components()` | 移除 width/height 都太短的元件 |
| `_keep_major_edge_components()` | 依 `area * max(width, height)` 保留 top components |

目前重要參數：

| 參數 | 值 |
| --- | ---: |
| `GEOMETRY_EDGE_MIN_COMPONENT_AREA` | `64` |
| `GEOMETRY_EDGE_MIN_COMPONENT_LENGTH` | `36` |
| `GEOMETRY_EDGE_KEEP_TOP_COMPONENTS` | `80` |
| `GEOMETRY_EDGE_CLOSE_KERNEL_SIZE` | `3` |
| `GEOMETRY_EDGE_CLOSE_ITERATIONS` | `1` |
| `GEOMETRY_EDGE_DILATE_ITERATIONS` | `1` |

這組設計會讓 geometry stream 更偏向主要輪廓，降低背景碎線對主體生成的干擾。

### 3.5 Geometry color homogenization

`_homogenize_geometry_color()` 進入 LAB space：

1. L channel 做 contrast 壓縮與 mean shift。
2. AB chroma 往平均 chroma 收縮。
3. 再轉回 RGB。

目前參數：

| 參數 | 值 | 效果 |
| --- | ---: | --- |
| `GEOMETRY_LUMA_TARGET_MEAN` | `175.0` | 讓 geometry 參考偏亮 |
| `GEOMETRY_LUMA_MEAN_STRENGTH` | `0.60` | mean shift 強度 |
| `GEOMETRY_LUMA_CONTRAST` | `0.65` | 降低亮度對比 |
| `GEOMETRY_CHROMA_STRENGTH` | `0.25` | 大幅降低色彩差異 |

這裡的重點是：geometry image 不是直接當 ControlNet edge map，而是當 CLIP Vision reference。它仍需要 RGB image 型態，但內容被設計成「弱色彩、強輪廓」。

## 4. 三路 CLIP Vision Token

核心在 `IPAdapterPlusXL.get_image_embeds()`。

### 4.1 使用 hidden states 而非 pooled image embedding

這版使用：

```python
self.image_encoder(..., output_hidden_states=True).hidden_states[-2]
```

也就是 CLIP Vision 倒數第二層 patch-level hidden states，而不是單一 pooled image embedding。這對三路控制很重要：

- patch tokens 保留較多 spatial / local pattern。
- texture 可以保留局部紋路。
- geometry 可以保留形狀與位置線索。
- color 可以保留粗略色彩分布與區域色彩。

接著每一路 token 會進同一個 IP-Adapter Plus XL `Resampler`，輸出固定數量的 prompt tokens。

### 4.2 三路 stream token 數

`IPAdapterPlusXL.__init__()` 設定：

```python
self.ip_stream_token_sizes = [self.num_tokens, self.num_tokens, self.num_tokens]
```

入口使用 `num_tokens=16`，因此：

```text
color tokens    = 16
geometry tokens = 16
texture tokens  = 16
total IP tokens = 48
```

拼接順序固定為：

```text
[color, geometry, texture]
```

attention processor 會依照這個順序把 encoder hidden states 最後 48 個 token 切開。

## 5. Token-Level 解耦

三路 token 解耦是目前方法最重要的部分。它使用 projection subtraction：

```text
proj_B(A) = ((A dot B) / ||B||^2) * B
A_res = A - scale * proj_B(A)
```

程式中 helper 是：

```python
proj_res(A, B, scale)
```

### 5.1 解耦參數關係

`get_image_embeds()` 支援多個 decouple 方向：

| 參數 | 預設來源 | 意義 |
| --- | --- | --- |
| `texture_geometry_decouple` | `geometry_sub_scale` | texture/geometry 互扣共用預設 |
| `texture_to_geometry_decouple` | `texture_geometry_decouple` | texture 扣 geometry 方向 |
| `geometry_to_texture_decouple` | `texture_geometry_decouple` | geometry 扣 texture 方向 |
| `color_to_geometry_decouple` | `geometry_color_decouple` | color 扣 geometry 方向 |
| `color_to_texture_decouple` | `texture_color_decouple` | color 扣 texture 方向 |
| `texture_to_color_decouple` | `texture_color_decouple` | texture 扣 color 方向 |

入口目前只顯式傳入：

```text
geometry_sub_scale
geometry_color_decouple
texture_color_decouple
color_to_geometry_decouple
```

其他方向會使用預設 fallback。

### 5.2 實際解耦順序

目前解耦流程：

```text
c_ori = color CLIP hidden states
t_ori = texture CLIP hidden states
g_ori = geometry CLIP hidden states

g_pre = g_ori - geometry_color_decouple * proj_color(g_ori)

t_after_g = t_ori - texture_to_geometry_decouple * proj_g_pre(t_ori)
g_final = g_pre - geometry_to_texture_decouple * proj_t_ori(g_pre)
color_after_g = c_ori - color_to_geometry_decouple * proj_geometry(c_ori)

c_final = color_after_g - color_to_texture_decouple * proj_t_after_g(color_after_g)
t_final = t_after_g - texture_to_color_decouple * proj_c_ori(t_after_g)
```

直覺解讀：

1. Geometry 先扣 color，避免 geometry reference 的顏色污染 shape stream。
2. Texture 與 geometry 互扣，避免 texture reference 的構圖/subject 藉 texture stream 洩漏，也避免 geometry 帶太多 texture pattern。
3. Color 也扣 geometry，降低 color reference 中的 layout 影響。
4. Color 與 texture 互扣，讓 color stream 更像 palette，texture stream 更像 pattern。

### 5.3 能量補償

每次 projection subtraction 可能讓 token norm 變小，導致 stream 被削弱。程式用 `restore(original, processed)` 做能量補償：

```text
processed = processed * (||original|| / ||processed||)
```

這代表解耦主要改變方向，不希望大幅改變 token energy。這是避免 texture 或 geometry collapse 的關鍵。

### 5.4 Texture gray punishment

texture stream 會再經過 gray punishment。流程：

1. 從 texture grayscale image 建立一張純灰圖，其像素值等於原 texture mean。
2. 將 texture token 與 pure-gray token concat。
3. 呼叫 `punish_weight_module()`，在 token/channel 空間做 soft-weight 或其他方法。
4. 只取回 texture token 的部分。

目的：

- 壓制 texture 中接近平坦灰階/低資訊的方向。
- 讓 texture stream 更集中在真正的材質變化。

但實驗筆記也指出，如果 `punish_weight` 太高，texture 會被削弱到只剩弱明暗，導致材質不明顯。

## 6. Stream 組裝方式

### 6.1 Color stream

目前 color stream：

```text
color_stream = c_final * color_scale - color_gray_token * substract_scale
```

這保留原始 SADis 的 color disentanglement 思路：

- `c_final * color_scale` 提供 color reference。
- `- color_gray_token * substract_scale` 扣掉 color image 的灰階結構，使 color stream 更偏 palette。

目前已移除舊版中把 texture token 加進 appearance stream 的設計。這點很重要：color stream 不再混入 texture stream。

### 6.2 Geometry stream

```text
geometry_stream = g_final * geometry_scale
```

若沒有 geometry reference，則使用 unconditional IP token 佔位。這樣總 token 數保持固定，attention processor 的切 token 邏輯不會壞。

### 6.3 Texture stream

```text
texture_stream = t_final * texture_scale
```

如果提供 `texture_ref_img2` 且 `texture2_scale != 0`，會另外做 high-pass texture：

```text
texture2_rgb - GaussianBlur(texture2_rgb) + 127.5
```

再送入 CLIP Vision，加入 texture stream。這條路是為了強化筆觸或高頻材質，但目前入口預設 `texture2_scale = 0.0`。

### 6.4 Resampler 與 token concat

三路 stream 各自進同一個 `image_proj_model`：

```text
color_p = Resampler(color_stream)
geo_p   = Resampler(geometry_stream)
tex_p   = Resampler(texture_stream)

image_prompt_embeds = concat([color_p, geo_p, tex_p], dim=token)
```

unconditional branch 則是：

```text
uncond_image_prompt_embeds = concat([uncond_p, uncond_p, uncond_p])
```

最後 `generate()` 會把 text prompt embedding 與 image prompt embedding 串接：

```text
prompt_embeds = concat([text_prompt_embeds, image_prompt_embeds], dim=token)
negative_prompt_embeds = concat([negative_text_embeds, uncond_image_prompt_embeds], dim=token)
```

## 7. Layer-Aware Stream Scaling

`IPAdapterPlusXL.get_layer_stream_scales()` 依照 U-Net layer 給三路不同 base scale：

| Layer | Color | Geometry | Texture | 解讀 |
| --- | ---: | ---: | ---: | --- |
| `down_blocks` | `0.80` | `1.50` | `0.18` | 早期結構鎖定，geometry 最強 |
| `mid_block` | `0.95` | `1.05` | `0.95` | 全域語義/構圖/材質較均衡 |
| `up_blocks` | `1.10` | `0.65` | `1.00` | 後段外觀與細節，geometry 降低 |

這個設計反映 diffusion U-Net 的角色分工：

- Down blocks 對粗結構與 layout 影響大，所以 geometry 權重大。
- Mid block 是全域語義與整體組合區，所以三者接近。
- Up blocks 負責細節重建，所以 color/texture 權重較高、geometry 退居輔助。

## 8. Phase-Aware Attention Mixing

核心在 `ip_adapter/attention_processor.py` 的 `IPAttnProcessor2_0`。

### 8.1 三路 token 切分

attention processor 從 encoder hidden states 尾端取出 IP tokens：

```text
text_embeds = encoder_hidden_states[:, :end_pos, :]
all_ip_embeds = encoder_hidden_states[:, end_pos:, :]

clr_ip = all_ip_embeds[:, 0:16, :]
geo_ip = all_ip_embeds[:, 16:32, :]
tex_ip = all_ip_embeds[:, 32:48, :]
```

text attention 先照原本 SDXL cross-attention 計算。IP attention 是額外一路，再加回 hidden states：

```text
hidden_states = text_attention_output + current_ip_scale * ip_attention_output
```

### 8.2 Phase 定義

pipeline 每一步依 step ratio 設定 phase：

```text
front: step_ratio < front_end_ratio
mid:   front_end_ratio <= step_ratio < mid_end_ratio
back:  step_ratio >= mid_end_ratio
```

目前：

```text
front = 0.00 ~ 0.40
mid   = 0.40 ~ 0.70
back  = 0.70 ~ 1.00
```

### 8.3 Phase mix table

`get_phase_mix_table(save_name)` 依 layer 名稱回傳 `(color, geometry, texture)` 權重。

Down blocks：

| Phase | Color | Geometry | Texture |
| --- | ---: | ---: | ---: |
| front | `0.00` | `1.00` | `0.00` |
| mid | `0.10` | `0.80` | `0.10` |
| back | `0.00` | `0.50` | `0.50` |

Mid block：

| Phase | Color | Geometry | Texture |
| --- | ---: | ---: | ---: |
| front | `0.10` | `0.80` | `0.10` |
| mid | `0.30` | `0.50` | `0.20` |
| back | `0.50` | `0.00` | `0.50` |

Up blocks 0：

| Phase | Color | Geometry | Texture |
| --- | ---: | ---: | ---: |
| front | `0.35` | `0.40` | `0.25` |
| mid | `0.40` | `0.20` | `0.40` |
| back | `0.50` | `0.00` | `0.50` |

Other up blocks：

| Phase | Color | Geometry | Texture |
| --- | ---: | ---: | ---: |
| front | `0.30` | `0.40` | `0.30` |
| mid | `0.40` | `0.30` | `0.30` |
| back | `0.40` | `0.00` | `0.60` |

另外有特殊 override：

```text
up_blocks.0.attentions.1.attn2.processor:
front/mid/back = (0.50, 0.00, 0.50)
```

這代表在這個「黃金渲染層」完全移除 geometry，只保留 color + texture。目的應該是避免 geometry 在細節渲染期造成格線、重複主體或構圖殘影。

### 8.4 有效 IP scale

每一步的 IP scale 同時受三個因素影響：

```text
current_ip_scale = global_ip_scale * effective_base * effective_step
```

其中：

```text
effective_base =
    color_weight * color_layer_base
  + geometry_weight * geometry_layer_base
  + texture_weight * texture_layer_base

effective_step =
    color_weight * color_step_factor
  + geometry_weight * geometry_step_factor
  + texture_weight * texture_step_factor
```

所以一個 stream 的真正影響力不是單看 `color_scale` 或 `texture_scale`，而是由以下因素共同決定：

1. token 本身在 `get_image_embeds()` 中的 scale。
2. layer base stream scale。
3. phase mix table。
4. denoising step factor。
5. IP-Adapter global scale。

這也解釋為什麼調參時只改 `texture_scale` 不一定有效。如果 phase mix 或 step factor 不給 texture 空間，texture token 再大也可能無法在正確 layer 發揮。

### 8.5 Down block geometry low-pass

attention processor 在 down blocks 對 geometry token 做 `_token_lowpass_3x3()`：

```text
geo_source = lowpass(geo_ip) if in down_blocks else geo_ip
```

這會把 geometry patch token 視為 square grid，做 3x3 low-pass。直覺上是讓早期 structural control 更平滑，降低碎邊或高頻 geometry noise 對主體結構的破壞。

### 8.6 Geometry confidence gate

在 down blocks，若 geometry 正在作用，會額外計算 geometry attention entropy：

1. 用 query 對 geometry-only key 算 attention。
2. 計算 entropy。
3. entropy 越低代表 geometry attention 越集中。
4. gate 在 `0.85 ~ 1.15` 間調整。

```text
geom_gate = clamp(1.25 - entropy_norm, 0.85, 1.15)
```

這是輔助性 gating，若數值不穩會 fallback 到 1.0。目的應該是讓 geometry control 在結構層根據 confidence 微調強度。

## 9. Denoising Loop 中的 Stream Schedule

`pipeline_stable_diffusion_xl.py` 的 denoising loop 每一步會呼叫：

```python
self.set_ip_stream_step_scales([color_step_factor, geometry_step_factor, texture_step_factor])
self.set_ip_stream_phase(phase_name)
```

目前入口設定使每一步 phase factor 為：

| Phase | Color | Geometry | Texture |
| --- | ---: | ---: | ---: |
| front | `1.0` | `1.0` | `1.0` |
| mid | `1.0` | `1.0` | `1.0` |
| back | `1.0` | `0.0` | `1.25` |

因此在後期：

- geometry step factor 變成 0。
- texture step factor 提升到 1.25。
- color 保持 1.0。

這是目前方法對「後期不讓 geometry 破壞 texture/color 收尾」的主要保護。

### 9.1 Attention block 與 step 的整體策略

目前 attention block × step 的策略可以理解成兩層控制：

1. **Block-level role assignment**：不同 U-Net block 負責不同生成層級。
2. **Step-level temporal routing**：同一個 block 在不同 denoising phase 對三路 stream 給不同權重。

也就是說，方法不是只設定「geometry 比較強」或「texture 比較強」，而是回答兩個更細的問題：

```text
在哪些 block 讓哪個 stream 強？
在哪些 step 讓哪個 stream 接手？
```

目前策略的核心假設是：

```text
early steps + down blocks  -> 決定大形、姿勢、layout
middle steps + mid block   -> 協調語義、全域構圖、色彩氛圍
late steps + up blocks     -> 補顏色、材質、筆觸、局部細節
```

因此 geometry 不應該全程都強；texture 也不應該太早全面介入。geometry 太晚還很強，容易造成輪廓殘影、重複主體或硬邊；texture 太早太強，容易把 texture reference 的 subject content 或 composition 帶進輸出。

### 9.2 Down blocks 策略：structure lock

Down blocks 是最偏結構的區域。這裡 latent resolution 較低，但對後續構圖影響大，因此目前設計讓 geometry 在 down blocks 前期最強。

目前 down blocks 的 layer base scale：

```text
color=0.80, geometry=1.50, texture=0.18
```

phase mix：

| Phase | Color | Geometry | Texture | 策略 |
| --- | ---: | ---: | ---: | --- |
| front | `0.00` | `1.00` | `0.00` | 完全鎖 geometry，先決定大形 |
| mid | `0.10` | `0.80` | `0.10` | 仍以 geometry 為主，少量 color/texture 進入 |
| back | `0.00` | `0.50` | `0.50` | geometry 與 texture 平分，讓表面開始貼合輪廓 |

這裡的設計意義：

- front phase 不讓 color/texture 干擾主體位置。
- geometry base scale `1.50` 讓 down blocks 對 geometry reference 很敏感。
- texture base scale `0.18` 很低，避免 texture reference 在早期變成 layout source。
- down blocks 內還有 geometry token low-pass，讓結構 token 更平滑。

如果出現「構圖不跟 geometry」：

- 優先提高 `geometry_scale` 或 down blocks geometry base。
- 延長 front phase，例如提高 `front_end_ratio`。
- 不要先提高 texture，因為 texture 會帶來 content leakage。

如果出現「雙主體、雙頭、重複輪廓」：

- 優先降低 down blocks geometry base 或 `geometry_scale`。
- 縮短 geometry 強勢時間，例如降低 `front_end_ratio`。
- 讓 down blocks back phase 的 geometry 從 `0.50` 再降一些。

### 9.3 Mid block 策略：global negotiation

Mid block 是全域語義與 layout 的折衷區。它不像 down blocks 那麼只管結構，也不像 up blocks 那麼只管細節。這裡的任務是讓 prompt、color、geometry、texture 彼此協調。

目前 mid block 的 layer base scale：

```text
color=0.95, geometry=1.05, texture=0.95
```

phase mix：

| Phase | Color | Geometry | Texture | 策略 |
| --- | ---: | ---: | ---: | --- |
| front | `0.10` | `0.80` | `0.10` | 前期仍讓 geometry 主導全域構圖 |
| mid | `0.30` | `0.50` | `0.20` | color 開始進來，geometry 降到半數 |
| back | `0.50` | `0.00` | `0.50` | 後期完全關 geometry，讓 color/texture 接手 |

這裡的關鍵是 mid block back phase 的 geometry 為 `0.00`。這表示一旦大構圖建立，mid block 不再用 geometry 修正全域語義，避免 geometry reference 的物件身份持續干擾 prompt。

如果生成結果「語義被 geometry reference 帶走」：

- 降低 mid block front/mid 的 geometry mix。
- 提早讓 mid block 進入 back-like 行為。
- 強化 prompt 與 negative prompt 的 subject 約束。

如果生成結果「整體色彩不穩」：

- 提高中期 color mix，例如 mid phase 從 `0.30` 往上。
- 搭配 WCT window 放在 mid phase，不要放太早。

### 9.4 Up blocks 策略：appearance rendering

Up blocks 是顏色、材質、筆觸和局部細節最重要的區域。因此目前 up blocks 的策略是讓 geometry 快速退場，改由 color 與 texture 主導。

一般 up blocks 的 layer base scale：

```text
color=1.10, geometry=0.65, texture=1.00
```

other up blocks phase mix：

| Phase | Color | Geometry | Texture | 策略 |
| --- | ---: | ---: | ---: | --- |
| front | `0.30` | `0.40` | `0.30` | 仍保留一些 geometry，避免細節偏離輪廓 |
| mid | `0.40` | `0.30` | `0.30` | color 開始提升 |
| back | `0.40` | `0.00` | `0.60` | 完全關 geometry，texture 最強 |

`up_blocks.0` 被視為較核心的外觀渲染區，策略更激進：

| Phase | Color | Geometry | Texture |
| --- | ---: | ---: | ---: |
| front | `0.35` | `0.40` | `0.25` |
| mid | `0.40` | `0.20` | `0.40` |
| back | `0.50` | `0.00` | `0.50` |

另外對 `up_blocks.0.attentions.1.attn2.processor` 有特殊 override：

```text
front/mid/back = (0.50, 0.00, 0.50)
```

這代表這個 attention block 永遠不看 geometry，只做 color + texture。這是一個很重要的保護：當某些 up block 對視覺細節影響很大時，geometry token 容易把線條、格子、輪廓殘影或 reference subject 帶回來，所以直接把 geometry 關掉。

如果「texture 出不來」：

- 優先提高 up blocks back phase texture mix。
- 提高 `texture_stage_factors` 的 back 值。
- 降低或關閉 up blocks 的 geometry mix。
- 不一定要先提高全域 `texture_scale`，因為 block/phase 沒給位置時，全域 scale 會比較粗暴。

如果「細節有了但主體輪廓崩」：

- 不要直接全局降低 texture。
- 可只在 up blocks front/mid 保留少量 geometry。
- 或讓 geometry 在 front phase 留住，但 back phase 仍歸零。

### 9.5 Step phase 邊界策略

目前 phase 邊界由：

```text
front_end_ratio = 0.40
mid_end_ratio = 0.70
```

切成：

```text
front: 0%  - 40%
mid:   40% - 70%
back:  70% - 100%
```

在 50 steps 下，大約是：

| Phase | Step 範圍 | 主要任務 |
| --- | --- | --- |
| front | `1-20` | 大形、主體數量、pose、layout |
| mid | `21-35` | 全域語義協調、色彩穩定、材質開始進場 |
| back | `36-50` | texture、brushstroke、局部細節、最後色彩 |

調 phase boundary 時，可以用這個判斷：

- geometry 不穩、構圖散：延長 front，或提高 front/mid 的 geometry。
- duplicate subject：縮短 front，或降低 front/mid geometry。
- texture 太弱：讓 texture 更早進 mid，或提高 back texture。
- texture content leakage：延後 texture，降低 front/mid texture。
- color 太晚才出現：提高 mid color，或讓 WCT window 對齊 mid phase。

目前 WCT window 約在 `0.45 ~ 0.55`，剛好落在 mid phase。這是合理的，因為它在大形已初步建立後補 color statistics，又不會太晚才影響整體 tone。

### 9.6 Attention block × step 的調參優先順序

遇到 failure mode 時，建議按「局部 block/phase」優先於「全域 scale」的順序調整。

| 問題 | 優先調整 | 避免一開始就做 |
| --- | --- | --- |
| geometry 不跟 | down/mid front geometry mix、`front_end_ratio` | 大幅提高全域 `geometry_scale` |
| duplicate subject | down/mid geometry mix、geometry 退場時間 | 只靠 negative prompt |
| texture 不明顯 | up/back texture mix、back `texture_stage_factors` | 盲目拉高 `texture_scale` |
| texture reference content 被畫出來 | front/mid texture mix、texture 進場時間、`geometry_sub_scale` | 只降低全部 texture |
| color 不穩 | mid/back color mix、WCT window、`color_scale` | 把 geometry color 混進 color stream |
| 後期有硬邊/格線 | up block geometry mix 歸零 | 降低所有 geometry，導致 pose 崩 |

比較細的原則：

1. **先改 step factor，再改 token scale**：step factor 能控制 stream 在什麼時間生效，比全域 scale 更容易定位問題。
2. **先改 up blocks，再動 down blocks**：若問題是材質或細節，通常在 up blocks 處理；動 down blocks 會影響整體構圖。
3. **duplicate subject 優先查 down/mid geometry**：重複主體通常是在早期被鎖進去，不是後期 texture 單獨造成。
4. **texture leakage 優先查 front/mid texture**：如果 texture 在早期參與太多，它就會變成另一個 content reference。
5. **color 問題優先查 mid phase 與 WCT**：color 是全域統計，mid phase 的修正通常比 late-only 修正更有效。

### 9.7 可考慮的下一版 attention block profile

目前 profile 已經能運作，但可以針對常見問題做幾種候選 profile。

**Profile A：single-subject safe**

目標是降低雙主體與 reference subject leakage：

```text
down front:  geometry high, texture zero
down mid:    geometry medium-high, texture very low
mid front:   geometry medium, color low, texture low
mid back:    geometry zero
up all:      geometry zero or very low
```

適用情況：

- 狗、人像、小孩等單主體任務。
- geometry reference subject 很強。
- texture reference 容易帶出非目標物件。

**Profile B：texture recovery**

目標是強化材質與筆觸：

```text
down: geometry remains dominant
mid mid/back: texture increases earlier
up mid/back: texture strongest
up_blocks.0.attentions.1: color + texture only
```

適用情況：

- 目前結果構圖穩，但材質太平。
- texture 沒有 content leakage，只是不夠明顯。

**Profile C：color faithful**

目標是讓 color reference 更穩：

```text
down front: geometry only
mid phase: color increases to 0.4-0.6
back phase: color + texture, geometry zero
WCT window aligned to mid phase
```

適用情況：

- 色彩偏灰、偏 prompt default palette。
- texture 有出來但 color ref 不夠貼。

**Profile D：geometry faithful**

目標是提高 pose/layout：

```text
down front/mid: geometry very high
mid front/mid: geometry medium-high
up front: keep small geometry
up back: geometry zero
```

適用情況：

- 建築、姿勢、場景 layout 需要更像 geometry reference。
- 但不適合已經有 duplicate subject 的任務。

## 10. WCT Latent Guidance

WCT 在 pipeline 中不是 IP token，而是 latent-level guidance。

### 10.1 WCT 觸發條件

只有在以下條件都成立時啟動：

```text
wct_guidance != 0
color reference exists
i >= wct_starts_step
i < wct_ends_step
i + 1 < len(timesteps)
```

目前入口設定約在 50 steps 的第 22.5 到 27.5 步附近啟動，也就是中期一小段。

### 10.2 WCT transform

`wct_transform_with_zt_add_noise()` 流程：

1. color reference resize 到 1024。
2. 將 color image encode 成 VAE latent，並在指定 timestep 加噪。
3. 對目前 latent `latents` 與 color latent 做 `whiten_and_color()`。
4. 根據 scheduler 的 beta 加回一點初始噪聲：

```text
result_noise = WCT(latents, color_latents) + wctnoise_add_scale * sqrt(beta_t) * zT_latent
```

5. 與原本 scheduler step 後的 latent 混合：

```text
latents = (1 - wct_guidance) * latents + wct_guidance * result_noise
```

### 10.3 WCT 的角色

WCT 主要補強 color reference 的整體 latent statistics，不直接負責 geometry 或 texture。

它的優點：

- 可以強化色彩氛圍。
- 不需要額外訓練。
- 和 IP token control 是不同通道，可互補。

風險：

- `wct_guidance` 太高會讓色彩過度壓過 prompt 或 texture。
- window 太早可能干擾結構形成。
- window 太晚可能只改到表面 tone，效果有限。

## 11. 輸出後處理

推論後會呼叫 `_enhance_final_output()`：

| 參數 | 目前值 |
| --- | ---: |
| `FINAL_OUTPUT_BRIGHTNESS` | `1.00` |
| `FINAL_OUTPUT_SATURATION` | `1.24` |
| `FINAL_OUTPUT_CONTRAST` | `1.02` |

主要是補償生成結果偏灰、偏低飽和的問題。這不改變方法本身的條件控制，但會影響 metrics 中的 color histogram 與視覺評估。

## 12. Runtime Modes

`run_infer_style_plus_all_modes.py` 支援四種模式：

| Mode | Color | Geometry | Texture |
| --- | --- | --- | --- |
| `color_only` | on | off | off |
| `color_texture_legacy` | on | off | on |
| `color_geometry` | on | on | off |
| `color_texture_geometry` | on | on | on |

這些模式是重要 ablation：

- `color_only` 測 palette 是否穩。
- `color_texture_legacy` 測沒有 geometry 時 texture 是否帶構圖。
- `color_geometry` 測 geometry 是否能獨立控制 layout。
- `color_texture_geometry` 測完整方法的三路協同。

`test_runtime_config.py` 也確認 runtime files 不再依賴 `mode_profiles`，目前 stream scale 與 phase mix 都是固定在程式內。

## 13. Batch Experiment 設計

`run_dog_tuning_experiments.py` 是完整 sweep 與評估工具。

### 13.1 Asset triple

每次 run 使用：

```text
AssetTriple = (geometry, color, texture, optional texture2)
```

支援 geometry categories：

```text
animal, face, house, person, seen
```

預設啟用：

```text
animal, face, house, seen
```

### 13.2 Prompt 設計

每個 geometry category 有預設 prompt，例如：

- animal: `A single dog on the ground`
- face: `A portrait of a man with a beard`
- house: `One traditional Chinese temple on the ground`
- seen: `A building of a beautiful church in a city`

也支援 prompt file。針對 child/person 任務，程式會加強 single-subject positive / negative clauses。

### 13.3 Baseline images

batch script 會建立：

- geometry baseline：只根據 geometry baseline prompt 生成，作為幾何對照。
- output baseline：不使用完整三路方法的對照輸出。

這些 baseline 用來計算 geometry residual / improvement ratio。

## 14. Metrics

目前 metrics 包含：

| Metric | 用途 |
| --- | --- |
| `clip_text` | 輸出與 prompt 的 CLIP image-text similarity |
| `ms_ssim_geometry` | output sketch 與 geometry sketch 的 MS-SSIM |
| `edge_recall_geometry` | output edge 對 geometry edge 的 recall |
| `clip_texture_gray` | 輸出灰階與 texture 灰階的 CLIP image-image similarity |
| `color_histogram_distance` | 輸出與 color ref 的色彩 histogram 距離 |
| `geometry_residual_l2` | geometry residual 差異 |
| `geometry_residual_cosine` | geometry residual cosine |
| `geometry_improvement_ratio` | 相對 baseline 的 geometry improvement |

### 14.1 Geometry metric preprocessing

geometry metric 不是直接比 RGB，而是先 `sketchify_for_clip_geometry()` 或 Canny：

```text
output image -> sketch / canny
geometry image -> sketch / canny
compare sketch-level similarity
```

這比直接 RGB similarity 更符合 geometry stream 的目標。

### 14.2 Texture metric

texture metric 目前使用灰階 CLIP similarity：

```text
CLIP(output grayscale RGB, texture grayscale RGB)
```

這能粗略評估 texture pattern，但也有風險：如果 texture reference 的 subject content 被畫進輸出，分數可能變高，卻不代表真正的材質轉移成功。因此人工評估仍需要檢查 texture content leakage。

## 15. ControlNet 版本

`infer_style_controlnet_color_texture.py` 是 ControlNet-based image-to-image stylization。

它使用：

- SDXL ControlNet Canny。
- IPAdapterPlusXL。
- color / texture disentanglement。
- 可選 direct sketch 作為 ControlNet image。

目前這個腳本仍比較接近原始 color-texture SADis + ControlNet，沒有完整使用主入口那套三路 geometry stream schedule。geometry 在 ControlNet 腳本中主要以 `control_image` 形式進入 ControlNet，而不是完全等同 `infer_style_plus_color_texture.py` 的 geometry stream。

## 16. 和原始 SADis 的差異

原始 SADis 主要是：

```text
color image RGB token - color image grayscale token + texture grayscale token
```

目前方法的改動更大：

1. 新增 geometry stream，三路 token 固定拼接。
2. color / geometry / texture 不再單純線性混成一個 appearance token，而是保留成三組 IP tokens。
3. 新增多方向 projection subtraction，控制 stream 間 leakage。
4. 新增 geometry reference 前處理，用 RGB-form geometry cue 送入 CLIP Vision。
5. 新增 layer-aware stream base scale。
6. 新增 phase-aware attention mix table。
7. pipeline denoising loop 每一步會更新 stream phase 與 step factor。
8. 保留 WCT latent guidance 作為 color 補強。
9. 新增 batch metrics，將 text、geometry、texture、color 分別量化。

可以把現在方法理解成：

```text
SADis 原始 color-texture disentanglement
+ Geometry reference stream
+ Token-level multi-direction disentanglement
+ Layer/phase adaptive IP attention routing
+ Latent WCT color statistics guidance
+ Geometry-aware experiment metrics
```

## 17. 目前方法的優點

### 17.1 控制維度更清楚

color、geometry、texture 分成三路後，可以各自調：

- reference source
- token scale
- decouple scale
- layer scale
- phase schedule
- metrics

比舊版單一 style embedding 更容易做 ablation。

### 17.2 Geometry 不只是 ControlNet

geometry stream 用 CLIP Vision token 進 IP-Adapter，不要求輸入一定是 Canny map，也能使用經過 homogenization 的 reference image。這讓 geometry 控制比較柔性，不會像 ControlNet 那樣強硬鎖邊。

### 17.3 Attention routing 有時間感

同一組 IP tokens 在不同 denoising phase 會被不同方式混合：

- 早期：geometry 幫助 layout。
- 中期：color/geometry/texture 協調。
- 後期：geometry 退場，color/texture 收尾。

這符合 diffusion 生成過程中「先大形、後細節」的特性。

### 17.4 不需要訓練

方法主要使用：

- CLIP Vision encoder。
- IP-Adapter Plus XL Resampler 與 attention weights。
- SDXL base。
- 手工 token operations。
- 手工 schedule。

因此不需要額外 training，但仍能做多參考圖控制。

## 18. 主要風險與 Failure Modes

### 18.1 Texture content leakage

texture reference 即使轉灰階，仍可能保留 subject / composition。當 `texture_scale` 高、texture 太早介入、或 geometry subtraction 不夠時，texture reference 的內容會被畫進輸出。

相關因素：

- `texture_ref_img.convert("L")` 只去色，不去語義。
- `texture_late_start` 如果太早，texture 會參與構圖。
- `texture_to_geometry_decouple` 太低，texture 中的 shape 沒被扣乾淨。
- `punish_weight` 太低時，texture subject 可能更強；太高時，texture 會消失。

### 18.2 Geometry semantic leakage

geometry reference 經 CLIP Vision 後仍可能攜帶物件身份，而不只是輪廓。若 geometry 在 down/mid 太強，可能造成：

- duplicate subject。
- 雙頭、雙身體。
- reference subject 進入 output。
- prompt subject 與 geometry subject 混合。

這也是 `note.md` 中提出 residual geometry token 的原因：未來可嘗試從 geometry patch token 扣掉 text semantic direction。

### 18.3 Texture 被過度削弱

texture 會經過：

```text
grayscale
texture <-> geometry projection subtraction
texture <-> color projection subtraction
gray punishment
layer/phase weighting
```

這些步驟疊加後，texture 可能只剩弱明暗，導致毛流、筆觸、材質 pattern 不明顯。

### 18.4 Color / texture / geometry 調參耦合

單一參數的效果可能被其他 routing 參數抵消。例如：

- 提高 `texture_scale`，但 attention phase 不給 texture 權重，效果仍弱。
- 降低 `geometry_scale`，但 down block phase mix 仍是 geometry-heavy，早期仍可能被 geometry 鎖住。
- 提高 `color_scale`，但 WCT guidance 同時存在，可能難判斷是哪個 color path 生效。

## 19. 目前調參邏輯

依照 `note.md` 和目前程式，較合理的調參順序是：

### 19.1 先穩定主體數量與 geometry

如果出現 duplicate subject：

1. prompt 加上 single subject，例如 `one dog, single dog, solo dog portrait`。
2. negative prompt 加上 `two dogs, multiple dogs, duplicate dog, extra dog, two heads, duplicate body`。
3. 降低 `geometry_scale`。
4. 讓 geometry 更早退場，降低後期 `geometry_stage_factors`。
5. 降低 down/mid 的 geometry layer 或 phase 權重。

### 19.2 再補 texture

主體穩定後再調 texture：

1. 降低 `texture_color_decouple`。
2. 降低或關閉 `punish_weight`。
3. 提高 `texture_scale`。
4. 增加後期 texture factor。
5. 測 raw gray / material gray / high-pass texture 變體。

### 19.3 最後處理 texture content leakage

若 texture content 被畫進輸出：

1. 提高 `texture_to_geometry_decouple` 或 `geometry_sub_scale`。
2. 延後 texture 介入。
3. 降低 early texture factor。
4. 改用更 material-like 的 texture preprocess。
5. negative prompt 加上 `copy texture reference subject`、`texture reference composition`、`non-dog subject` 等。

## 20. 建議的 Method 命名

若要在報告或論文中描述目前版本，可以考慮命名為：

```text
Geometry-Aware SADis with Phase-Aware Multi-Stream IP Attention
```

或較短：

```text
Tri-Stream SADis
```

方法一句話摘要：

```text
We extend SADis into a training-free tri-stream reference conditioning framework that separately encodes color, geometry, and texture references, purifies their CLIP Vision tokens by projection-based disentanglement, and routes them through layer-aware and phase-aware IP-Adapter cross-attention during SDXL denoising.
```

繁中摘要：

```text
我們將 SADis 擴展為免訓練的三路 reference conditioning 方法，分別編碼 color、geometry、texture 參考圖，透過 projection subtraction 在 CLIP Vision token space 中降低三路語義污染，再於 SDXL denoising 過程中用 layer-aware 與 phase-aware 的 IP-Adapter cross-attention 動態路由三路條件。
```

## 21. 可寫進 Method Section 的公式化描述

令三張 reference 經 CLIP Vision 得到 patch tokens：

```text
C = E_v(I_c)
G = E_v(P_g(I_g))
T = E_v(P_t(I_t))
```

其中 `P_g` 是 geometry preprocess，`P_t` 是 texture grayscale / high-pass preprocess。

Projection residual 定義為：

```text
R(A, B; lambda) = A - lambda * ((A · B) / (||B||^2 + epsilon)) B
```

Geometry color removal：

```text
G_pre = R(G, C; lambda_gc)
```

Texture-geometry disentanglement：

```text
T_g = R(T, G_pre; lambda_tg)
G_final = R(G_pre, T; lambda_gt)
```

Color disentanglement：

```text
C_g = R(C, G_pre; lambda_cg)
C_final = R(C_g, T_g; lambda_ct)
T_final = R(T_g, C; lambda_tc)
```

Energy restoration：

```text
restore(A, A') = A' * ||A|| / (||A'|| + epsilon)
```

Stream tokens：

```text
S_c = Resampler(color_scale * restore(C, C_final) - substract_scale * E_v(gray(I_c)))
S_g = Resampler(geometry_scale * restore(G, G_final))
S_t = Resampler(texture_scale * restore(T, T_final))
```

Final encoder hidden states：

```text
H = concat(H_text, S_c, S_g, S_t)
```

At each U-Net cross-attention layer `l` and denoising phase `p`, the active IP token is:

```text
S_active(l, p) =
    alpha_c(l, p) S_c
  + alpha_g(l, p) S_g
  + alpha_t(l, p) S_t
```

The IP attention output is added to the text attention output:

```text
Attn_out = Attn_text(Q, K_text, V_text)
         + s_ip(l, p, step) * Attn_ip(Q, K_ip(S_active), V_ip(S_active))
```

其中 `s_ip` 由 global IP scale、layer stream scale、phase mix、step factor 共同決定。

## 22. 目前最關鍵的程式對照表

| 方法概念 | 程式位置 |
| --- | --- |
| reference path / runtime mode | `infer_style_plus_color_texture.py` |
| geometry preprocess | `preprocess_geometry_for_rules()` |
| geometry edge filtering | `_filter_small_edge_components()`、`_filter_short_edge_components()`、`_keep_major_edge_components()` |
| CLIP Vision hidden states | `IPAdapterPlusXL.get_image_embeds()` |
| projection subtraction | `proj_res()` |
| energy restoration | `restore()` |
| texture gray punishment | `punish_weight_module()` 與 `get_image_embeds()` 內 gray token path |
| three-stream token concat | `image_prompt_embeds = torch.cat([color_p, geo_p, tex_p], dim=1)` |
| text + image token concat | `IPAdapterPlusXL.generate()` |
| per-layer stream scale | `IPAdapterPlusXL.get_layer_stream_scales()` |
| phase mix table | `get_phase_mix_table()` |
| three-stream attention split | `IPAttnProcessor2_0.__call__()` |
| denoising phase schedule | `StableDiffusionXLPipeline.__call__()` |
| WCT latent guidance | `wct_transform_with_zt_add_noise()` |
| batch sweep | `run_dog_tuning_experiments.py` |
| metrics | `compute_metrics()` |

## 23. 後續可改進方向

### 23.1 Residual geometry token

`note.md` 已提出一個重要方向：geometry token 扣除 text semantic direction。

目前 geometry purification 主要扣 color / texture，還沒有直接扣掉物件身份語義。未來可以做：

```text
G_res = G_patch - lambda_text * text_semantic_direction
```

目標是讓 geometry stream 更像 layout / pose，不像「另一個物件 reference」。

### 23.2 Texture material representation

目前 texture 主要是 grayscale。未來可比較：

- raw grayscale。
- high-pass grayscale。
- band-pass texture。
- local contrast normalized texture。
- structure-suppressed material map。

目標是保留材質 pattern，同時壓掉 texture reference 的 subject content。

### 23.3 更細的 layer profile

目前 `get_layer_stream_scales()` 只分 down/mid/up。未來可以細到：

- specific block id。
- specific attention index。
- only attn2 processors。
- different profile for animal / face / building。

這能更精準控制 geometry 何時退場、texture 何時接手。

### 23.4 Metrics 補強

目前 metrics 已可用，但 texture metric 仍可能被 content leakage 誤導。可加入：

- texture reference subject leakage classifier。
- edge overlap penalty for texture reference。
- color harmony metric。
- single-subject detector。
- human evaluation rubric：S1 single subject、S2 texture、S3 geometry、S4 color。

## 24. 總結

目前 SADis 方法的核心不是單純「把三張圖丟進模型」，而是：

1. 將 reference control 明確拆成 color、geometry、texture 三個 stream。
2. 在 CLIP Vision patch-token space 中用 projection subtraction 做多方向解耦。
3. 保留每一路獨立 token，避免過早混成單一 appearance embedding。
4. 在 IP-Adapter cross-attention 中根據 layer 與 denoising phase 動態混合三路 token。
5. 用 geometry 前處理與後期 geometry 退場降低構圖污染。
6. 用 texture grayscale、gray punishment、texture/geometry decouple 嘗試保留材質並壓制 content leakage。
7. 用 WCT 在 latent space 補強 color statistics。
8. 用 batch experiments 和多指標 metrics 分別觀察 text、geometry、texture、color 的控制效果。

一句話來說，這版是「免訓練、三路解耦、phase-aware attention routing」的 SADis 擴展版。
