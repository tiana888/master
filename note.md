# SADis Dog Tuning Note

更新日期：2026-03-14

## 1. 目前問題

目前版本的主要 failure mode 有兩個：

- `texture 出不來`：生成結果只有大致明暗或色塊，毛流、材質、表面紋理不明顯。
- `會有兩隻狗`：畫面容易出現重複主體、雙頭、雙身體，或額外犬隻輪廓。

這次先不改 Python 程式，只整理成下一輪可直接執行的調參策略。

## 2. 目前實作判讀

### 2.1 三路條件流的現況

`infer_style_plus_color_texture.py` 目前同時平衡三個條件流：

- `color stream`
- `geometry stream`
- `texture stream`

從目前的 baseline 與實驗分組來看，調參方向偏向：

- 早期與中期讓 `geometry` 參與感偏強
- `texture` 主要靠灰階或高通後的訊號進入
- 真正能代表材質的 texture 訊號，在進投影前已被削弱多次

### 2.2 `ip_adapter.py` 的關鍵影響

`IPAdapterPlusXL.get_image_embeds()` 目前對 geometry / texture 做了幾層很強的處理：

- geometry 先做 token 對比增強
- geometry 再做對 color 方向的正交扣除
- geometry 再做 energy normalization
- texture 先轉灰階
- texture 可能再被 `geometry_sub_scale` 扣掉一部分
- texture 再被 `punish_weight` / `punish_type` 壓抑
- texture 最後再被 `texture_color_decouple` 扣除 color 方向

這個組合很容易出現：

- geometry 語義過強，提早把「狗的主體數量」鎖死，導致雙狗
- texture 在進入 `image_proj_model` 前已經太弱，只剩弱明暗訊號，導致 texture 不明顯

### 2.3 當前 layer stream scale 的風險

`get_layer_stream_scales()` 的預設配置是：

- `down_blocks`: `[0.75, 1.20, 0.20]`
- `mid_block`: `[0.70, 1.35, 0.35]`
- `up_blocks`: `[1.05, 0.55, 0.70]`

這代表 geometry 在 `down_blocks` / `mid_block` 是最強流，而這兩段正是最容易把版面、主體數量、輪廓配置鎖住的位置。
因此「第二隻狗」更可能是 geometry stream 過早、過強介入造成，而不是 texture 本身造成。

## 3. Root Cause 假設

### H1. 雙狗主要來自 geometry stream 語義洩漏

最可能原因不是 texture 太強，而是：

- geometry 參與時機太早
- geometry 在 early / mid block 權重太高
- geometry reference 不只提供輪廓，也帶入了額外語義

結果是模型在早期就決定畫面裡需要第二個犬隻形體或重複肢體。

### H2. texture 弱主要來自 decouple / punish 過重

目前 texture 訊號經過：

- grayscale 化
- geometry subtraction
- gray punish
- color decouple

連續處理後，CLIP embedding 中真正和材質相關的可用訊號被沖淡，最後只剩下弱對比與弱紋路。

### H3. prompt / negative prompt 對單主體約束不夠

如果 prompt 仍是偏寬鬆的 `a dog`，負面詞也沒有明確壓制 duplicate subject，模型在強 geometry 條件下更容易出現：

- two dogs
- extra dog
- duplicate dog
- duplicate body
- duplicate head
- multiple animals

所以下一輪應明確改成偏單主體描述，例如：

- `one dog`
- `single dog`
- `solo dog portrait`

並加強 duplicate-subject negative prompt。

## 4. 建議的優先實驗順序

### Step 1. 先穩定單狗，再談 texture

第一輪先只處理「雙狗」：

- 提高 prompt 對單主體的描述強度
- 在 negative prompt 明確加入 duplicate subject 類詞
- 降低 geometry 在 early / mid 的主導程度
- 縮短 geometry 生效區間

這一步不先追 texture，目標只有：

- 畫面只剩一隻狗
- silhouette / pose 還能維持

建議先測：

- `prompt`: `one dog, single dog, solo dog`
- negative prompt 加入：`two dogs, multiple dogs, duplicate dog, extra dog, two heads, duplicate body, multiple animals`

幾何排程優先往這個方向試：

- `guidance_scale`: 往上微調
- `geometry_scale`: 往下壓
- `geometry_early_end`: 更早結束
- `geometry_late_factor`: 更低

### Step 2. 在單狗穩住後回補 texture

第二輪才開始追 texture：

- 先測 `raw_gray`
- 再測 `material_gray`
- 優先降低 `texture_color_decouple`
- 優先降低或直接關掉 `punish_weight`
- 讓 texture 更早開始介入

理由：

- `raw_gray` 比 `material_gray` 更能保留原始紋理節奏
- 如果 texture 已經很弱，再繼續 decouple / punish 只會更乾
- texture 介入太晚時，早期結構已被 geometry 鎖定，表面細節很難翻盤

### Step 3. 最後再做 attention / token follow-up

如果 Step 1 + Step 2 後仍然 texture 弱，再做第三輪：

- 新增 texture-focused attention profile
- 測不對稱 token 配置
- 不要只一味增加 geometry tokens

下一輪注意方向：

- geometry tokens 增加，不一定會改善單狗，反而可能放大雙狗
- 更值得測的是讓 texture stream 在對的 layer 有更高權重

## 5. 建議下一輪參數範圍

以下不是一次全開，而是作為下一輪 sweep 範圍。

| 項目 | 目前常見值 | 下一輪建議 |
| --- | --- | --- |
| `guidance_scale` | `8.8` | `8.8 ~ 9.4` |
| `geometry_scale` | `1.10 ~ 1.20` | `0.90 ~ 1.10` |
| `geometry_early_end` | `0.10 ~ 0.15` | `0.06 ~ 0.12` |
| `geometry_late_factor` | `0.03 ~ 0.08` | `0.00 ~ 0.04` |
| `texture_scale` | `1.00 ~ 1.20` | `1.10 ~ 1.35` |
| `texture_late_start` | `0.60 ~ 0.68` | `0.45 ~ 0.60` |
| `texture_early_factor` | `0.18 ~ 0.22` | `0.22 ~ 0.35` |
| `texture_color_decouple` | `0.16` | `0.05 ~ 0.12` |
| `punish_weight` | `0.0005 ~ 0.0008` | `0 ~ 0.0004` |

調參原則：

- 一旦 texture 消失，先降 `texture_color_decouple` 與 `punish_weight`
- 一旦雙狗出現，先降 `geometry_scale`、縮短 `geometry_early_end`、壓低 `geometry_late_factor`
- `guidance_scale` 只做小幅上調，不要用它硬扛所有問題

## 6. 建議的實驗設計

### Group A. Single-dog stabilization

目標：先把第二隻狗壓掉。

建議固定：

- `texture_scale` 暫時不要再往上加
- `texture_color_decouple` 先維持
- `punish_weight` 先維持

只改：

- `prompt`
- `negative_prompt`
- `guidance_scale`
- `geometry_scale`
- `geometry_early_end`
- `geometry_late_factor`

通過標準：

- `S1 >= 4`
- `S3 >= 4`

### Group B. Texture recovery

前提：Group A 已經能穩定維持單狗。

建議順序：

1. `raw_gray` vs `material_gray`
2. `texture_color_decouple` 往下
3. `punish_weight` 往下或設 `0`
4. `texture_late_start` 提早
5. `texture_early_factor` 往上
6. `texture_scale` 最後才往上加

原因：
如果一開始就只提高 `texture_scale`，很容易把弱 texture embedding 放大，但不一定真的增加材質訊號。

### Group C. Attention / token follow-up

只在前兩組仍不夠時進行。

優先測：

- texture-focused stream scales
- color / geometry / texture 的不對稱 token 數

不建議第一優先就做：

- 單純把 geometry tokens 拉高

## 7. 如果下一步要改碼

這次先不改程式，但下一輪若要進 code，可以按以下順序：

### 7.1 `ip_adapter.py`

優先方向：

- 把 geometry sharpening 從現在的強增強改成更保守版本
- 下調 early / mid block 的 geometry stream scale
- 新增 texture-focused attention profile，而不是只有 geometry-focused profile
- 嘗試 RGB high-pass / band-pass texture feature，不要只依賴灰階 texture reference

### 7.2 `attention_processor.py`

如果這裡已經支援 per-stream weighting，下一輪應優先：

- 把 texture boost 放到 per-layer / per-stream 權重
- 不要只依賴全域 `texture_scale`

這樣比較有機會在不放大雙狗風險的前提下，增加 texture 可見度。

## 8. 評分與驗收標準

後續每個 candidate 都用同一套標準打分：

- `S1 single subject`：只能有一隻狗，不能有 duplicate head/body
- `S2 texture`：能看出 texture ref 的毛流、材質、表面 pattern，不只是平坦色調
- `S3 geometry`：姿勢、輪廓、主體結構維持正確
- `S4 color`：顏色分布仍貼近 color ref

建議最低驗收：

- `S1 >= 4`
- `S2 >= 4`
- `S3 >= 4`
- `S4 >= 4`

## 9. Seed 與 ablation 規則

每個 candidate 至少測 3 個 seed：

- `42`
- `123`
- `7`

只要出現以下情況，就直接淘汰：

- texture 有變好，但第二隻狗回來
- color 或 geometry 明顯崩掉

建議 ablation 順序固定為：

1. 先測 `prompt / negative_prompt + geometry schedule`
2. 再測 `texture_color_decouple + punish_weight`
3. 最後測 `attention profile + token allocation`

不要同一輪把三類改動混在一起，否則很難知道 texture 變好或雙狗消失到底是哪個因子造成。

## 10. 下一輪建議 baseline

下一輪可以從這組概念 baseline 開始：

- `prompt`: `one dog, single dog`
- `guidance_scale`: `9.0`
- `geometry_scale`: `1.00`
- `geometry_early_end`: `0.08`
- `geometry_late_factor`: `0.02`
- `texture_mode`: `raw_gray`
- `texture_scale`: `1.15`
- `texture_late_start`: `0.52`
- `texture_early_factor`: `0.28`
- `texture_color_decouple`: `0.08`
- `punish_weight`: `0.0002`

如果這組還是雙狗，先繼續壓 geometry。
如果這組只剩單狗但 texture 仍弱，先降 `punish_weight` 到 `0`，再測更早的 texture schedule。

## 11. Geometry Residual 方法提案

### 11.1 方法核心

這一版 geometry preprocess 的候選方向，不再只依賴 image-only 的 sharpen / color orthogonal subtraction，
而是改成直接在 CLIP Vision patch token 上做語義殘差：

`Image -> CLIP Vision -> (Image Embeds - Text Embeds) -> Residual Token -> Attention`

核心想法：

- geometry image 先進 CLIP Vision
- 取出 patch-level hidden states 作為空間結構表徵
- 再扣掉和主體語義對應的 text semantic direction
- 保留較純的空間配置殘差 token
- 最後把這組 residual token 當成 geometry stream 的 attention token

這個方法的目標不是讓 geometry 更強，而是讓 geometry 更乾淨。
重點是降低 geometry token 裡面和「物件身分」綁定過深的語義，避免它在 early / mid block 太早把第二隻狗鎖進去。

### 11.2 和現行 geometry 分支的差異

目前 [`ip_adapter.py`](/c:/Users/tianachen/SADis/ip_adapter/ip_adapter.py) 的 geometry branch 大致是：

- `geo_raw = image_encoder(...).hidden_states[-2]`
- `geo_raw -> token contrast sharpen`
- `geo_raw -> remove color direction`
- `geo_raw -> energy align`
- `geometry_stream_embeds = geometry_scale * geo_clip_image_embeds`

也就是說，現行方法本質上還是 image-only token purification，
只是額外對 color 方向做扣除，還沒有直接對「文字語義方向」做 subtraction。

新法改成：

- `patch_embeds = image_encoder(...).hidden_states[-2]`
- `text_feat = _get_text_semantic_feature(label)`
- `residual_geo = patch_embeds - geometry_text_neg_scale * text_tokens`
- `residual_geo = renorm(residual_geo)`
- `geometry_stream_embeds = geometry_scale * residual_geo`

要在 note 中明確記住：

- 這不是 preprocess geometry 圖片本身
- 這是 preprocess geometry token
- 也就是 geometry_ref_img 仍然可以沿用現在的輸入圖與前處理，但真正要替換的是進 CLIP 之後的 token purification 邏輯

### 11.3 建議整合點

主改動位置應放在 [`ip_adapter.py`](/c:/Users/tianachen/SADis/ip_adapter/ip_adapter.py) 的 `IPAdapterPlusXL.get_image_embeds()` geometry branch。

建議整合方式：

- 保留現有 `geometry_ref_img -> clip_image_processor -> image_encoder(..., output_hidden_states=True)` 入口
- 在 geometry 分支新增 `get_purified_geometry_token()` helper
- 在 class 內新增 `_get_text_semantic_feature()` helper
- `geometry_ref_img` 仍然從既有 preprocess 流進來
- 但進 CLIP 後，不再把目前的 `contrast sharpen -> remove color direction -> energy align` 當成唯一主流程
- 改由 residual geometry path 產生新的 geometry token

建議寫法方向：

- 在 `get_image_embeds()` 中，geometry token 生成改成呼叫 `get_purified_geometry_token(...)`
- 若 residual mode 關閉，才 fallback 到目前舊版 geometry 流程
- 這樣之後可以做 baseline / residual A/B 比較，不需要一次把舊法刪掉

### 11.4 建議的最小介面

未來若要實作，note 先把最小 helper 介面鎖定為：

- `get_purified_geometry_token(self, geo_clip_image, label)`
- `_get_text_semantic_feature(self, label)`

參數與預設：

- `label` 不要寫死成 `a tree`
- 目前 dog 任務預設用 `a dog`
- 後續若要泛化，再把它提升成可配置參數 `geometry_semantic_label`

關於扣除強度的命名：

- 目前 code 已經有 `geo_neg_scale`
- 但它現在主要是 color orthogonal subtraction 相關語意
- 若導入 text residual 法，建議新增更明確的新參數 `geometry_text_neg_scale`
- note 中先推薦用 `geometry_text_neg_scale`，避免和現有 `geo_neg_scale` 混淆

v1 介面原則：

- `geometry_text_neg_scale` 預設先獨立於 `geo_neg_scale`
- `geometry_semantic_label` 預設為 `a dog`
- 若沒有提供 label，就 fallback 到 `a dog`

### 11.5 預期效益與風險

預期效益：

- geometry stream 的物件身分洩漏變少
- 第二隻狗、雙頭、重複身體的機率下降
- early / mid block 接收到的 geometry token 更偏向布局與姿勢，而不是「又一隻狗」

主要風險：

- 如果 text semantic 扣除太強，geometry 可能只剩下碎片化輪廓
- 姿勢、輪廓、肢體連續性可能反而變差
- 若 residual token 過弱，geometry stream 會失去本來的姿勢控制能力

另一個技術風險：

- CLIP 的 pooled text feature 和 patch token 在粒度上不完全一致
- pooled text feature 偏全域語義
- patch tokens 偏局部空間結構
- 所以 subtraction 可能有效，也可能因語義空間不完全對齊而產生不穩定殘差

因此 note 要明確記錄：

- v1 先用 pooled text feature 做全域語義方向
- v2 再比較 `pooled_prompt_embeds` 和 `prompt_embeds mean pooling`

## 12. Implementation Prerequisites

在真正動 code 之前，先把目前 repo 現況記清楚：

- 現有 code 沒有 `_get_text_semantic_feature()` helper
- 現有 geometry branch 已經使用 `hidden_states[-2]` patch tokens，這和 residual geometry 方案相容
- 現有 `self.pipe.encode_prompt(...)` 已存在，可以作為文字特徵入口
- 但目前還沒有統一好的文字語義抽取 helper，所以這一步需要先補

v1 預設選擇：

- 先用 `self.pipe.encode_prompt(...)` 回傳的 `pooled_prompt_embeds` 作為全域語義方向
- 先不要在第一版就引入 token-level text alignment
- 先做最小可驗證版本，確認 residual subtraction 是否真的改善雙狗

## 13. 建議偽碼

```python
# geometry branch inside get_image_embeds()
geo_outputs = self.image_encoder(geo_clip_image, output_hidden_states=True)
patch_embeds = geo_outputs.hidden_states[-2]

text_feat = self._get_text_semantic_feature(label)  # v1: pooled_prompt_embeds
text_tokens = text_feat[:, None, :].repeat(1, patch_embeds.shape[1], 1)

residual_geo = patch_embeds - geometry_text_neg_scale * text_tokens
residual_geo = residual_geo / (residual_geo.norm(p=2, dim=-1, keepdim=True) + 1e-6)

geometry_stream_embeds = geometry_scale * residual_geo
```

補充原則：

- `patch_embeds` 維持使用 `hidden_states[-2]`
- `text_feat` v1 先使用 pooled text feature
- renorm 不能省略，否則 geometry residual 的能量可能不穩
- residual 版先不要混入現在那段強 contrast sharpen，避免難以判斷效果來源

## 14. v1 不做的事

第一版先不要同時擴大問題範圍：

- 不同 block 使用不同 label
- 動態 label per prompt
- text token attention map 對齊
- multi-label residual subtraction
- 和 texture branch 一起重構
- 和 attention profile / token profile 一起耦合調整

v1 的目標很單純：

- 驗證「geometry token 減去 text semantic direction」是否能降低雙狗
- 並且不明顯破壞 pose / silhouette

## 15. Residual Geometry 實驗設計

### Phase R1: replace-only

第一輪只做純替換：

- geometry branch 從舊法換成 residual 法
- color / texture 參數完全固定
- prompt / negative prompt / seed / texture mode 全部和 baseline 相同

比較重點：

- 單狗率有沒有提升
- 姿勢穩定度有沒有下降
- texture 有沒有被間接改善或拖垮

### Phase R2: subtraction strength sweep

第二輪只 sweep `geometry_text_neg_scale`：

- `0.15`
- `0.30`
- `0.45`
- `0.60`

目標：

- 找到「雙狗顯著下降」但「silhouette 不崩」的甜蜜點

判讀原則：

- 太低：語義扣除不夠，雙狗仍在
- 太高：geometry 被扣太乾淨，姿勢與輪廓不穩

### Phase R3: text feature variant

第三輪再比較文字特徵形式：

- `pooled_prompt_embeds`
- `prompt_embeds mean pooling`

v1 預設：

- 先做 `pooled_prompt_embeds`
- 若效果接近可用，再做第二種變體比較

## 16. 驗收標準與對照組規範

Residual geometry 實驗的驗收標準：

- `S1`: 單狗率提升，重複主體下降
- `S3`: 姿勢與輪廓至少不比舊 geometry branch 差
- `S2`: texture 不要求直接提升，但不能因 residual geometry 讓 texture 更塌

對照組規範：

- baseline 必須是目前 `get_image_embeds()` 的舊 geometry branch
- residual 版和 baseline 必須使用相同 seed
- residual 版和 baseline 必須使用相同 prompt / negative prompt
- residual 版和 baseline 必須使用相同 texture mode
- 每組至少 3 個 seed：`42` / `123` / `7`

建議最小對照矩陣：

- Baseline geometry x 3 seeds
- Residual geometry (`geometry_text_neg_scale=0.30`) x 3 seeds
- 確認有正向訊號後，再擴到 sweep

## 17. G13_E02 成功案例與問題判讀

參考圖片：

- `results/dog_tuning/20260314_200413/group13_texture_recover/G13_E02_group13_texture_recover_cfg9.0_geo1.0_gend0.08_glate0.02_color1.85_sub0.25_tex1.15_tlate0.48_tearly0.32_pw0.0002_geomode-infer_original_texmode-raw_gray_attn-default_seed42.png`

這張圖目前的觀察是：

- 整體畫面品質不錯
- 姿勢與狗的主體維持得比早期版本穩
- texture 感有被拉起來
- 但 texture reference 內的 content / subject 也被帶進輸出，造成不是單純把 texture 貼到狗身上，而是把 texture ref 的內容語義一起畫進來

### 17.1 這張圖的 method 記錄

這張圖對應的是 `group13_texture_recover / G13_E02 / seed 42`。

固定條件：

- `prompt`: `one dog, single dog, solo dog portrait`
- `negative_prompt`: 單狗約束版 negative prompt
- `geometry_mode`: `infer_original`
- `texture_mode`: `raw_gray`
- `attention_profile`: `default`
- `token_profile`: `default`

實際參數：

- `guidance_scale = 9.0`
- `color_scale = 1.85`
- `substract_scale = 0.25`
- `texture_scale = 1.15`
- `geometry_scale = 1.0`
- `geometry_sub_scale = 0.30`
- `texture_color_decouple = 0.08`
- `geometry_color_decouple = 0.50`
- `geometry_early_end = 0.08`
- `geometry_late_factor = 0.02`
- `texture_late_start = 0.48`
- `texture_early_factor = 0.32`
- `punish_weight = 0.0002`
- `punish_type = soft-weight`

### 17.2 問題成因判讀

這張圖會把 texture ref 的 content 一起畫出來，我目前判斷主因是：

- `raw_gray` 仍保留不少 texture reference 的主體結構與內容語義
- texture 介入時間偏早：`texture_late_start = 0.48`
- texture early influence 偏高：`texture_early_factor = 0.32`
- `geometry_sub_scale = 0.30` 偏保守，還不足以把 texture branch 裡的 shape / content 扣乾淨

這個問題不是 color branch 不夠強，也不是 geometry color 沒進 color stream。
相反地，這一輪應明確避免把 geometry 的 color 混入 color branch，因為那會把問題從「texture content 洩漏」變成「color 來源混濁」。

本輪原則：

- `color` 只維持 color ref
- `geometry` 只維持形狀與姿勢
- `texture` 只保留表面 pattern，不攜帶 texture ref 的 subject content

## 18. Texture Content Suppression 實驗設計

這一組實驗的目標是：

- 解 `texture content 蓋過狗`
- 同時明確避免 `color` 去吃 geometry 的色彩
- 保持目前 G13_E02 已經得到的畫質與單狗穩定性

### 18.1 設計原則

這一輪不做的事：

- 不把 geometry color 混進 color stream
- 不改 color ref 的來源
- 不先碰 residual geometry

這一輪只處理 texture content 洩漏：

- 強化 texture branch 對 geometry 的 subtraction
- 或把 texture 出現時間往後推
- 或把 texture mode 改成更偏材質的表示
- 再搭配更明確的 negative prompt 壓 texture subject content

### 18.2 新增 group15 的 5 組實驗

新的 group 名稱：

- `group15_texture_content_guard`

共同 baseline 直接繼承 `G13_E02`：

- `guidance_scale = 9.0`
- `geometry_scale = 1.0`
- `geometry_early_end = 0.08`
- `geometry_late_factor = 0.02`
- `color_scale = 1.85`
- `substract_scale = 0.25`
- `texture_scale = 1.15`
- `texture_late_start = 0.48`
- `texture_early_factor = 0.32`
- `texture_color_decouple = 0.08`
- `geometry_color_decouple = 0.50`
- `geometry_sub_scale = 0.30`
- `punish_weight = 0.0002`

共同 prompt：

- `one dog, single dog, solo dog portrait`

共同 negative prompt 額外強化：

- `copy texture reference subject`
- `same animal as texture reference`
- `same object as texture reference`
- `copy texture reference content`
- `texture reference composition`
- `non-dog subject`

五組實驗：

- `G15_E01`: `raw_gray`，只把 `geometry_sub_scale` 提到 `0.45`
- `G15_E02`: `raw_gray`，只把 `geometry_sub_scale` 提到 `0.60`
- `G15_E03`: 改成 `material_gray`，其餘維持 `G13_E02`
- `G15_E04`: `raw_gray`，改成 `texture_late_start = 0.52`，`texture_early_factor = 0.26`
- `G15_E05`: `raw_gray`，改成 `texture_late_start = 0.56`，`texture_early_factor = 0.22`

### 18.3 實驗優先順序

建議先跑順序：

1. `G15_E01`
2. `G15_E03`
3. `G15_E04`
4. `G15_E02`
5. `G15_E05`

原因：

- 先測 `geometry_sub_scale` 增強是否已足夠壓掉 texture content
- 再測 `material_gray` 是否能更乾淨地只留下材質
- 最後再用 schedule rollback 微調 texture 的介入時機

### 18.4 驗收標準

這組實驗的主要評分：

- `S1`: 單狗維持
- `S2`: texture 仍可見，但不能直接把 texture ref 的主體內容畫出來
- `S3`: geometry / pose 不崩
- `S4`: color 仍只由 color ref 主導，不被 geometry 染色

淘汰條件：

- texture content 仍明顯蓋過狗
- texture 雖在，但狗的姿勢明顯崩掉
- color 開始偏向 geometry input 的色彩
