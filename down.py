from datasets import load_dataset
import os
import io
from PIL import Image  # 記得匯入 PIL 處理圖片

# 1. 載入資料集
dataset = load_dataset("parquet", data_files="wikiart_dataset/data/*.parquet", split="train")
# 2. 建立存放圖片的資料夾
output_folder = "wikiart_extracted_images"
os.makedirs(output_folder, exist_ok=True)

# 3. 處理並儲存圖片
for i in range(len(dataset)):
    img_data = dataset[i]['image']
    
    # 檢查 img_data 是否為字典，且裡面有 'bytes' 資料
    if isinstance(imgdata, dict) and 'bytes' in img_data:
        # 將二進位位元組 (bytes) 轉換回 PIL 圖片物件
        image_bytes = img_data['bytes']
        img = Image.open(io.BytesIO(image_bytes))
    else:
        # 如果它已經是正常的圖片物件，就直接使用
        img = img_data
        
    # 存檔
    file_path = os.path.join(output_folder, f"artwork_{i}.jpg")
    img.save(file_path)
    print(f"已儲存: {file_path}")