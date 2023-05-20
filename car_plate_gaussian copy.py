from ultralytics import YOLO

# Load a model
model = YOLO("./yolov8n.yaml")  # build a new model from scratch
model = YOLO("C:/Users/jacky/Desktop/_Recognize/detect/train/weights/last.pt")  # load a pretrained model (recommended for training)

# 預測
results = model("C:/Users/jacky/Desktop/_Recognize/Cars-Plate.v1i.yolov8/test/images/Cars182_png.rf.c1031e740bf488fe8528f5e9d050ebaf.jpg")

# 顯示物件類別
print(results[0].boxes.cls)
print()
# 顯示物件座標
print(results[0].boxes.xyxy)
#座標顯示
import numpy as np
boxes = results[0].boxes.xyxy.numpy()
x1, y1, x2, y2 = boxes[0]
print(f"左上角座標：({x1}, {y1})")
print(f"右下角座標：({x2}, {y2})")

#存檔
from ultralytics import YOLO
from PIL import Image
import cv2
im1 = Image.open("C:/Users/jacky/Desktop/_Recognize/Cars-Plate.v1i.yolov8/test/images/Cars182_png.rf.c1031e740bf488fe8528f5e9d050ebaf.jpg")
# save=True：存檔
results = model.predict(source=im1, save=True)

import os
import cv2
#找出所有runs/detect以下所有圖片最新的檔案
predict_folder = "C:/Users/jacky/Desktop/_Recognize/runs/detect"
folders = os.listdir(predict_folder)
folders = [f for f in folders if f.startswith("predict")]
latest_file = None
latest_time = 0

for folder in folders:
    files = os.listdir(os.path.join(predict_folder, folder))
    files = [f for f in files if f.endswith(".jpg")]
    if len(files) == 0:
        continue
    file_path = os.path.join(predict_folder, folder, files[-1])
    file_time = os.path.getmtime(file_path)
    if file_time > latest_time:
        latest_file = file_path
        latest_time = file_time

if latest_file is not None:
    im = cv2.imread(latest_file)


#找到座標
import cv2
from PIL import Image
# im = cv2.imread("D:/ultralytics-main/ultralytics-main/car777.jpg")#已標記之車牌
x1, y1, x2, y2 = boxes[0]
x1, y1, x2, y2 = map(int, boxes[0])
roi = im[y1:y2, x1:x2]

# 馬賽克效果
level = 15  # 縮小比例 ( 可當作馬賽克的等級 )
h, w = roi.shape[:2]
mosaic_h = int(h/level)   # 按照比例縮小後的高度 ( 使用 int 去除小數點 )
mosaic_w = int(w/level)   # 按照比例縮小後的寬度 ( 使用 int 去除小數點 )
mosaic_roi = cv2.resize(roi, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)   # 根據縮小尺寸縮小
mosaic_roi = cv2.resize(mosaic_roi, (w, h), interpolation=cv2.INTER_NEAREST) # 放大到原本的大小
# 將處理後的圖像放回原圖
im[y1:y2, x1:x2] = mosaic_roi


# cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2) #框出車牌位置
cv2.imwrite("C:/Users/jacky/Desktop/_Recognize/detect/output4.jpg", im)

# cv2.imshow("image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 取得矩形區域的子圖像
#證明框出位置是車牌的位置

