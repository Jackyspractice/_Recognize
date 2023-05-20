import cv2
import threading
from queue import Queue
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import tqdm
import numpy

model = None
reslution = (640, 640)
fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
out = cv2.VideoWriter('video4.mp4', fourcc, 30.0, reslution)
progress = tqdm.tqdm(total=186)

class VideoReader:
    def __init__(self, filename, queue):
        self.filename = filename
        self.queue = queue
        self.cap = cv2.VideoCapture(self.filename)
        
    def read(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.queue.put(frame)
        self.cap.release()

class VideoPlayer:
    def __init__(self, queue):
        self.queue = queue
        self.stopped = False
        
    def play(self):
        while not self.stopped:

            if not self.queue.empty():

                frame = self.queue.get()
                frame = cv2.resize(frame, (640, 640))
                image = self.draw_boxs(frame, self.predict_frame(frame))

                out.write(image)
                progress.update(1)
                #cv2.imshow("Video", image)
                cv2.waitKey(1)
                self.queue.task_done()

    def Stop(self):
        print("Stop!")
        self.stopped = True

    def predict_frame(self, frame):

        # visualize your prediction
        predict_image = model.predict(frame, confidence=20, overlap=30)

        #print(predict_image.json())

        if len(predict_image.json()['predictions']) == 0:
            return False

        #predict_image.save(frame + ".jpg")

        return predict_image

    def draw_boxs(self, frame, predict_result):

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        if predict_result == False:
            #print("No object detected!")
            return frame
        else:
            print("Object detected!")

        for box in predict_result:
            
            color = "#4892EA"
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2
            
            draw.rectangle([
                x1, y1, x2, y2
            ], outline=color, width=3)

            # find predicted object's class & confidence
            text = box['class'] + " " + str(box["confidence"])[0:4]
            text_size = font.getsize(text)
            
            # set button size + 10px margins
            button_size = (text_size[0]+20, text_size[1]+20)
            button_img = Image.new('RGBA', button_size, color)

            # put text on button
            button_draw = ImageDraw.Draw(button_img)

            # to customize the text color for object's label: update "fill" value
            button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

            # put button on source image in position (0, 0)
            img.paste(button_img, (int(x1), int(y1) - 25))

            #print("#################Before Type is", type(img))
            #img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
            img = numpy.array(img)
            #integer_array = img.astype(int)
            #print("#################After Type is", type(img))
            
            img = self.mosaic(int(x1), int(x2), int(y1), int(y2), img)
            img = Image.fromarray(img)


        img = numpy.array(img)
        return img
    
    def mosaic(self, x1, x2, y1, y2, image):

        roi = image[y1:y2, x1:x2]
        level = 15  
        h, w = roi.shape[:2]
        mosaic_h = int(h/level)   
        mosaic_w = int(w/level)
        #print("mosaic_h", mosaic_h)
        #print("mosaic_w", mosaic_w)
        mosaic_roi = cv2.resize(roi, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)  
        mosaic_roi = cv2.resize(mosaic_roi, (w, h), interpolation=cv2.INTER_NEAREST) 
        
        image[y1:y2, x1:x2] = mosaic_roi

        return image
             
def load_model():

    global model
    rf = Roboflow(api_key="x6vyQ8ceJzvDxz1zqxkv")
    project = rf.workspace("car-plate-o3kq6").project("cars-plate-yzoqy")
    model = project.version(4).model

if __name__ == '__main__':

    load_model()

    filename = "own_test_image\Testvideo3.mp4"
    #filename = "hub2.mp4"
    queue = Queue(maxsize=10)
    
    # 創建一個影片讀取執行緒
    reader = VideoReader(filename, queue)
    t1 = threading.Thread(target=reader.read)
    
    # 創建一個影片顯示執行緒
    player = VideoPlayer(queue)
    t2 = threading.Thread(target=player.play)
    
    # 啟動執行緒
    t1.start()
    t2.start()
    
    # 等待影片處理完成
    t1.join()
    queue.join()

    if queue.empty():
        player.Stop()
        progress.close()
    

    print("Finish!")
    
    # 釋放資源
    out.release()
    cv2.destroyAllWindows()
