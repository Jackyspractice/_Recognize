from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import tqdm
import numpy
import cv2

model = None
reslution = (640, 640)
fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
out = cv2.VideoWriter('video.mp4', fourcc, 30.0, reslution)
progress = tqdm.tqdm(total=186)


test_image = "own_test_image\Cars361.png"
video_path = "own_test_image\Testvideo.mp4"

cap = cv2.VideoCapture(video_path)
#cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
#cap = cv2.VideoCapture(0)

def load_model():

    global model
    rf = Roboflow(api_key="x6vyQ8ceJzvDxz1zqxkv")
    project = rf.workspace("car-plate-o3kq6").project("cars-plate-yzoqy")
    model = project.version(1).model

def predict_frame(frame):

    # visualize your prediction
    predict_image = model.predict(frame, confidence=0, overlap=30)

    #print(predict_image.json())

    if len(predict_image.json()['predictions']) == 0:
        return False

    #predict_image.save(frame + ".jpg")

    return predict_image

def draw_boxs(frame, predict_result):

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    if predict_result == False:
        #print("No object detected!")
        return frame

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

    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":

    load_model()

    print("start predicting...")

    while cap.isOpened():
        
        ret, frame = cap.read()

        if ret == False:
            break

        frame = cv2.resize(frame, (640, 640))

        image = draw_boxs(frame, predict_frame(frame))

        #cv2.imshow('predict', image)
        out.write(image)
        progress.update(1)

        #if cv2.waitKey(1) == ord('q'):
            #break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Release Finish!")