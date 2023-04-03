from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont

model = None
test_image = "own_test_image\Video_640_segment.png"

def load_model():

    global model
    rf = Roboflow(api_key="x6vyQ8ceJzvDxz1zqxkv")
    project = rf.workspace("car-plate-o3kq6").project("cars-plate-yzoqy")
    model = project.version(1).model

def resize_image(image_path):

    image = Image.open(image_path)
    image = image.resize((640, 480))
    image.save("Video_640_segment.png")

    return

def predict_image(image_path):

    # visualize your prediction
    predict_image = model.predict(image_path, confidence=40, overlap=30)

    print(predict_image.json())

    if len(predict_image.json()['predictions']) == 0:
        return False

    predict_image.save(image_path + ".jpg")

    return predict_image

def draw_boxs(image_path, predict_result):

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if predict_result == False:
        print("No object detected!")
        return image

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
        image.paste(button_img, (int(x1), int(y1) - 25))

    return image

if __name__ == "__main__":
    
    #resize_image(test_image)
    load_model()
    image = draw_boxs(test_image, predict_image(test_image))
    
    image.show()