import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("traffic_sign_model.h5")

# Label names
labels = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End of Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Overtaking",
    10: "No Overtaking (Trucks)",
    11: "Right-of-way at Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Traffic Both Ways",
    16: "No Trucks",
    17: "No Entry",
    18: "Danger",
    19: "Bend Left",
    20: "Bend Right",
    21: "Bend",
    22: "Uneven Road",
    23: "Slippery Road",
    24: "Road Narrows",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Restrictions",
    33: "Turn Right",
    34: "Turn Left",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout",
    41: "End of No Overtaking",
    42: "End of No Overtaking (Trucks)"
}


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)
    return img

# Test image # put any traffic sign image here
image_path = r"D:\Traffic Sign Recognition\DataSet\trafficSign\42.png"   # put any traffic sign image here
img = preprocess(image_path)

prediction = model.predict(img)
class_id = np.argmax(prediction)

print("Predicted Sign ID:", class_id)
print("Sign Meaning:", labels.get(class_id, "Unknown Sign"))
