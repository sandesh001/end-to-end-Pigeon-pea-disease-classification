from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np

from model.main import densenet_121
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

app = Flask(__name__)

img_size = 224
num_classes = 5
input_shape = (img_size, img_size, 3)

model = densenet_121(input_shape, num_classes)
model.load_weights('model/weights/densenet_121.h5')
data_generator = ImageDataGenerator(rescale=1./255) 


label_map = ["Dry root rot", "Healthy", "Phytophthora blight", "Sterility mosaic disease", "wilt"]


dic = {0:"Dry root rot", 1:"Healthy", 2:"Phytophthora blight", 3:"Sterility mosaic disease", 4:"wilt"}

def predict(img_path):
    img = load_img(img_path, target_size=input_shape)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = data_generator.flow(array)[0]
    pred = model.predict(array)
    return label_map[int(np.argmax(pred))]

def predict_label(img_path):
	i = image.load_img(img_path, target_size=input_shape)
	i = image.img_to_array(i)/255.0
	i = np.expand_dims(i, axis=0)
	p = model.predict(i)
	print(p.shape)
	return dic[p[0]]

# routes
@app.route('/',methods = ['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		print(img_path)
		p = predict(img_path)
		#p = predict_label(img_path)
	return render_template("home.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	app.run(debug = True)
