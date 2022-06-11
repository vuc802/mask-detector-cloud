try:
    import unzip_requirements
except ImportError:
    pass
import json
from io import BytesIO
import os
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

import base64

import boto3
import numpy as np


def img_to_base64_str(img):
	buffered = BytesIO()
	img.save(buffered, format="jpg")
	buffered.seek(0)
	img_byte = buffered.getvalue()
	img_str = "data:examples/jpg;base64," + base64.b64encode(img_byte).decode()
	return img_str
    
def load_models(s3, bucket):
	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = s3.get_object(Bucket=bucket, Key=f"deploy.prototxt")

	weightsPath = s3.get_object(Bucket=bucket, Key=f"res10_300x300_ssd_iter_140000.caffemodel")

	models['fece_net'] = cv2.dnn.readNet(prototxtPath, weightsPath)
	print("[INFO] loading mask detector model...")
	
	
	
	mask_obj = s3.get_object(Bucket=bucket, Key=f"mask_detectot.model")
	models['mask_net'] = load_model(mask_obj )
	return net

gpu = -1

s3 = boto3.client("s3")
bucket = "testmaskdetection"

models = load_models(s3, bucket)
print(f"models loaded ...")


def lambda_handler(event, context):
	"""
	lambda handler to execute the image transformation
	"""
	# warming up the lambda
	if event.get("source") in ["aws.events", "serverless-plugin-warmup"]:
		print("Lambda is warm!")
		return {}

	data = json.loads(event["body"])
	print("data keys :", data.keys())
	image = data["image"]
	image = image[image.find(",") + 1 :]
	dec = base64.b64decode(image + "===")
	image = Image.open(BytesIO(dec))
	image = image.convert("RGB")

	# load the model with the selected style
	load_size = int(data["load_size"])
	face_net = models['face_net']
	mask_net = models['mask_net']
	
	# resize the image
	h = image.size[0]
	w = image.size[1]
	ratio = h * 1.0 / w
	if ratio > 1:
		h = load_size
		w = int(h * 1.0 / ratio)
	else:
		w = load_size
		h = int(w * ratio)

	image = image.resize((h, w), Image.BICUBIC)
	image = np.asarray(image)
	

	blob = cv2.dnn.blobFromImage(image, 1.0, (h, w),
	(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	face_net.setInput(blob)
	detections = face_net.forward()


	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

		output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)

		output_image = Image.fromarray(output_image)

	#
	result = {"output": img_to_base64_str(output_image)}

	return {
	"statusCode": 200,
	"body": json.dumps(result),
	"headers": {
	    "Content-Type": "application/json",
	    "Access-Control-Allow-Origin": "*",
	},
	}
