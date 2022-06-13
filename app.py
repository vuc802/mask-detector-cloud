import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import os
import boto3
import numpy as np
import numpy as np
import PIL.Image as Image
from io import BytesIO
import base64

s3 = boto3.client("s3")
bucket = "testmaskdetection"


# def readImageFromBucket(key, bucket_name):
# 	bucket = s3.Bucket(bucket_name)
# 	object = bucket.Object(key)
# 	response = object.get()
# 	return Image.open(response['Body'])


def img_to_base64_str(img):
	buffered = BytesIO()
	img.save(buffered, format="jpg")
	buffered.seek(0)
	img_byte = buffered.getvalue()
	img_str = "data:examples/jpg;base64," + base64.b64encode(img_byte).decode()
	return img_str

def load_models(s3, bucket):
	# load our serialized face detector model from disk
	models = {}
	prototxtPath = s3.get_object(Bucket=bucket, Key=f"deploy.prototxt")

	weightsPath = s3.get_object(Bucket=bucket, Key=f"res10_300x300_ssd_iter_140000.caffemodel")

	models['fece_net'] = cv2.dnn.readNet(prototxtPath, weightsPath)
	

	mask_obj = s3.get_object(Bucket=bucket, Key=f"mask_detectot.model")
	models['mask_net'] = load_model(mask_obj )
	print("[INFO] loading finish")
	return models

def lambda_handler(event, context):
	"""
	lambda handler to execute the image transformation
	"""
	bucket_name = event['Records'][0]['s3']['bucket']['name']
	key = event['Records'][0]['s3']['object']['key']
	
	models = load_models(s3, bucket)
	print(f"models loaded ...")
	# image = readImageFromBucket(key, bucket_name)
	
	# extracting the image form the payload and converting it to PIL format
	data = json.loads(event["body"])
	print("data keys :", data.keys())
	image = data["image"]
	image = image[image.find(",")+1:]
	dec = base64.b64decode(image + "===")
	image = Image.open(BytesIO(dec))
	image = image.convert("RGB")
	
	# load the model 
	data = json.loads(event["body"])
	face_net = models['face_net']
	mask_net = models['mask_net']
	
	# resize the image
	h = image.size[0]
	w = image.size[1]
	
	image = image.resize((h, w))
	image = np.array(image)
	

	blob = cv2.dnn.blobFromImage(image, 1.0, (h, w), (104.0, 177.0, 123.0))

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
		if confidence > 0.5:
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
			(mask, withoutMask) = mask_net.predict(face)[0]

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

	# output_image = Image.fromarray(image)
	output_image = image.numpy()
	output_image = np.uint8(output_image)
	output_image = Image.fromarray(output_image)
	
	# convert the PIL image to base64
	result = {
		"output": img_to_base64_str(output_image)
	}

	# send the result back to the client inside the body field
	return {
		"statusCode": 200,
		"body": json.dumps(result),
		"headers": {
			'Content-Type': 'application/json',
			'Access-Control-Allow-Origin': '*'
		}
	}

