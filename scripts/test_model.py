

import tensorflow as tf
import cv2
import numpy as np
import time


if __name__ == '__main__':

	model_file = 'models/mobilenet_1.0_224.pb'
	labels = ['green', 'none', 'red', 'yellow']

	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	input_operation = graph.get_operation_by_name('import/input')
	output_operation = graph.get_operation_by_name('import/final_result')

	tt = t = time.time()
	with tf.Session(graph=graph) as sess:
		newt = time.time()
		print("Time1: {0}".format(newt-t))
		t=newt
		input_h = 224
		input_w = 224
		input_m = 0
		input_s = 255

		image = cv2.imread('test_pics/001.png')
		newt = time.time()
		print("Time2: {0}".format(newt-t))
		t=newt
		#image = cv2.resize(image, dsize=(input_h, input_w), interpolation=cv2.INTER_CUBIC)
		np_image = np.asarray(image)
		dims_expander = tf.expand_dims(np_image, 0)
		resized = tf.image.resize_bilinear(dims_expander, [input_h, input_w])
		normalized = tf.divide(tf.subtract(resized, [input_m]), [input_s])
		
		newt = time.time()
		print("Time3: {0}".format(newt-t))
		t=newt
		
		tensor = sess.run(normalized)
		newt = time.time()
		print("Time4: {0}".format(newt-t))
		t=newt
		
		results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
		newt = time.time()
		print("Time5: {0}".format(newt-t))
		t=newt
		
		results = np.squeeze(results)
		top_k = results.argsort()[-5:][::-1]
		print("Total: {0}. Result: {1}".format(time.time() - tt, ['{0}-{1}'.format(labels[i], results[i]) for i in top_k]))