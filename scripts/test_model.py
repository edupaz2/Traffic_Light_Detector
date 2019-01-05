

import tensorflow as tf
import cv2
import numpy as np


if __name__ == '__main__':

	model_file = 'models/output_graph.pb'

	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	print("Operations: {0}".format(graph.get_operations()))

	input_operation = graph.get_operation_by_name('import/input')
	output_operation = graph.get_operation_by_name('import/final_result')

	with tf.Session(graph=graph) as sess:
		input_h = 224
		input_w = 224
		input_m = 0
		input_s = 255

		image = cv2.imread('train_pics/green/001.jpg')
		image = cv2.resize(image, dsize=(input_h, input_w), interpolation=cv2.INTER_CUBIC)
		np_image = np.asarray(image)
		dims_expander = tf.expand_dims(np_image, 0)
		resized = tf.image.resize_bilinear(dims_expander, [input_h, input_w])
		normalized = tf.divide(tf.subtract(resized, [input_m]), [input_s])

		tensor = sess.run(normalized)
		results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})

		results = np.squeeze(results)
		top_k = results.argsort()[-5:][::-1]
		print(top_k)
		print(results)