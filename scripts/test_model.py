

import tensorflow as tf
import cv2
import numpy as np
import time
import argparse
import glob

def load_and_prepare_image(image, img_size=224):
    """
    :param filename:
    :param img_size:
    :return:
    """
    img = cv2.resize(image, dsize=(img_size, img_size))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)

    # Preprocess
    img = (img - 127.) / 127.

    return img

def load_image_tensor(image, input_h, input_w, input_m, input_s):
	dims_expander = tf.expand_dims(image, 0)
	resized = tf.image.resize_bilinear(dims_expander, [input_h, input_w])
	normalized = tf.divide(tf.subtract(resized, [input_m]), [input_s])
	return normalized

if __name__ == '__main__':

	model_file = 'models/mobilenet_1.0_224.site.pb'
	labels = ['green', 'none', 'red', 'yellow']

	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", help="dir of images to be processed")
	args = parser.parse_args()

	dir_name = 'test_pics'
	if args.dir:
		dir_name = args.dir

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
		input_h = 224
		input_w = 224
		input_m = 0
		input_s = 255

		l = glob.glob(dir_name + '/**/*.jpg')
		for f in l:
			img = cv2.imread(f)
			tensor = sess.run(load_image_tensor(img, input_h, input_w, input_m, input_s))
			results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
			results = np.squeeze(results)
			top_k = results.argsort()[-5:][::-1]
			print('File: {0} Time: {1}'.format(f, time.time()-tt))
			print('   Result: {0}'.format(labels[top_k[0]]))
			print('      Results: {0}'.format(['{0}: {1}'.format(labels[i], results[i]) for i in top_k]))
