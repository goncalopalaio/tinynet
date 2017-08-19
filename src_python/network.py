import os
import numpy as np
import functools
import time
import tensorflow as tf
from scipy import misc
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

#
#
#
# todo: Take validation elements from the training set, not the test set
# todo: Implement dropout. Don't forget to set a placeholder value set to 1.0 during evaluation
#
#
#


dataset_path = "../../datasets/cifar/"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
DEBUG = 1

USE_SMALL_DATASET = False
SMALL_DATASET_PERCENT = 0.05

def save_image_gallery(array):
	nindex, height, width, intensity = array.shape
	w = 2
	h = 2
	gs = gridspec.GridSpec(w, h, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
	
	for x in range(w):
		for y in range(h):
			index = x+y*w
			im = array[index,:,:,:]
			ax = plt.subplot(gs[x,y])
			ax.imshow(im, cmap='gray')

	plt.savefig("gallery.png")
	print("Gallery saved")

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('[TIMEIT] %r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def get_dataset_folder(suffix_path):
	path = dataset_path + suffix_path
	print("Reading from ", path)
	class_name_list = []
	image_list = []

	filelist = os.listdir(path)
	for f in filelist:
		file = path + f
		class_name = f.replace(".","_").split("_")[1]
		
		if class_name == 'DS':
			print("Skipping: ", file)
			continue

		image_list.append(file)
		class_name_list.append(class_name)

	if DEBUG:
		classes = np.unique(np.array(class_name_list))
		print(classes)
		print("Number of classes: %d" % len(classes))
		print("Number of images: %d" % len(image_list))

	if USE_SMALL_DATASET:
		count = int(len(class_name_list) * SMALL_DATASET_PERCENT)
		print("\n\n\n\n\nWARNING: USE_SMALL_DATASET FLAG IS ON.\nRETURNING SUBSET OF THE DATASET\n\n\n\n")
		class_name_list = class_name_list[0:count]
		image_list = image_list[0:count]

	return class_name_list, image_list

def get_train():
	return get_dataset_folder("train/")

@timeit
def get_validation_and_test(only_test = False, only_validation = False):
	class_name_list, image_list = get_dataset_folder("test/")

	validation_classes = []
	validation_images = []

	print("Dividing into validation and test")
	count = len(class_name_list)
	validation_count = int(count * 0.05)
	print("From %d taking %d for validation" % (count, validation_count))

	ind_take = np.sort(np.random.choice(count, validation_count))

	# Let's not mess around with np arrays for now
	for ind in ind_take:
		validation_classes.append(class_name_list[ind])
		validation_images.append(image_list[ind])

	if only_validation:
		return validation_classes, validation_images

	ind_to_delete = ind_take[::-1] #reverse it
	for x in ind_to_delete:
		del class_name_list[x]
		del image_list[x]

	test_classes = class_name_list
	test_images = image_list

	if only_test:
		return test_classes, test_images		

	#todo ensure both validation and test have elements from the all classes or is this pretty much guaranteed?

	print("Total images: ", count)
	print("Test should have: ", count-validation_count)
	print("Validation should have: ", validation_count)
	print("Test: classes len: %d images len: %d" % (len(test_classes), len(test_images)))
	print("Validation: classes len: %d images len: %d" % (len(validation_classes), len(validation_images)))
	print("Validation has %d classes " % len(np.unique(np.array(validation_classes))))
	print("Test has %d classes " % len(np.unique(np.array(test_classes))))
	print("Validation: ", np.unique(np.array(validation_classes), return_counts=True))
	print("Test: ", np.unique(np.array(test_classes), return_counts=True))

	return validation_classes, validation_images, test_classes, test_images

def normalize_image(image):
	# todo: implement me properly
	norm = image / 255.0
	return norm

def load_set_element(class_name, image_file, index):
	mode_grayscale = 'L'
	mode_rgb = 'RGB'
	full_img = misc.imread(image_file, mode = mode_rgb)
	img = normalize_image(full_img)
	class_number = CLASSES.index(class_name)
	#print("index: %d \nclass: %s classnumber: %d \nfilename: %s, imageshape: %s" % (index, c, class_number, i, str(np.shape(img))))
	return class_number, img

def shuffle_dataset(class_list, image_list):
	indexes = np.random.permutation(len(class_list))
	assert len(indexes) == len(class_list)

	print("Shuffling dataset")
	c = []
	img = []
	for i in indexes:
		c.append(class_list[i])
		img.append(image_list[i])

	return c,img


def load_batch_data(class_list, image_list):
	classes = []
	images = np.empty(shape=(len(class_list), 32,32, 3))

	for i in range(0, len(class_list)):
		clazz, img = load_set_element(class_list[i], image_list[i], i)
		images[i,:,:,:] = img
		classes.append(clazz)

	classes = np.array(classes)
	#print("Batch shape: ", np.shape(images), " classes: ", np.shape(classes))

	return images, classes

def load_train_batch(class_list, image_list, batch_index_start, batch_index_end):
	# ensure we have the elements
	end = min(batch_index_end, len(class_list))
	classes = [class_list[ind] for ind in range(batch_index_start, end)]
	image_paths = [image_list[ind] for ind in range(batch_index_start, end)]
	
	return load_batch_data(classes, image_paths)

def evaluate(x_data, y_data, batch_size, accuracy_operation, x_placeholder, y_placeholder):
	total_examples = len(y_data)
	total_accuracy = 0
	
	sess = tf.get_default_session()
	for batch_start in range(0, total_examples, batch_size):
		batch_end = batch_start + batch_size

		batch_x, batch_y = x_data[batch_start:batch_end], y_data[batch_start:batch_end]
		accuracy_value = sess.run(accuracy_operation, feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y})
		#print("Batch start: %d Accuracy: %f" % (batch_start, accuracy_value))
		total_accuracy += (accuracy_value * len(batch_y))
	return total_accuracy / float(total_examples)

def layer_conv2d(x, _shape, _bias_count, mu = 0, sigma = 0.1):
	weights = tf.Variable(tf.truncated_normal(shape = _shape, mean = mu, stddev = sigma))
	bias = tf.Variable(tf.zeros(_bias_count))
	conv = tf.nn.conv2d(x, weights, strides = [1,1,1,1], padding = 'VALID') + bias
	return conv

def layer_matmul(x, _shape, _bias_count,  mu = 0, sigma = 0.1):
	w = tf.Variable(tf.truncated_normal(shape = _shape, mean = mu, stddev = sigma))
	b = tf.Variable(tf.zeros(_bias_count))
	return tf.matmul(x, w) + b

def network(x):
	mu = 0
	sigma = 0.1

	layer = layer_conv2d(x, (3,3,3,128), 128)
	layer = tf.nn.relu(layer)
	layer = layer_conv2d(x, (3,3,3,128), 128)
	layer = tf.nn.relu(layer)
	layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	#layer = tf.nn.dropout(layer, 0.25) # To use dropout don't forget to create a placeholder and set prob=1.0 during evaluation

	layer = layer_conv2d(layer, (2, 2, 128, 3), 3)
	layer = tf.nn.relu(layer)	
	layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

	layer = flatten(layer)
	out_dim = int(np.shape(layer)[1])
	layer = layer_matmul(layer, (out_dim, 10), 10)
	layer = tf.nn.relu(layer)

	return layer

def main():
	net_learning_rate = 0.001
	net_epochs = 5
	net_batch_size = 10
	net_dropout = 0.75

	train_class_metadata, train_image_metadata = get_train()
	validation_class_metadata, validation_image_metadata = get_validation_and_test(only_validation=True)

	print("\n\n######################## Running training ########################\n\n")
	print("Loading validation dataset")
	x_validation, y_validation = load_batch_data(validation_class_metadata, validation_image_metadata)
	save_image_gallery(x_validation)

	total_train_examples = len(train_class_metadata)
	print("Total Training examples: ", total_train_examples)

	x = tf.placeholder(tf.float32, (None, 32, 32, 3))
	y = tf.placeholder(tf.int32, (None))
	one_hot_y = tf.one_hot(y, 10)

	logits_out = network(x)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits_out, labels = one_hot_y)
	loss = tf.reduce_mean(cross_entropy)
	loss_minimization = tf.train.AdamOptimizer(learning_rate = net_learning_rate).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(logits_out,1), tf.argmax(one_hot_y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		train_class_metadata, train_image_metadata = shuffle_dataset(train_class_metadata, train_image_metadata)

		for epoch in range(net_epochs):
			# todo: shuffle the dataset
			
			for batch_start in range(0, total_train_examples, net_batch_size):
				batch_end = batch_start + net_batch_size

				#print("total: %d <- start: %d end %d" % (total_train_examples, batch_start, batch_end))

				batch_x, batch_y = load_train_batch(train_class_metadata, train_image_metadata, batch_start, batch_end)
				sess.run(loss_minimization, feed_dict={x: batch_x, y: batch_y})

			accuracy_validation = evaluate(x_validation, y_validation, net_batch_size, accuracy, x, y)
			print("\nEpoch %d validation accuracy: %f" % (epoch, accuracy_validation))

		saver.save(sess, 'network-v1')

	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))

		test_class_metadata, test_image_metadata = get_validation_and_test(only_test=True)
		x_test, y_test = load_batch_data(test_class_metadata, test_image_metadata)
		
		accuracy_test = evaluate(x_test, y_test, net_batch_size, accuracy, x, y)
		print("Number of test elements: %d \nTest accuracy: %f" % (len(test_class_metadata), accuracy_test))

if __name__ == '__main__':
	main()