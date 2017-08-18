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
dataset_path = "../../datasets/cifar/"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
DEBUG = 1

USE_SMALL_DATASET = True

def save_image_gallery(array):
	nindex, height, width, intensity = array.shape
	w = 8
	h = nindex // w
	gs = gridspec.GridSpec(w, h, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
	
	for x in range(w):
		for y in range(h):
			im = array[x+y*w,:,:,-1]
			ax = plt.subplot(gs[x,y])
			ax.imshow(im, cmap='gray')

	plt.savefig("gallery.png")
	print "Gallery saved"

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '[TIMEIT] %r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed

def get_dataset_folder(suffix_path):
	path = dataset_path + suffix_path
	print "Reading from ", path
	class_name_list = []
	image_list = []

	filelist = os.listdir(path)
	for f in filelist:
		file = path + f
		class_name = f.replace(".","_").split("_")[1]
		
		if class_name == 'DS':
			print "Skipping: ", file
			continue

		image_list.append(file)
		class_name_list.append(class_name)

	if DEBUG:
		classes = np.unique(np.array(class_name_list))
		print classes
		print "Number of classes: %d" % len(classes)
		print "Number of images: %d" % len(image_list)

	if USE_SMALL_DATASET:
		count = int(len(class_name_list) * 0.1)
		print "\n\n\n\n\nWARNING: USE_SMALL_DATASET FLAG IS ON.\nRETURNING SUBSET OF THE DATASET\n\n\n\n"
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

	print "Dividing into validation and test"
	count = len(class_name_list)
	validation_count = int(count * 0.05)
	print "From %d taking %d for validation" % (count, validation_count)

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

	#todo maybe we should check if all of this is correct by asserting there's no common elements?
	#todo ensure both validation and test have elements from the all classes or is this pretty much guaranteed?

	print "Total images: ", count
	print "Test should have: ", count-validation_count
	print "Validation should have: ", validation_count
	print "Test: classes len: %d images len: %d" % (len(test_classes), len(test_images))
	print "Validation: classes len: %d images len: %d" % (len(validation_classes), len(validation_images))
	print "Validation has %d classes " % len(np.unique(np.array(validation_classes)))
	print "Test has %d classes " % len(np.unique(np.array(test_classes)))
	print "Validation: ", np.unique(np.array(validation_classes), return_counts=True)
	print "Test: ", np.unique(np.array(test_classes), return_counts=True)

	return validation_classes, validation_images, test_classes, test_images

# note: let's keep it simple for now and work with grayscale images
def normalize_image(image):
	
	# todo: Actually transform it to grayscale and normalize it

	grayscale = image[:,:,0] / 255.0
	return grayscale

def load_set_element(class_name, image_file, index):
	full_img = misc.imread(image_file)
	img = normalize_image(full_img)
	class_number = CLASSES.index(class_name)
	#print "index: %d \nclass: %s classnumber: %d \nfilename: %s, imageshape: %s" % (index, c, class_number, i, str(np.shape(img)))
	return class_number, img

def load_batch_data(class_list, image_list):
	classes = []
	images = np.empty(shape=(len(class_list), 32,32, 1))

	for i in range(0, len(class_list)):
		clazz, img = load_set_element(class_list[i], image_list[i], i)
		images[i,:,:,0] = img
		classes.append(clazz)

	classes = np.array(classes)
	#print "Batch shape: ", np.shape(images), " classes: ", np.shape(classes)

	return images, classes

def load_train_batch(class_list, image_list, batch_index_start, batch_index_end):
	classes = [class_list[ind] for ind in range(batch_index_start, batch_index_end)]
	image_paths = [image_list[ind] for ind in range(batch_index_start, batch_index_end)]
	
	return load_batch_data(classes, image_paths)

def evaluate(x_data, y_data, batch_size, accuracy_operation, x_placeholder, y_placeholder):
	total_examples = len(y_data)
	total_accuracy = 0
	
	sess = tf.get_default_session()
	for batch_start in range(0, total_examples, batch_size):
		batch_end = batch_start + batch_size

		batch_x, batch_y = x_data[batch_start:batch_end], y_data[batch_start:batch_end]
		accuracy_value = sess.run(accuracy_operation, feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y})
		#print "Batch start: %d Accuracy: %f" % (batch_start, accuracy_value)
		total_accuracy += (accuracy_value * len(batch_y))
	return total_accuracy / float(total_examples)

def network(x):
	mu = 0
	sigma = 0.1

	# convolution layer 1
	conv1_w = tf.Variable(tf.truncated_normal(shape = (3,3,1,6), mean = mu, stddev = sigma), name='conv1_w')
	conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
	conv1 = tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b
	# activation
	relu1 = tf.nn.relu(conv1)
	# pooling
	maxpool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

	# note: Removing this layer for now
	if False:
		# convolution layer 2
		conv2_w = tf.Variable(tf.truncated_normal(shape = (1,6,6,2), mean = mu, stddev = sigma), name='conv2_w')
		#conv2_b = tf.Variable(tf.zeros(6), name='conv2_b')
		conv2 = tf.nn.conv2d(maxpool1, conv2_w, strides = [1,1,1,1], padding = 'VALID') #+ conv2_b
		# activation
		relu2 = tf.nn.relu(conv2)
		# pooling
		maxpool2 = tf.nn.max_pool(relu2, ksize=[1,5,5,1], strides = [1,3,3,1], padding = 'VALID')



		# fully connected
		fc2_in = flatten(maxpool2)
	else:
		# fully connected
		fc2_in = flatten(maxpool1)
	
	fc2_in = tf.nn.dropout(fc2_in, 0.75)

	fc2_w = tf.Variable(tf.truncated_normal(shape = (1350,10), mean = mu, stddev = sigma), name = 'fc2_w')
	fc2_b = tf.Variable(tf.zeros(10), name = 'fc2_b')
	logits = tf.matmul(fc2_in, fc2_w) + fc2_b
	return logits

def main():
	net_learning_rate = 0.001
	net_epochs = 40 # if USE_SMALL_DATASET else 10
	net_batch_size = 4 # if USE_SMALL_DATASET else 200
	net_dropout = 0.75

	train_class_metadata, train_image_metadata = get_train()
	validation_class_metadata, validation_image_metadata = get_validation_and_test(only_validation=True)

	print "\n\n######################## Running training ########################\n\n"
	print "Loading validation dataset"
	x_validation, y_validation = load_batch_data(validation_class_metadata, validation_image_metadata)
	save_image_gallery(x_validation)

	total_train_examples = len(train_class_metadata)
	print "Total Training examples: ", total_train_examples
	
	x = tf.placeholder(tf.float32, (None, 32, 32, 1))
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

		for epoch in range(net_epochs):
			# todo: shuffle the dataset
			
			for batch_start in range(0, total_train_examples, net_batch_size):
				batch_end = batch_start + net_batch_size

				#print "total: %d <- start: %d end %d" % (total_train_examples, batch_start, batch_end)

				batch_x, batch_y = load_train_batch(train_class_metadata, train_image_metadata, batch_start, batch_end)
				sess.run(loss_minimization, feed_dict={x: batch_x, y: batch_y})

			accuracy_validation = evaluate(x_validation, y_validation, net_batch_size, accuracy, x, y)
			print "\nEpoch %d validation accuracy: %f" % (epoch, accuracy_validation)

		saver.save(sess, 'network-v1')

	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))

		test_class_metadata, test_image_metadata = get_validation_and_test(only_test=True)
		x_test, y_test = load_batch_data(test_class_metadata, test_image_metadata)
		
		accuracy_test = evaluate(x_test, y_test, net_batch_size, accuracy, x, y)
		print "Number of test elements: %d \nTest accuracy: %f" % (len(test_class_metadata), accuracy_test)

if __name__ == '__main__':
	main()