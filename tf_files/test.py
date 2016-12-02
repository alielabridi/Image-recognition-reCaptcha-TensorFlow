# Creator: Ali ELABRIDI and Brandon Crane
# should install pip install icrawler
# https://pypi.python.org/pypi/icrawler/0.1.5
from icrawler.examples import GoogleImageCrawler
import os
import sys
import tensorflow as tf, sys
import subprocess
import commands


label = sys.argv[1]

directory = "/tf_files/flower_photos"+ "/" +label

if not os.path.exists(directory):
	#the label does not exists
	os.mkdir(directory)
	google_crawler = GoogleImageCrawler(directory)
	google_crawler.crawl(keyword=label, offset=0, max_num=40,
	                     date_min=None, date_max=None, feeder_thr_num=1,
	                     parser_thr_num=1, downloader_thr_num=4,
	                     min_size=(200,200), max_size=None)

for root, dirs, files in os.walk(directory):
	for currentFile in files:
		ext = '.jpg'
		s = commands.getstatusoutput('file ' + directory + '/' + currentFile)[1]
		if ((s.find('JPEG image data') == -1) or (not currentFile.lower().endswith(ext))):
			os.remove(os.path.join(root,  directory + '/' + currentFile))

# then train the NN
subprocess.call("python /tensorflow/tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/flower_photos", shell=True)

minIndex = ""
# we start by 2 because it is where the compared images are
for x in sys.argv[2:]:
	print x
	image_path = x

	# Read in the image_data
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line
	                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

	# Unpersists graph from file
	with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    _ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	    # Feed the image_data as input to the graph and get first prediction
	    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	    predictions = sess.run(softmax_tensor, \
	             {'DecodeJpeg/contents:0': image_data})

	    # Sort to show labels of first prediction in order of confidence
	    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

	    for node_id in top_k:
	        human_string = label_lines[node_id]
	        score = predictions[0][node_id]
	        print('%s is a %s (score = %.5f)' % (x,human_string, score))

	        if(human_string == label and x == sys.argv[2]):
	        	minIndex = x
	        	minScore = score

	        if(human_string == label and x!= sys.argv[2] and minScore > score):
	        	minIndex = x
	        	minScore = score

print "The one that is not a "+label+ " is: "+ minIndex
