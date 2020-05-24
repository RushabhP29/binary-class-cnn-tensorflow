import cv2
import tensorflow as tf
import os
import numpy as np
from build_model import model_tools

model = model_tools()
model_folder = 'checkpoints'
image = 'car2.jpg'
img = cv2.imread(image)
session = tf.Session()
img = cv2.resize(img,(100,100))
img = img.reshape(1,100,100,3)
labels = np.zeros((1, 2))


saver = tf.train.import_meta_graph(os.path.join(model_folder,'.meta'))

saver.restore(session,os.path.join(model_folder,'.\\'))

graph = tf.get_default_graph()

network = graph.get_tensor_by_name("add_4:0")

im_ph = graph.get_tensor_by_name("Placeholder:0")
label_ph = graph.get_tensor_by_name("Placeholder_1:0")

network = tf.nn.sigmoid(network)


feed_dict_testing = {im_ph: img, label_ph: labels}
result = session.run(network, feed_dict=feed_dict_testing)
print(result)
if result[0][0] > result[0][1]:
    print('Airplane')
else:
    print('Car')



