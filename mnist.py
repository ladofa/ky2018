import tensorflow as tf
import numpy as np
import cv2
import os
import random

#데이터 로드
def load_data(path):
    data = []

    for root, dirs, files in os.walk(path):
        if len(files) == 0:
            continue
        
        label = int(root[-1])

        for file in files:
            path = root + '/' + file
            image = cv2.imread(path)
            sample = {'label':label, 'image':image}
            data.append(sample)

    return data

def get_random(data, batch_size):
    selected = random.sample(data, batch_size)

    images = []
    labels = []

    for sample in selected:
        images.append(sample['image'])
        labels.append(sample['label'])

    return images, labels


def simple_network(x):
    x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
    x = tf.layers.dense(x, 100, activation=tf.nn.sigmoid)
    x = tf.layers.dense(x, 10)
    return x

def conv_network(x):
    x = tf.layers.conv2d(x, 8, 3, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 3, 2, padding='same') #reduce to 14 x 14
    x = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 3, 2, padding='same') #reduce to 7 x 7
    x = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 7, 1) #reduce to 1 x 1
    x = tf.reshape(x, [-1, 32])
    x = tf.layers.dense(x, 10)
    return x


data = load_data('mnist_mini/mini50')
train_count = int(len(data) * 0.8)
train_data = data[:train_count]
valid_data = data[train_count:]


_images = tf.placeholder(tf.uint8, [None, 28, 28, 3], name='input')
_labels = tf.placeholder(tf.int32, [None], 'label')

floated = tf.cast(_images, tf.float32) / 128 - 1
logits = conv_network(floated)
loss = tf.losses.sparse_softmax_cross_entropy(_labels, logits)

_sum_train_loss = tf.placeholder(tf.float32, [])
_sum_valid_loss = tf.placeholder(tf.float32, [])
sum_train_loss = tf.summary.scalar('loss', _sum_train_loss)
sum_valid_loss = tf.summary.scalar('loss', _sum_valid_loss)
sum_op = tf.summary.merge([sum_train_loss, sum_valid_loss])

gs = tf.Variable(0, trainable=False)

sess = tf.Session()
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss, global_step=gs)

sess.run(tf.global_variables_initializer())
file_writer = tf.summary.FileWriter('logs', sess.graph)

saver = tf.train.Saver()
#saver.restore(sess, 'training/simple.ckpt-10000')

print('학습 시작..')
while True:
    images, labels = get_random(train_data, 8)
    _, eval_loss, eval_gs = sess.run([train_op, loss, gs], feed_dict={_images:images, _labels:labels})
    print(eval_gs, eval_loss)
    summary = sess.run(sum_train_loss, feed_dict={_sum_train_loss:eval_loss})
    file_writer.add_summary(summary, eval_gs)

    if eval_gs % 1000 == 0:
        #모든 validation 데이터에 대해 검증 수행
        valid_loss = 0
        for sample in valid_data:
            images = [sample['image']]
            labels = [sample['label']]
            eval_loss = sess.run(loss, feed_dict={_images:images, _labels:labels})
            valid_loss += eval_loss
        valid_loss /= len(valid_data)
        summary = sess.run(sum_valid_loss, feed_dict={_sum_valid_loss:valid_loss})
        file_writer.add_summary(summary, eval_gs)

        saver.save(sess, 'training/simple.ckpt-%d' % eval_gs)

