print('복잡한 model 학습')
print('학습 최적화 : global_step, learning_rate')

import tensorflow as tf
import random

def gen_ground_truth():
    x = random.randint(0, 100)
    y = x * 3 + 10
    return x, y

def my_network(x):
    w1 = tf.get_variable('weight_alpha', [])
    w2 = tf.get_variable('weight_beta', [])
    y = x * w1 + w2
    return y

def my_network2(x):
    w1 = tf.get_variable('weight_alpha', [])
    w2 = tf.get_variable('weight_beta', [])
    w3 = tf.get_variable('weight_gamma', [])
    y = x * x * w1 + x * w2 + w3
    return y

_x = tf.placeholder(tf.float32, [])
_y = tf.placeholder(tf.float32, [])
y = my_network(_x)
loss = tf.square(y - _y)

#gs
#lr

sess = tf.Session()
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss)

sess.run(tf.global_variables_initializer())

print('학습 시작..')
while True:
    gt_x, gt_y = gen_ground_truth()
    _, eval_loss = sess.run([train_op, loss], feed_dict={_x:gt_x, _y:gt_y})
    print(eval_loss)
    if eval_loss < 0.001:
        break

print('결과 확인.. ')
for i in range(10):
    eval_y = sess.run(y, feed_dict={_x:i})
    print(i, eval_y)


