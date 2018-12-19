print('로그 남기기')
print('learning rate 재조정')

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
y = my_network2(_x)
loss = tf.square(y - _y)


_sum_loss = tf.placeholder(tf.float32, [])
_sum_lr = tf.placeholder(tf.float32, [])
sum_loss = tf.summary.scalar('loss', _sum_loss)
sum_lr = tf.summary.scalar('learning rate', _sum_lr)
sum_op = tf.summary.merge([sum_loss, sum_lr])


gs = tf.Variable(0, trainable=False)
_lr = tf.placeholder(tf.float32, [])
current_lr = 0.5

sess = tf.Session()
opt = tf.train.AdamOptimizer(learning_rate=_lr)
train_op = opt.minimize(loss, global_step=gs)

sess.run(tf.global_variables_initializer())

file_writer = tf.summary.FileWriter('logs', sess.graph)

print('학습 시작..')
while True:
    gt_x, gt_y = gen_ground_truth()
    _, eval_loss, eval_gs = sess.run([train_op, loss, gs], feed_dict={_x:gt_x, _y:gt_y, _lr:current_lr})
    print(eval_gs, current_lr, eval_loss)

    if eval_gs < 1000:
        current_lr = 0.5
    elif eval_gs < 5000:
        current_lr = 0.1
    else:
        current_lr = 0.01

    if eval_gs == 10000:
        break

    summary = sess.run(sum_op, feed_dict={_sum_loss:eval_loss, _sum_lr:current_lr})
    file_writer.add_summary(summary, eval_gs)

print('결과 확인.. ')
for i in range(10):
    eval_y = sess.run(y, feed_dict={_x:i})
    print(i, eval_y)