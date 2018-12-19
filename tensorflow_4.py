print('다항식 3 * x = 6의 미지수 찾기')

import tensorflow as tf

_x = tf.placeholder(tf.float32, [])
_y = tf.placeholder(tf.float32, [])
w = tf.get_variable('weight_alpha', [])
y = _x * w
loss = tf.square(y - _y)
opt = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

while True:
    _, eval_y, eval_loss, eval_w = sess.run([train_op, y, loss, w], feed_dict={_x:3, _y:6})
    print(eval_y, eval_loss, eval_w)
    if eval_loss < 0.001:
        break




