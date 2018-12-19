print('place holder 실습')

import tensorflow as tf

_x = tf.placeholder(tf.float32, shape=[])
three = tf.constant(3)
four = tf.constant(4)
mul = tf.multiply(x, four)
add = tf.add(mul, three)
sess = tf.Session()
y = sess.run(add, feed_dict={_x:10})
print(y)

