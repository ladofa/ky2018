print('기본 연산 실습')

import tensorflow as tf

three = tf.constant(3)
four = tf.constant(4)
five = tf.constant(5)
mul = tf.multiply(five, four)
add = tf.add(mul, three)

sess = tf.Session()
result = sess.run(add)
print(result)

