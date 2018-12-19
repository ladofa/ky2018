#numpy 실습
import numpy as np


print('실습 1')
a = np.array([1, 2, 3, 4])
print(a)
b = a + 1
print(b)
c = a * b
print(c)

print('실습 2')
print(a.dtype)
a[0] = 3.14
print(a)
print(np.float32(a))

print('실습 3')
a = np.arange(1, 13)
a = a.reshape([3, 4])
print(a)
print(a[1, 2])
print(a.shape)
print('rows', a.shape[0])
print('cols', a.shape[1])

print('실습 4')
print(a[:, 0])
print(a[1, :])
print(a[1:3, 2:4])

print('실습 5')
b = a[1:3, 2:4]
print(b)
b[0, 1] = -1
print(b)
print(a)


print('실습 5')
k = a > 5
print(k)
print(a[k])
a[k] = 100
print(a)
a[k] = [1, 2, 3, 4, 5, 6]
print(a)
