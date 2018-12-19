#파이썬 연습 코드 1

print('리스트 연습 1')
my_list = []
#my_list = ['초기값을', '이렇게', '넣어줘도', '된다.']
my_list.append(1)
my_list.append(3.14)
my_list.append('이름')
print(my_list)
print(my_list[0])
print(my_list[1])
print(my_list[2])

print('리스트 연습 2')
size = len(my_list)
for i in range(size):
    print(my_list[i])

print('리스트 연습 3')
for e in my_list:
    print(e)


print('리스트 연습 4')
for index, value in enumerate(my_list):
    print(index, value)


print('딕셔너리 연습 1')

my_dict = {}
my_dict = {'초기값을':'이렇게', '넣어줘도':'된다.'}
my_dict[0] = 10
my_dict[-100] = '음수건 뭔건 상관없다'
my_dict['asdf'] = 'qwer'

print(my_dict[0])
print(my_dict[-100])
print(my_dict['asdf'])
print(my_dict['이런 건 없는데?'])

print('딕셔너리 연습 2')
keys = my_dict.keys()
for key in keys:
    value = my_dict[key]
    print(key, value)

print('딕셔너리 연습 2')
for key, value in my_dict.items():
    print(key, value)

print('해쉬값은 정렬되어 있다.')