import tensorflow as tf
 
sess = tf.Session()

#pb 파일로부터 네트워크 불러오기
with tf.gfile.GFile('export/my_model.pb', 'rb') as fid:
    serialized_graph = fid.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(graph_def, name='')

graph = tf.get_default_graph()
_x = graph.get_tensor_by_name('Placeholder:0')
y = graph.get_tensor_by_name('add_1:0')


for i in range(10):
    eval_y = sess.run(y, feed_dict={_x:i})
    print(i, eval_y)