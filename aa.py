import tensorflow as tf

w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]
with tf.GradientTape(persistent=True) as tape:
  y = x @ w + b
  loss = tf.reduce_mean(y**2)
[dl_dw, dl_db] = tape.gradient(loss, [w, b])
print(dl_dw, dl_db)
print(w,b)
w.assign_add(dl_dw)
b.assign_add(dl_db)
print(w,b)
[dl_dw, dl_db] = tape.gradient(loss, [w, b])
print(dl_dw, dl_db)
