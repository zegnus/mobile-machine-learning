import tensorflow as tf

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]

# y = Wx + b
W = tf.Variable(initial_value=[1.0], dtype=tf.float32, name='W')
b = tf.Variable(initial_value=[1.0], dtype=tf.float32, name='b')

x = tf.placeholder(dtype=tf.float32, name='x')
y_input = tf.placeholder(dtype=tf.float32, name='y_input')

# y_output = W * x + b
y_output = tf.add(x=tf.multiply(x=W, y=x, name='multiply'), y=b, name='y_output')

loss = tf.reduce_mean(input_tensor=tf.square(x=y_output-y_input), name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01, name='optimizer')
train_step = optimizer.minimize(loss=loss, name='train_step')

saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())

# write shape of the graph
tf.train.write_graph(graph_or_graph_def=session.graph_def,
                     logdir='.',
                     name='linear_regression.pbtxt',
                     as_text=False)

loss_before_training = session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train})
print(loss_before_training)

for _ in range(1000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})

# save a checkpoint after training
saver.save(sess=session, save_path='./linear_regression.ckpt')

print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))
print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))