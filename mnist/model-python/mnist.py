import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# train_image = mnist_data.train.images[0]
# train_label = mnist_data.train.labels[1]
#
# 28x28 = 784

# Y = Wx + b

x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
W = tf.Variable(initial_value=tf.zeros(shape=[784, 10]), name='W')
b = tf.Variable(initial_value=tf.zeros(shape=[10]), name='b')
y = tf.add(x=tf.matmul(a=x_input, b=W), y=b, name='y')

y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_input')

cross_entropy_loss = tf.reduce_mean(
    input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y),
    name='cross_entropy_loss'
)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5, name='optimizer')
train_step = optimizer.minimize(loss=cross_entropy_loss, name='train_step')

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for _ in range(1000):
    # we have 60k images, we will split them onto batches
    batch = mnist_data.train.next_batch(100)
    train_step.run(feed_dict={x_input: batch[0], y_input: batch[1]})

correct_prediction = tf.equal(x=tf.argmax(y, 1), y=tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype=tf.float32))  # zero or one
print(accuracy.eval(feed_dict={x_input: mnist_data.test.images, y_input: mnist_data.test.labels}))

print(session.run(fetches=y, feed_dict={x_input: [mnist_data.test.images[0]]}))
