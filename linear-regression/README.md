#Linear Regression

In this project we will build a simple linear regression graph in Tensorflow that we will later use to predict values from an Android application

## Steps

We will create a graph in Tensorflow that will use linear regression for our loss process and we will use a standard gradient descent for our optimisation process.

After our model is build and trained, we will then save our model so that somebody else can use it. In our case it will be our mobile application. But it this model could be used by anybody. You will find pre-trained models on the Internet for many different datasets that are being trained using different loss and optimisation functions.

Once we have our model saved, for actual usage we will go through a process called **graph freeze**. This process will remove unnecessary nodes in our graph that are not used in the step of inception (feed your model and get a result); it can also be optimized so that it consumes less memory and runs faster.

### Linear regression model graph

A linear regression model uses this function `y = Wx + b` that is the function of a line. This means that our model will try to fit a straight line considering all the input data and a reference data (as this is a supervised training).

```
# y_output = W * x + b
y_output = tf.add(x=tf.multiply(x=W, y=x, name='multiply'), y=b, name='y_output')
```

This `y_output` will be the final result after we feed our function with values. Notice that we are naming this node, this is very important because from the Android application we will request this node **by name** in order to retrieve the result.

`W` and `b` are variables that our training method will modify in order to minimise our loss, so let's define them:

```
W = tf.Variable(initial_value=[1.0], dtype=tf.float32, name='W')
b = tf.Variable(initial_value=[1.0], dtype=tf.float32, name='b')
```

Now we have to define the last two crucial pieces of our model, our inputs and outputs. Let me explain a bit more this step. Our model is going to be _used_ in two different ways:
1. While **training** we are going to perform a **supervised training**. Meaning that we need to working examples for inputs and outputs.
2. While **inference** we are going to **feed the model with input values**, and we will read output values predicted from our model.

For inference, we have already defined our `y_output`, but for training we have to define our working examples fields. Looking at our `y_output` function, it needs an input `x`, let's define it:

```
x = tf.placeholder(dtype=tf.float32, name='x')
```

This `x` will be used through inference as source of values for our model and **also** through training for our input examples. Now we have just to define an output for our working examples, we'll call it `y_input` because we will feed it into our model (hence the _input_):

```
y_input = tf.placeholder(dtype=tf.float32, name='y_input')
```

Now we will define a set of very simple working examples that will perfectly fit a line:
```
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]
```

Now knowing all our defined placeholders, we can see that our `x_train` will be feed into the placeholder `x` and our `y_train` will be feed into the placeholder `y_input`. And this is why these two are **placeholders** and not variables.

####Training model

In order to train a model we need to define two functions:

**The loss function:**

The loss function will tell the algorithm how far off we are between the expected output and the output that the model in that particular state is giving to us. The objective of the algorithm is to **minimise** that loss.

For this example we will use the square of the difference between the expected output and the correct output
```
loss = tf.reduce_mean(input_tensor=tf.square(x=y_output-y_input), name='loss')
```

There are many different loss functions that you will find on the Internet, some functions work better for for some particular scenarios than others, so its always good to use something that has been proven that works instead of reinventing the wheel.

**The optimizer**

The optimizer function will modify the variables of the function `W` and `b` for the following training step. There are lots of different optimizers, for this example we will use a very common optimizer called `gradiant descent`. This optimizer calculates the derivative between the current step and the previous step for all nodes in the graph so that the variables of the function can be **slightly** changed towards minimising the loss. The amount of change that the variables will assume is going to be defined by a parameter called **training step**.

I do not want to go into much detail about the importance of the training step but it's a very interesting parameter. The gradient descent calculation is not a smooth uniform function, meaning that if you imagine an inverted mountain (where the lowest point is the lowest loss), the walls of that mountain can go up a little bit before going down again.

Now imagine that we are down the mountain by a particular amount, let's say 1. It is possible that using this step you are stuck in that wall going a bit up because your step cannot go beyond it. Also it could happen that if your step is too big, you never actually find the lowest point of the mountain because you jump from one wall to another.

The quantity of the learning rate also has a performance impact. Fine tuning it is very important and there are many different techniques for doing so. A very good one is to use a dynamic learning rate where you start with a small value, and while your loss is getting smaller, your increase it. But this is just one of the techniques than can be used.

By the way, this value of the model is called a **hyper parameter**, and there is a whole domain centered on fine tuning hyper parameters, one of my favourites is to use genetic algorithms.

The optimizer that we are going to use for this example is the following:
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01, name='optimizer')
```

We will create a training step function that will help us write less code afterwards. This function is going to tell the algorithm, this optimizer will try to minimise the following loss function:
```
train_step = optimizer.minimize(loss=loss, name='train_step')
```

**Let's train!**

We have everything in place for train our model. As we are using a hand-written model in Tensorflow we will have to use it's session and run everything as is expected. It could be possible to avoid all this by using Tensorflow's build it **estimators**, but this is something that we won't do for this example as what we want is to go through how everything works at the low level of implementation.

We have to create a sessions and then we have to initialise the variables that we have defined previously:
```
session = tf.Session()
session.run(tf.global_variables_initializer())
```

Now we want to execute our functions in Tensorflow, we can do that using this `session.run(my_stuff)`, for example we can print the loss of our current model before our training:
```
loss_before_training = session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train})
print(loss_before_training)
> loss: 41.0
```

Following with the actual training of our model, executing our training step for a few iterations as a very simple approach. For actual projects we wouldn't just do it in this simple fashion. We would first split our data-set (expected correct inputs and outputs) in different buquets such as training, validation and test. We would shuffle the data and then go though them in _epochs_ instead of going through them all at once. We can also tell the system to stop when the loss reaches a certain value so that we do not iterate through the data more times that needed, this is specially important when on production you might spend many hours or days training your models. But for now a simple approach will work:
```
for _ in range(1000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})
```

Once we have our model trained we can print again our loss and we can also find out what are the values of the variables that the algorithm has modified:
```
print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))
> loss: 3.4114193e-05
> W: -1.004861
> b: 0.01429191
```

Finally we are going to do inference on our trained model. We are going to feed some random values into our `y_output`:
```
print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
> [ -5.010013 -10.034318 -15.058623]
> expected: [-5.0, -10.0, -15.0]
```

Not bad for a very simple model with very unpolished values. We started with a loss of 41 and we ended up with a loss of 0.00003. And for the inference we've got a precision of 0.01 .


### Save the graph

Now that we have our model trained what we might want is for somebody else to use it. You can find lots of pre-trained models on the internet and you have just created your first one, congratulations!

In order to save the model we are going to use the saver function from Tensorflow. Before training the model (in our code, that's before our `for _ in range` line) we are going to write into a file the graph that we have created. This will write our graph in a **binary protobuffer** (Google's solution for serializing structured data):

```
saver = tf.train.Saver()

tf.train.write_graph(graph_or_graph_def=session.graph_def,
                     logdir='.',
                     name='linear_regression.pb',
                     as_text=False)

```

Then after our model is fully training we will save it. In production code we will possible want to save the model at different points of the training and not just after we finish. This is useful in case you want to continue the training after a certain point or even after certain hours of training. This will create a **checkpoint** file with all our variables with the final values.

```
saver.save(sess=session, save_path='./linear_regression.ckpt')
```

### Freeze and optimise the graph

Once we have our graph definition and our checkpoint with all the final variables, we can freeze the graph so that it can be optimised and prepared for production use.

```
> graph definition at linear_regression.pb
> checkpoint with final variables at linear_regression.ckpt
```

The code is a bit long but the important bits in the parameters are the location of the graph and the checkpoint for freezing the graph, and the name of our input and output placeholders `x` and `y_output`

```
# Freeze graph and write it to frozen_linear_regression.pb
freeze_graph.freeze_graph(input_graph='linear_regression.pb',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='linear_regression.ckpt',
                          output_node_names='y_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='frozen_linear_regression.pb',
                          clear_devices=True,
                          initializer_nodes='',
                          variable_names_blacklist='')

# Read frozen graph, optimize it and write it to optimized_frozen_linear_regression.pb

# input_graph_def contains all the data from the pb file, converted into a String
input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_linear_regression.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def=input_graph_def,
                                                                     input_node_names=['x'],
                                                                     output_node_names=['y_output'],
                                                                     placeholder_type_enum=tf.float32.as_datatype_enum)

f = tf.gfile.FastGFile(name='optimized_frozen_linear_regression.pb', mode='w')

f.write(file_content=output_graph_def.SerializeToString())

> Converted 2 variables to const ops.
```

Freezing and optimising the graph can take many different forms, it's worth to go through the documentation so that you are aware of what's possible and how to use it for your production environments.

## Create an Android application with the necessary libraries

We are going to create an Android application that can use our trained model in order to make prediction, this step is call do to `inference` on a model.

We are going to need:
1. Tensorflow libraries for Android
2. Import our model using the libraries
3. Use the model for inference

### Tensorflow libraries for Android

// todo

### Import the frozen graph and use it to make predictions
// todo 