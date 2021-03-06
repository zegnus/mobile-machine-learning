# Linear Regression

In this project we will build a simple [linear regression](https://en.wikipedia.org/wiki/Linear_regression) graph in [Tensorflow](https://www.tensorflow.org) that we will later use to predict values from an Android application

## Steps

We will create a graph in Tensorflow that will use linear regression for predicting values, for our loss process and we will use the [root mean square](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and a standard [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) for our optimisation process.

After our model is build and trained, we will then save our model so that somebody else can use it. In our case it will be our mobile application. But it this model could be used by anybody. You will find pre-trained models on the Internet for many different datasets that are being trained using different loss and optimisation functions.

Once we have our model saved, for actual usage we will go through a process called **graph freeze**. This process will remove unnecessary nodes in our graph that are not used in the step of inception (feed your model and get a result); it can also be optimized so that it consumes less memory and runs faster.

### Linear regression model graph

A linear regression model uses this function `y = Wx + b` that is the function of a line. This means that our model will try to fit a straight line considering all the input data and a reference data (as this is a supervised training).

The values that define the type of line are **W** and **b**; with those two parameters, given an input **x** you will get a result **y**. We will know x and y, so that our intention is that a program learns the best W and b that minimises the error between this line and all the provided points.

```python
# y_output = W * x + b
y_output = tf.add(x=tf.multiply(x=W, y=x, name='multiply'), y=b, name='y_output')
```

This `y_output` will be the final result after we feed our function with values. Notice that we are naming this node, this is very important because from the Android application we will request this node **by name** in order to retrieve the result. And this is also valid for any model that you import, you will need to know the name of the nodes that you want to use.

`W` and `b` are variables that our training method will modify in order to minimise our loss, so let's define them:

```python
W = tf.Variable(initial_value=[1.0], dtype=tf.float32, name='W')
b = tf.Variable(initial_value=[1.0], dtype=tf.float32, name='b')
```

Now we have to define the last two crucial pieces of our model, our inputs and outputs. Let me explain a bit more this step. Our model is going to be _used_ in two different ways:

1. While **training** we are going to perform a **supervised training**. Meaning that we need working examples for inputs and outputs (correct pairs of x, y).

2. While **inference** we are going to **feed the model with input values**, and we will read output values predicted from our model (we will only provide x values).

For inference we have already defined our `y_output`, but for training we have to define our working example fields. Looking at our `y_output` function it needs an input `x`, let's define it:

```python
x = tf.placeholder(dtype=tf.float32, name='x')
```

This `x` will be used through inference as source of values for our model and **also** through training for our input examples. Now we have just to define an output for our working examples, we'll call it `y_input` because we will feed it into our model (hence the _input_):

```python
y_input = tf.placeholder(dtype=tf.float32, name='y_input')
```

Now we will define a set of very simple working examples that will perfectly fit a line:

```python
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]
```

If we were to manualy find out the linear function that fist this dataset we can just assign the value -1 to W and zero to be:

```python
y = W*x + b
-1 = W*1 + b
-1 = (-1)*1 + 0 // W = -1, b = 0
```

We will see at the end wether our model can find out those values...

Now knowing all our defined placeholders, we can see that our `x_train` will be feed into the placeholder `x` and our `y_train` will be feed into the placeholder `y_input`. And this is why these two are **placeholders** and not variables.

#### Training model

In order to train a model we need to define two functions:

**The loss function:**

The loss function will tell the algorithm how far off we are between the expected output and the output that the model in that particular state is giving to us. The objective of the algorithm is to **minimise** that loss.

For this example we will use the square of the difference between the expected output and the correct output

```python
loss = tf.reduce_mean(input_tensor=tf.square(x=y_output-y_input), name='loss')
```

There are many different loss functions that you will find on the Internet, some functions work better for for some particular scenarios than others, so its always good to use something that has been proven that works instead of reinventing the wheel.

**The optimizer:**

The optimizer function will modify the variables of the function `W` and `b` for the following training step. There are lots of different optimizers, for this example we will use a very common optimizer called `gradiant descent`. This optimizer calculates the derivative between the current step and the previous step for all nodes in the graph so that the variables of the function can be **slightly** changed towards minimising the loss. The amount of change that the variables will assume is going to be defined by a parameter called **training step**.

I do not want to go into much detail about the importance of the training step but it's a very interesting parameter. The gradient descent calculation is not a smooth uniform function, meaning that if you imagine an inverted mountain (where the lowest point is the lowest loss), the walls of that mountain can go up a little bit before going down again.

Now imagine that we are down the mountain by a particular amount, let's say 1. It is possible that using this step you are stuck in that wall going a bit up because your step cannot go beyond it. Also it could happen that if your step is too big, you never actually find the lowest point of the mountain because you jump from one wall to another.

The quantity of the learning rate also has a performance impact. Fine tuning it is very important and there are many different techniques for doing so. A very good one is to use a dynamic learning rate where you start with a small value, and while your loss is getting smaller, your increase it. But this is just one of the techniques than can be used.

By the way, this value of the model is called a **hyper parameter**, and there is a whole domain centered on fine tuning hyper parameters one of my favourites is to use genetic algorithms. Two very interesting articles about this subject:
- [Hyperparameter Optimization with Keras](https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53)
- [Artificial Intelligence: Hyperparameters](https://towardsdatascience.com/artificial-intelligence-hyperparameters-48fa29daa516)

The optimizer that we are going to use for this example is the following:

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01, name='optimizer')
```

We will create a training step function that will help us write less code afterwards. This function is going to tell the algorithm, this optimizer will try to minimise the following loss function:

```python
train_step = optimizer.minimize(loss=loss, name='train_step')
```

**Let's train!**

We have everything in place for train our model. As we are using a hand-written model in Tensorflow we will have to use it's session and run everything as is expected. It could be possible to avoid all this by using Tensorflow's build it [**estimators**](https://www.tensorflow.org/guide/estimators), but this is something that we won't do for this example as what we want is to go through how everything works at the low level of implementation.

We have to create a sessions and then we have to initialise the variables that we have defined previously:

```python
session = tf.Session()
session.run(tf.global_variables_initializer())
```

Now we want to execute our functions in Tensorflow, we can do that using this `session.run(my_stuff)`, for example we can print the loss of our current model before our training:

```python
loss_before_training = session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train})
print(loss_before_training)
> loss: 41.0
```

Following with the actual training of our model, executing our training step for a few iterations as a very simple approach. For actual projects we wouldn't just do it in this simple fashion. We would first split our data-set (expected correct inputs and outputs) in different buquets such as training, validation and test. We would shuffle the data and then go though them in _epochs_ instead of going through them all at once. We can also tell the system to stop when the loss reaches a certain value so that we do not iterate through the data more times that needed, this is specially important when on production you might spend many hours or days training your models. But for now a simple approach will work:

```python
for _ in range(1000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})
```

Once we have our model trained we can print again our loss and we can also find out what are the values of the variables that the algorithm has modified:

```python
print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))
> loss: 3.4114193e-05
> W: -1.004861
> b: 0.01429191
```

If you remember our manually calculated values for W and b, the model find out pretty close values!

Finally we are going to do inference on our trained model. We are going to feed some random values into our `y_output`:

```python
print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
> [ -5.010013 -10.034318 -15.058623]
> expected: [-5.0, -10.0, -15.0]
```

Not bad for a very simple model with very unpolished values. We started with a loss of 41 and we ended up with a loss of 0.00003. And for the inference we've got a precision of 0.01.

### Save the graph

Now that we have our model trained what we might want is for somebody else to use it. You can find lots of pre-trained models on the internet and you have just created your first one, congratulations!

In order to save the model we are going to use the [saver function from Tensorflow](https://www.tensorflow.org/guide/saved_model). Before training the model (in our code, that's before our `for _ in range` line) we are going to write into a file the graph that we have created. This will write our graph in a **binary protobuffer** (Google's solution for serializing structured data):

```python
saver = tf.train.Saver()

tf.train.write_graph(graph_or_graph_def=session.graph_def,
                     logdir='.',
                     name='linear_regression.pb',
                     as_text=False)

```

Then after our model is fully training we will save it. In production code we will possible want to save the model at different points of the training and not just after we finish. This is useful in case you want to continue the training after a certain point or even after certain hours of training. This will create a **checkpoint** file with all our variables with the final values.

```python
saver.save(sess=session, save_path='./linear_regression.ckpt')
```

### Freeze and optimise the graph

Once we have our graph definition and our checkpoint with all the final variables, we can freeze the graph so that it can be optimised and prepared for production use.

- Our graph definition is serialised in **linear_regression.pb**

- Our checkpoint with final variables is serialised in **linear_regression.ckpt**

For freezing the graph we will need to feed a function with the definition of our graph, our checkpoint with all our trained variables and which output file we want to produce, in this case **frozen_linear_regression.pb**

```python
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
```

Once we have our graph frozen we will optimize the graph for inference. This is a three step process. 

First we are going to load the frozen graph into memory

```python
input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_linear_regression.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)
```

Second we are going to feed a function with our in-memory frozen graph, the input node name and the output node name that we defined previously in our model. This is one of the reasons we named those nodes.

```python
output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def=input_graph_def,
                                                                     input_node_names=['x'],
                                                                     output_node_names=['y_output'],
                                                                     placeholder_type_enum=tf.float32.as_datatype_enum)

```

Third we will serialise the optimised graph:

```python
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

It used to be very complex to use Tensorflow on Android because the library is written in C++, you use to had to build the libraries and create the JNI layer between Java and C++. You can still do so if you want and it might actually be beneficial for your project; building the libraries yourself can [reduce the size of the libraries](https://medium.com/@daj/how-to-shrink-the-tensorflow-android-inference-library-cb698facf758) as you can target it for your specific model just compiling what you really need and also you can target the selected mobile architectures that you desire.

But for this example we will use the latest invention from Google, released on 2017. We can now [import the library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android) with just one line of code in your `/app/build.grade dependencies{...}` file:

```java
implementation 'org.tensorflow:tensorflow-android:1.8.0'
```

Once we got the library, we can directly use the Java interface `TensorFlowInferenceInterface` provided by `import org.tensorflow.contrib.android.TensorFlowInferenceInterface;`

We are then going to copy of optimised graph into the `/app/src/main/assets` folder in our Android project

### Import the frozen graph and use it to make predictions

Now we need to define some values from our model:

- The name of our input node from the graph

- The name of our output node from the graph

- The shape of our input

- The path to our model

```java
private static final String MODEL_NAME = "file:///android_asset/optimized_frozen_linear_regression.pb";
private static final String INPUT_NODE = "x";
private static final String OUTPUT_NODE = "y_output";
private static final long[] INPUT_SHAPE = {1L, 1L};
```

Once we have everything defined we are going to create an instance of the Tensorflow inference interface:

```java
tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_NAME);
```

We now have everything in place for using our model and start predicting values. Once we have an input value (that can come from an EditText in our Android application) we will feed our inference with it, we will run it and then we will read the result.

- Feed our inference model:

```java
float[] floatArray = {input};
tensorFlowInferenceInterface.feed(INPUT_NODE, floatArray, INPUT_SHAPE);
```

- Run the inference:

```java
tensorFlowInferenceInterface.run(new String[] {OUTPUT_NODE});
```

- Extract the results:

```java
float[] results = {0.0f};
tensorFlowInferenceInterface.fetch(OUTPUT_NODE, results);
```

Our value will be at `results[0]` as we know there is only one output for our input (also highlighted in our input shape)

And that is it! You have created from zero a model and you have end up using it in an Android application, remember that all the code can be found in my [GitHub repository](https://github.com/zegnus/mobile-machine-learning/tree/master/linear-regression)

Next chapters we will take more complex models and problems.

I encourage you to download the code and try different inputs and outputs, check the loss function and the resulting values.

Thanks for reading!

## Bonus

This was a very simple problem, can we do more? Yeah we can. I would like to mention two extensions from this solution that might inspire you.

### What if my domain is richer

We have seen in this example that we input one value and we get one value as an output. But in a real case scenario what we might have a bunch of values for single output, or even a bunch of outputs.

For example you might want to predict how many kilometers your car will make, that will be your unique output Y. But that value might depend on just not only one input but of a series of values like road temperature, wind, average speed and so on.

So how do we use multiple inputs? Very easy. Notice that our the input in this example **x** is an array. Every position of the array is a value that corresponds to an output value. We need to change this array by a **matrix** (array of arrays), and then you can use matrix multiplications for calculating W and b.

You are lucky and as long as the size of the arrays are correct, Tensorflow already knows how to perform matrix multiplications.

You can find more information on the following article:

- [Gentlest Intro to TensorFlow #3: Matrices & Multi-feature Linear Regression](https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-part-3-matrices-multi-feature-linear-regression-30a81ebaaa6c)

If you do this your function is no longer going to be a line, but a **plane**

### What if a line doesn't really work for me

This is a very interesting question. We have explored how to create a model that tries to fit a line but maybe a line is not *rich* enough for coping with all the variablity of your dataset.

What can we do then? We can go up the hierarchy and use a [polynomial function](https://www.chegg.com/homework-help/definitions/polynomial-functions-27) but we aware that if you increase the order of the function you might overfit or create artifacts:

![Polynomial function with several degrees of freedom](https://i.stack.imgur.com/uFe4X.png)

Wikipedia has good article on [polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression) and how it looks like.

Functions and approximations are a hot topic in Machine Learning. Searching for **polynomial regression** and **deep learning** will give you a bunch of links to explore.