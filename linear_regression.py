import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""in order to make random numbers predictable , seeds are set to be
reproducible by setting them at a value of 101 so the generator
creating a random value for the seed value (101) will always be
the same random value"""

np.random.seed(101)
tf.set_random_seed(101) # sets the global random seed
# generate 100 random linear data points ranging from 0 to 25
x = np.linspace(0, 25, 100)
y = np.linspace(0, 25, 100)
# add noise to the random linear data
x += np.random.uniform (-4, 4, 100)
y += np.random.uniform (-4, 4, 100)
n = len(x) # number of data points

"""
plt.scatter (x, y) # draws a scatter plot , plotting a single dot for each observation
plt.xlabel ('x') # xlabel sets the label for the x- axis
plt.ylabel ('y')
plt.title (" Training Data ") # sets the title for the plot
plt.show () # displays the figure
"""

"""In TensorFlow , variables are similar to standard coding variables
that are initialized and can be modified later . Placeholders , on
the other hand d o n t require that initial value . It
reserves a block of memory for future use."""

# define two placeholders
X = tf.placeholder(dtype=tf.float64) # placeholder X of type float
Y = tf.placeholder(dtype=tf.float64) # placeholder Y of type float

# declare two trainable TensorFlow variables and initialize them randomly using np. random . randn ()
W = tf.Variable(np.random.randn(),name="W", dtype=tf.float64 ) # the variable W denotes weight
b = tf.Variable(np.random.randn(),name="b", dtype=tf.float64 ) # the variable b denotes -> bias

learning_rate = 0.01
training_epochs = 1000

# building the hypothesis -> relationship between x and y
y_pred = tf.add(tf.multiply(X, W), b) # predicted y is the sum of (the product of X and W) and (b)

# Mean Squared Error Cost Function -> formula to determine value of the weight and bias from the given dataset
cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)

"""
tf.pow works out the the power of one tensor with another similar to how in
algebra xy would work -> then reduce_sum finds the sum of the
elements across dimensions.
"""

# Gradient Descent Optimizer -> algorithm utilized to work out the optimized / ideal parameters
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.global_variables_initializer()