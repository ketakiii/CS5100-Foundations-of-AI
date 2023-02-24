import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        dot = nn.DotProduct(self.w, x)
        return dot

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch = 1
        flag = True
        while flag:
            flag = False
            for x, y in dataset.iterate_once(batch):
                if self.get_prediction(x) == nn.as_scalar(y):
                    pass
                else:
                    self.w.update(nn.Constant(nn.as_scalar(y) * x.data), 1)
                    flag = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
        self.learning_rate = -0.005
        self.w1 = nn.Parameter(1, 50)
        self.b1 = nn.Parameter(1, 50)
        self.w2 = nn.Parameter(50, 1)
        self.b2 = nn.Parameter(1, 1)
        self.parameters = [self.w1, self.w2, self.b1, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        matrix1 = nn.Linear(x, self.w1)
        first_layer = nn.ReLU(nn.AddBias(matrix1, self.b1))

        matrix2 = nn.Linear(first_layer, self.w2)
        second_layer = nn.AddBias(matrix2, self.b2)
        return second_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        a = self.run(x)
        loss = nn.SquareLoss(a, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:

            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                self.w1.update(gradients[0], self.learning_rate)
                self.w2.update(gradients[1], self.learning_rate)
                self.b1.update(gradients[2], self.learning_rate)
                self.b2.update(gradients[3], self.learning_rate)

            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
        self.learning_rate = -0.005
        self.w1 = nn.Parameter(784, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 10)
        self.b2 = nn.Parameter(1, 10)
        self.parameters = [self.w1, self.w2, self.b1, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        matrix1 = nn.Linear(x, self.w1)
        first_layer = nn.ReLU(nn.AddBias(matrix1, self.b1))

        matrix2 = nn.Linear(first_layer, self.w2)
        second_layer = nn.AddBias(matrix2, self.b2)

        return second_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        a = self.run(x)
        loss = nn.SoftmaxLoss(a, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:

            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                self.w1.update(gradients[0], self.learning_rate)
                self.w2.update(gradients[1], self.learning_rate)
                self.b1.update(gradients[2], self.learning_rate)
                self.b2.update(gradients[3], self.learning_rate)

            if dataset.get_validation_accuracy() >= 0.97:
                return


class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """

    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"

        self.lr = -0.13
        self.numTrainingGames = 5000
        self.batch_size = 50
        self.w1 = nn.Parameter(state_dim, 1300)
        self.b1 = nn.Parameter(1, 1300)
        self.w2 = nn.Parameter(1300, 100)
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, action_dim)
        self.b3 = nn.Parameter(1, action_dim)

        self.all = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"

        Q_predict = self.run(states)
        loss = nn.SquareLoss(Q_predict, Q_target)
        return loss

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

        s = states
        s = nn.Linear(s, self.all[0])
        s = nn.AddBias(s, self.all[1])

        for i in range(1, len(self.all) // 2):
            s = nn.ReLU(s)
            s = nn.Linear(s, self.all[2 * i])
            s = nn.AddBias(s, self.all[2 * i + 1])

        return s


    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        l = self.get_loss(states, Q_target)
        g = nn.gradients(l, self.all)
        for i in range(len(self.all)):
            self.all[i].update(g[i], self.lr)

