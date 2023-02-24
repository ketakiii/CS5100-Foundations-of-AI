# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            temp = self.values.copy()
            for s in self.mdp.getStates():
                values = []

                for a in self.mdp.getPossibleActions(s):
                    val = 0.0
                    for next, k in self.mdp.getTransitionStatesAndProbs(s, a):
                        reward = self.mdp.getReward(s, a, next)
                        val += k * (reward + self.discount * temp[next])
                    values.append(val)

                if values:
                    self.values[s] = max(values)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        currentstate = state
        q = 0
        for next, transition in self.mdp.getTransitionStatesAndProbs(currentstate, action):
            reward = self.mdp.getReward(currentstate, action, next)
            q += transition * (reward + (self.discount * self.getValue(next)))
        return q

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxvalue = - 99999
        currentaction = None

        for a in self.mdp.getPossibleActions(state):
            v = self.getQValue(state, a)
            if v > maxvalue:
                maxvalue = v
                currentaction = a
        return currentaction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        c = 0
        iter = self.iterations
        while c < iter:
            for s in self.mdp.getStates():
                q = util.Counter()  # the Q values for given the specified state

                for a in self.mdp.getPossibleActions(s):
                    q[a] = self.computeQValueFromValues(s, a)

                self.values[s] = q[q.argMax()]

                c += 1
                if c >= iter:
                    return


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        pred = {}
        for s in self.mdp.getStates():
            pred[s] = set()

        p = util.PriorityQueue()

        for s in self.mdp.getStates():
            q = util.Counter()

            for a in self.mdp.getPossibleActions(s):
                for n, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    if prob != 0:
                        pred[n].add(s)

                q[a] = self.computeQValueFromValues(s, a)

            if not self.mdp.isTerminal(s):
                max = q[q.argMax()]
                difference = abs(self.values[s] - max)
                p.update(s, -difference)

        for i in range(self.iterations):
            if p.isEmpty():
                return

            state = p.pop()

            if not self.mdp.isTerminal(state):
                qs = util.Counter()
                for a in self.mdp.getPossibleActions(state):
                    qs[a] = self.computeQValueFromValues(state, a)

                self.values[state] = qs[qs.argMax()]

            for k in pred[state]:
                Q = util.Counter()
                for a in self.mdp.getPossibleActions(k):
                    Q[a] = self.computeQValueFromValues(k, a)

                max2 = Q[Q.argMax()]
                diff = abs(self.values[k] - max2)

                if diff > self.theta:
                    p.update(k, -diff)
