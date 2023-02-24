# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        no_of_food = successorGameState.getNumFood()
        #print("scared time",newScaredTimes)
        min_val = float("inf")
        for g in newGhostStates:
            ghostdistance = util.manhattanDistance(newPos, g.getPosition())
            if min_val > ghostdistance:
                min_val = ghostdistance


        food = newFood.asList()
        mini = float("inf")
        for f in food:
            fooddistance = util.manhattanDistance(newPos, f)
            if mini > fooddistance:
                mini = fooddistance
        if mini == float("inf"):
            mini = 0


        scoreDiff = successorGameState.getScore() - scoreEvaluationFunction(currentGameState)
        minscared = float("inf")
        for n in newScaredTimes:
            if minscared > n:
                minscared = n
        if minscared == float("inf"):
            minscared = 0

        if minscared != 0:
            ghostdistance = -ghostdistance * 3


        # if minscared > 0:
        #     score = (15 / fooddistance + 1) + (8 / ghostdistance) + (80 * no_of_food)
        # else:
        #     score = (15 / fooddistance + 1) + (8 * ghostdistance) + (80 * no_of_food)

        return (((18 / (mini + 1)) + (75 / (no_of_food + 1))) + ((ghostdistance * 1) / 7.9) + 0.9 * scoreDiff)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState):
            cur_agent = self.index
            new_agent = cur_agent + 1
            if cur_agent == gameState.getNumAgents() - 1:
                new_agent = self.index  #set to 0



            best_action = None
            best_action_value  = -float("inf")
            for action in gameState.getLegalActions(cur_agent):
                successor = gameState.generateSuccessor(cur_agent, action)
                current_val = minvalue(successor, 0, new_agent)
                if current_val > best_action_value:
                    best_action_value = current_val
                    best_action = action

            return best_action

        def maxvalue(gameState,depth,agent):
            if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
                return self.evaluationFunction(gameState)

            new_depth = depth
            if agent == gameState.getNumAgents() - 1:
                new_depth = depth + 1   #ply
                new_agent = self.index  #set to 0
            else:
                new_agent = agent + 1
            v = -float('inf')

            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                v = max(v, minvalue(successor, new_depth, new_agent))
            return v

        def minvalue(gameState,depth,agent):
            if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
                return self.evaluationFunction(gameState)

            new_depth = depth
            if agent == gameState.getNumAgents() - 1:
                new_depth = depth + 1   #ply
                new_agent = self.index  #set to 0
            else:
                new_agent = agent + 1
            v = float('inf')
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                if new_agent == self.index:
                    v = min(v, maxvalue(successor, new_depth, new_agent))
                else:
                    v = min(v, minvalue(successor, new_depth, new_agent))
            return v


        return minimax(gameState)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(gameState):
            v, action = maxvalue(gameState, 0, self.index, -float('inf'), float('inf'))
            return action


        def maxvalue(gameState, depth, agent, a, b):
            if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
                return self.evaluationFunction(gameState), None

            newdepth = depth
            if agent == gameState.getNumAgents() - 1:
                newdepth = depth + 1   #ply
                newagent = self.index  #set to 0
            else:
                newagent = agent + 1

            v = -float('inf')
            bestaction = None
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                v_temp = max(v, minvalue(successor, newdepth, newagent, a, b)[0])
                if v != v_temp:
                    bestaction = action
                    v = v_temp

                if v > b:
                    return v, bestaction
                a = max(a, v)
            return v, bestaction


        def minvalue(gameState, depth, agent, a, b):
            if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
                return self.evaluationFunction(gameState), None

            newdepth = depth
            if agent == gameState.getNumAgents() - 1:
                newdepth = depth + 1   #ply
                newagent = self.index  #set to 0
            else:
                newagent = agent + 1

            v = float('inf')
            bestaction = None
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                if newagent == self.index:
                    v_temp = min(v, maxvalue(successor, newdepth, newagent, a, b)[0])
                else:
                    v_temp = min(v, minvalue(successor, newdepth, newagent, a, b)[0])

                if v!= v_temp:
                    bestaction = action
                    v = v_temp

                if v < a:
                   return v, bestaction
                b = min(b, v)
            return v, bestaction
        return alphabeta(gameState)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction( self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState,depth,cur_agent):
            maxvalue = -float("inf")
            new_depth = depth + 1
            if (gameState.isWin() or gameState.isLose() or new_depth == self.depth):
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                maxvalue = max(maxvalue, exp_value(successor, new_depth, 1))
            return maxvalue

        def exp_value(gameState,depth,cur_agent):
            if (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            v = 0
            num_actions = len(gameState.getLegalActions(cur_agent))
            for action in gameState.getLegalActions(cur_agent):
                successor_state = gameState.generateSuccessor(cur_agent, action)
                if cur_agent == gameState.getNumAgents() - 1:
                    cur_val = max_value(successor_state,depth,0)
                else:
                    cur_val = exp_value(successor_state,depth,cur_agent+1)
                v += cur_val

            if num_actions == 0:
                return 0
            return float(v) / num_actions

        best_action = ''
        best_val = -float("inf")
        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0,action)
            val = exp_value(successor_state, 0, 1)
            if val > best_val:
                best_val = val
                best_action = action
        return best_action

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    no_of_food = currentGameState.getNumFood()
    # print("scared time",newScaredTimes)
    min_val = float("inf")
    ghostdistance = 0
    for g in newGhostStates:
        ghostdistance = util.manhattanDistance(newPos, g.getPosition())
        if min_val > ghostdistance:
            min_val = ghostdistance

    food = newFood.asList()
    mini = float("inf")
    for f in food:
        fooddistance = util.manhattanDistance(newPos, f)
        if mini > fooddistance:
            mini = fooddistance
    if mini == float("inf"):
        mini = 0

    scoreDiff = currentGameState.getScore()
    minscared = float("inf")
    for n in newScaredTimes:
        if minscared > n:
            minscared = n
    if minscared == float("inf"):
        minscared = 0

    if minscared != 0:
        ghostdistance = -ghostdistance * 3

    # if minscared > 0:
    #     score = (15 / fooddistance + 1) + (8 / ghostdistance) + (80 * no_of_food)
    # else:
    #     score = (15 / fooddistance + 1) + (8 * ghostdistance) + (80 * no_of_food)

    return (((18 / (mini + 1)) + (75 / (no_of_food + 1))) + ((ghostdistance * 1) / 7.9) + 0.9 * scoreDiff)

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
