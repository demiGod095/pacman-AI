# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
from game import Directions
import random, time, util, sys
import game
import distanceCalculator
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


class OffensiveAgent(CaptureAgent):
    
    def __init__(self, index):
        
        self.observationHistory = []
        self.numEnemyFood = "+inf"
        self.idealTime = 0
        self.index = index

    def registerInitialState(self, gameState):
        self.startPos = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

    def chooseAction(self, gameState):

        currentEnemyFood = len(self.getFood(gameState).asList())
        if self.numEnemyFood != currentEnemyFood:
          self.numEnemyFood = currentEnemyFood
          self.idealTime = 0
        else:
          self.idealTime += 1
          
        if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
          self.idealTime = 0

        actions = gameState.getLegalActions(self.index)
        if 'Stop' in actions: actions.remove(Directions.STOP)
        takeActions = []
        for a in actions:
          if not self.findCorridor(gameState, a, 5):
            takeActions.append(a)
        if len(takeActions) == 0:
          takeActions = actions
    
        values = []
        for a in takeActions:
          new_state = gameState.generateSuccessor(self.index, a)
          temp = 0
          for i in range(1,31):
            temp += self.simulate(10, new_state)
          values.append(temp)
    
        bestValue = max(values)
        ties = filter(lambda x: x[0] == bestValue, zip(values, takeActions))
        toPlay = random.choice(ties)[1]
    
        return toPlay        
    def evaluate(self,gameState, action):
        
        features = self.getFeatures(gameState, action)
        reward = self.getReward(gameState, action)
        
        return features * reward
    
    def getFeatures(self, gameState , action):
        
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        
        foodList = self.getFood(successor).asList()
        currPos = successor.getAgentState(self.index).getPosition()
        
        if len(foodList) > 0:
            minDis = min([self.getMazeDistance(currPos, food) for food in foodList])
            features['distanceToFood'] = minDis
        
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        attackers = filter(lambda agent: not agent.isPacman and agent.getPosition() != None, enemies)
        
        if len(attackers) > 0:
            moves = [a.getPosition() for a in attackers]
            neighbour = min(moves, key = lambda agent: self.getMazeDistance(currPos, agent))
            enemyDis = self.getMazeDistance(currPos, neighbour)
            if enemyDis <= 5:
                features['distanceToGhost'] = enemyDis

        
        if successor.getAgentState(self.index).isPacman:
            features['isPacman'] = 1 
        else:
            features['isPacman'] = 0
            
        
        return features
    
    
    def getReward(self, gameState, action):
        
        if self.idealTime > 80:
            return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 1000}
        
        successor = self.getSuccessor(gameState, action)
        currPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        attackers = filter(lambda agent: not agent.isPacman and agent.getPosition() != None, enemies)
        
        if len(attackers) > 0:
            moves = [a.getPosition() for a in attackers]
            neighbour = min(moves, key = lambda agent : self.getMazeDistance(currPos, agent))
            enemyDis = self.getMazeDistance(currPos, neighbour)
            nearest = filter(lambda x: x[0] == neighbour, zip(moves,attackers))
            for agent in nearest:
                if agent[1].scaredTimer > 0:
                    return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 0, 'isPacman': 0}
                
        return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}
        
    def simulate(self, depth, gameState):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        new_state = gameState.deepCopy()
        while depth > 0:
          # Get valid actions
          actions = new_state.getLegalActions(self.index)
          # The agent should not stay put in the simulation
          actions.remove(Directions.STOP)
          current_direction = new_state.getAgentState(self.index).configuration.direction
          # The agent should not use the reverse direction during simulation
          reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
          if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)
          # Randomly chooses a valid action
          a = random.choice(actions)
          # Compute new state and update depth
          new_state = new_state.generateSuccessor(self.index, a)
          depth -= 1
        # Evaluate the final simulation state
        return self.evaluate(new_state, Directions.STOP)
            
    def findCorridor(self, gameState, action, depth):
        
        if depth == 0:
            return False
        
        score = self.getScore(gameState)
        nextState = self.getSuccessor(gameState,action)
        newScore = self.getScore(nextState)
        
        if score < newScore:
            return False
        
        actions = nextState.getLegalActions(self.index)

        if 'Stop' in actions: actions.remove('Stop')
        
        if Directions.REVERSE[nextState.getAgentState(self.index).configuration.direction] in actions:
            actions.remove(Directions.REVERSE[nextState.getAgentState(self.index).configuration.direction])
            
        if len(actions) == 0:
            
            return False
        
        for action in actions:
            if not self.findCorridor(nextState, action, depth -1):
                return False
        print (actions)
        return True
    
    def getSuccessor(self,gameState,action):
        
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()
        
        if position != util.nearestPoint(position):
            
            return successor.generateSuccessor(self.index,action)
        else:
            return successor
        


                 
class DefensiveAgent(CaptureAgent):


  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.target = None
    self.lastObservedFood = None
    # This variable will store our patrol points and
    # the agent probability to select a point as target.
    self.patrolDict = {}

  def distFoodToPatrol(self, gameState):
    """
    This method calculates the minimum distance from our patrol
    points to our pacdots. The inverse of this distance will
    be used as the probability to select the patrol point as
    target.
    """
    food = self.getFoodYouAreDefending(gameState).asList()
    total = 0

    # Get the minimum distance from the food to our
    # patrol points.
    for position in self.noWallSpots:
      closestFoodDist = "+inf"
      for foodPos in food:
        dist = self.getMazeDistance(position, foodPos)
        if dist < closestFoodDist:
          closestFoodDist = dist
      # We can't divide by 0!
      if closestFoodDist == 0:
        closestFoodDist = 1
      self.patrolDict[position] = 1.0/float(closestFoodDist)
      total += self.patrolDict[position]
    # Normalize the value used as probability.
    if total == 0:
      total = 1
    for x in self.patrolDict.keys():
      self.patrolDict[x] = float(self.patrolDict[x])/float(total)

  def selectPatrolTarget(self):
    """
    Select some patrol point to use as target.
    """
    rand = random.random()
    sum = 0.0
    for x in self.patrolDict.keys():
      sum += self.patrolDict[x]
      if rand < sum:
        return x

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

    # Compute central positions without walls from map layout.
    # The defender will walk among these positions to defend
    # its territory.
    if self.red:
      centralX = (gameState.data.layout.width - 2)/2
    else:
      centralX = ((gameState.data.layout.width - 2)/2) + 1
    self.noWallSpots = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(centralX, i):
        self.noWallSpots.append((centralX, i))
    # Remove some positions. The agent do not need to patrol
    # all positions in the central area.
    while len(self.noWallSpots) > (gameState.data.layout.height -2)/2:
      self.noWallSpots.pop(0)
      self.noWallSpots.pop(len(self.noWallSpots)-1)
    # Update probabilities to each patrol point.
    self.distFoodToPatrol(gameState)


  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    # You can profile your evaluation time by uncommenting these lines
    #start = time.time()

    # If some of our food was eaten, we need to update
    # our patrol points probabilities.
    if self.lastObservedFood and len(self.lastObservedFood) != len(self.getFoodYouAreDefending(gameState).asList()):
      self.distFoodToPatrol(gameState)

    mypos = gameState.getAgentPosition(self.index)
    if mypos == self.target:
      self.target = None

    # If we can see an invader, we go after him.
    x = self.getOpponents(gameState)
    enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    if len(invaders) > 0:
      positions = [agent.getPosition() for agent in invaders]
      self.target = min(positions, key = lambda x: self.getMazeDistance(mypos, x))
    # If we can't see an invader, but our pacdots were eaten,
    # we will check the position where the pacdot disappeared.
    elif self.lastObservedFood != None:
      eaten = set(self.lastObservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
      if len(eaten) > 0:
        self.target = eaten.pop()

    # Update the agent memory about our pacdots.
    self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()

    # No enemy in sight, and our pacdots are not disappearing.
    # If we have only a few pacdots, let's walk among them.
    if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
      food = self.getFoodYouAreDefending(gameState).asList() \
           + self.getCapsulesYouAreDefending(gameState)
      self.target = random.choice(food)
    # If we have many pacdots, let's patrol the map central area.
    elif self.target == None:
      self.target = self.selectPatrolTarget()

    actions = gameState.getLegalActions(self.index)
    goodActions = []
    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      if not new_state.getAgentState(self.index).isPacman and not a == Directions.STOP:
        newpos = new_state.getAgentPosition(self.index)
        goodActions.append(a)
        fvalues.append(self.getMazeDistance(newpos, self.target))

    # Randomly chooses between ties.
    best = min(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

    #print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
    return random.choice(ties)[1]


