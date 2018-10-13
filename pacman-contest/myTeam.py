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
        self.index = index
        self.target = None
        self.prevOberservedFood = None
        self.areaToDefend = {}
        self.observationHistory = []
        
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        
        if self.red:
            midX = (gameState.data.layout.width -2 )/2
        else:
            midX = (gameState.data.layout.width -2 )/2 + 1
            

        self.noWalls = []
        for i in range(1,gameState.data.layout.height - 1 ):
            if not gameState.hasWall(midX, i):
                self.noWalls.append((midX , i))
                
        while len(self.noWalls) > (gameState.data.layout.height -2)/2:
            self.noWalls.pop(0)
            self.noWalls.pop(len(self.noWalls) -1)
            
        self.foodToProtect(gameState)

    def foodToProtect(self, gameState):
        foodList = self.getFoodYouAreDefending(gameState).asList()
        score = 0
        
        for position in self.noWalls:
            minDis = "+inf"
            for food in foodList:
                distance = self.getMazeDistance(position, food)
                if distance < minDis:
                    minDis = distance
            
            if minDis == 0:
                minDis = 1
            
            self.areaToDefend[position] = 1.0/float(minDis)
            score += self.areaToDefend[position]
            
        if score == 0:
            score = 1
        
        for area in self.areaToDefend.keys():
            self.areaToDefend[area] = float(self.areaToDefend[area]/float(score))
            
          
    def chooseAction(self, gameState):
        
        if self.prevOberservedFood and len(self.prevOberservedFood) != len(self.getFoodYouAreDefending(gameState).asList()):
            self.foodToProtect(gameState)
            
        currPos = gameState.getAgentPosition(self.index)
        if currPos == self.target:
            self.target = None
            
        agent = self.getOpponents(gameState)
        enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = filter(lambda agent: agent.isPacman and agent.getPosition() != None, enemies)
        
        if len(invaders) > 0:
            locations = [agent.getPosition() for agent in invaders]
            self.target = min(locations, key = lambda a: self.getMazeDistance(currPos, a))
        
        elif self.prevOberservedFood != None:
            food = set(self.prevOberservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
            if len(food) > 0:
                self.target = food.pop()
                
        self.prevOberservedFood = self.getFoodYouAreDefending(gameState).asList()
        
        if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(food)
        
        elif self.target == None:
            self.target = self.goToTargetArea()
            
        actions = gameState.getLegalActions(self.index)
        optimalActions = []
        values = []
        for action in actions:
          new_state = gameState.generateSuccessor(self.index, action)
          if not new_state.getAgentState(self.index).isPacman and not action == Directions.STOP:
            newpos = new_state.getAgentPosition(self.index)
            optimalActions.append(action)
            values.append(self.getMazeDistance(newpos, self.target))
    
        # Randomly chooses between ties.
        optimalValue = min(values)
        ties = filter(lambda x: x[0] == optimalValue, zip(values, optimalActions))
    
        #print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]
            
       
    def goToTargetArea(self):
        sum = 0.0
        randomValue = random.random()
        for x in self.areaToDefend.keys():
            sum += self.areaToDefend[x]
            if randomValue < sum:
                return x
             
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

