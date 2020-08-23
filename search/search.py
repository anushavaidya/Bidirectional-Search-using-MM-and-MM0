# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import numpy as np
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    visited=[]
    print(problem.getStartState())
    stack = util.Stack()
    path = []
    cost = 0
    if stack.isEmpty() == True:
        stack.push((problem.getStartState(),path,cost))

    while (not stack.isEmpty()):
        temp = stack.pop()
        if problem.isGoalState(temp[0]):
            return temp[1]

        if (temp[0] not in visited):           #if current position not in visited nodes, add it to the visited list
            visited.append(temp[0])

        successors = problem.getSuccessors(temp[0])
        # print(successors)
        for i in successors:
            if i[0] not in visited:         #pushing all successors in stack that are yet to be explored
                stack.push((i[0], temp[1] + [i[1]], temp[2]+i[2]))


    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited=[]          # visited nodes
    queue = util.Queue()
    path = []
    cost = 0
    if queue.isEmpty() == True:
        queue.push((problem.getStartState(),path,cost))

    while not queue.isEmpty():
        temp = queue.pop()
        # print(temp)
        if problem.isGoalState(temp[0]):
            # print('hello',temp[2])
            return temp[1]

        if (temp[0] not in visited):    #if current position not in visited nodes, add it to the visited list
            visited.append(temp[0])

            successors = problem.getSuccessors(temp[0])
            for i in successors:
                # print(i)
                if i[0] not in visited:  #pushing all successors in queue that are yet to be explored
                    queue.push((i[0], temp[1] + [i[1]], temp[2]+i[2]))


    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = []
    pqueue = util.PriorityQueue()
    path = []
    cost = 0
    if pqueue.isEmpty() == True:
        pqueue.push((problem.getStartState(),path,cost),cost)

    while (not pqueue.isEmpty()):
        temp = pqueue.pop()

        if problem.isGoalState(temp[0]):
            return temp[1]

        if (temp[0] not in visited):        #if current position not in visited nodes, add it to the visited list
            visited.append(temp[0])

            successors = problem.getSuccessors(temp[0])
            for i in successors:
                if i[0] not in visited:     #pushing all successors in priority queue that are yet to be explored
                    pqueue.push((i[0], temp[1] + [i[1]], temp[2]+i[2]), temp[2]+i[2])

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    visited = []
    pqueue = util.PriorityQueue()
    path = []
    cost = 0
    if pqueue.isEmpty() == True:
        pqueue.push((problem.getStartState(),path,cost),cost)

    while (not pqueue.isEmpty()):
        temp = pqueue.pop()
        # print(temp)
        if problem.isGoalState(temp[0]):
            # print(temp[2])
            return temp[1]

        if (temp[0] not in visited):        #if current position not in visited nodes, add it to the visited list
            visited.append(temp[0])

            successors = problem.getSuccessors(temp[0])
            # print(successors)
            for i in successors:
                if i[0] not in visited:         #pushing all successors in priority queue that are yet to be explored
                    pqueue.push((i[0], temp[1] + [i[1]], temp[2]+i[2]), temp[2]+i[2] + heuristic(i[0],problem))


def MM0(problem):
    def PathReverse(p):
        """
        Given a action list, return the reversed version of it for the nodes expanding in the backward direction.
        """
        path = []
        for x in p:
            if x == 'North':
                z = 'South'
                path.append(z)
            if x == 'South':
                z = 'North'
                path.append(z)
            if x == 'West':
                z = 'East'
                path.append(z)
            if x == 'East':
                z = 'West'
                path.append(z)
        return path[::-1]

    gF = 0                                                     # initialize gF value to be 0 as the cost of returning to the start state from the start node is zero
    
    gB = 0
    OpenF = util.PriorityQueue()                               # create a Priority Queue to store all the nodes expanded in the forward direction
    OpenB = util.PriorityQueue()                               # craete a Priority Queue to store all the nodes expanded in the backward direction
    OpenF.push((problem.getStartState(), [], 0), 2 * gF)       # push the start state to the Queue.Since the heuristic value is zero we take into account the the value of max(g,2*g) = 2*g  
    OpenB.push((problem.goal, [], 0), 2 * gB)                  # push the goal state to the Queue 

    ClosedF = {}                                                # dictionary to store the path to reach the node from start node, with the key being its location
    ClosedB = {}                                                # dictionary to store the path to reach the node from goal node, with the key being its location
    gF_dic = {}                                                 # dictionary to store the cost to reach the node from start node, with the key being its location
    gB_dic = {}                                                 # dictionary to store the cost to reach the node from goal node, with the key being its location
    gF_dic[problem.getStartState()] = gF
    gB_dic[problem.goal] = gB
    U = float('inf')

    while (not OpenF.isEmpty()) and (not OpenB.isEmpty()):

        CurrentPopF = OpenF.pop()
        CurrentPopB = OpenB.pop()
        StateF = CurrentPopF[0]
        StateB = CurrentPopB[0]
        gF = CurrentPopF[2]
        gB = CurrentPopB[2]
        pathF = CurrentPopF[1]
        pathB = CurrentPopB[1]

        C = min(gF, gB)                                          # find the minimum cost value (i.e from the forward node and backward node)

        if StateF == StateB:
            print('reached goal1')
            return pathF + PathReverse(pathB)
        if StateF in ClosedB:
            pathB = ClosedB[StateF]
            print('reached goal2')
            return pathF + PathReverse(pathB)
        if StateB in ClosedF:
            pathF = ClosedF[StateB]
            print('reached goal3')
            return pathF + PathReverse(pathB)

        if (C == gF):                                           # If the cost of expanding a node in the forward iteration is lesser, then expand node in forward direction
            OpenB.push(CurrentPopB,2 * gB)                      # Push back the popped node of backward iteration in the queue
            ClosedF[CurrentPopF[0]] = pathF                     # store the popped node's path in dictionary closedF with key as the node's location
            SuccessorsF = problem.getSuccessors(StateF)
            for i in SuccessorsF:
                if OpenF.isthere(i[0]) or i[0] in ClosedF:      # check if successor is already present in OpenF or in ClosedF(i.e. already visited nodes)
                    if gF_dic[i[0]] < gF + i[2]:                # If yes, check if this node's stored cost is less than sum of cost to current node + cost of edge to the successor node 
                        continue
                    if OpenF.isthere(i[0]):                     
                        OpenF.remove_by_value(i[0])             # Remove node from OpenF queue if the successor node is present there. Check function in util.py.
                    elif i[0] in ClosedF:
                        del ClosedF[i[0]]                       # remove node from ClosedF if the successor node is present there
                gF_dic[i[0]] = gF + i[2]                        # update the cost to reach the succesor node and then push it to the queue
                OpenF.push((i[0], pathF + [i[1]], gF + i[2]),2*(gF + i[2]))

        else:
            OpenF.push(CurrentPopF, 2 * gF)
            ClosedB[CurrentPopB[0]] = pathB
            SuccessorsB = problem.getSuccessors(StateB)
            for i in SuccessorsB:
                if OpenB.isthere(i[0]) or i[0] in ClosedB:
                    if gB_dic[i[0]] < gB + i[2]:
                        continue
                    if OpenB.isthere(i[0]):
                        OpenB.remove_by_value(i[0])
                    elif i[0] in ClosedB:
                        del ClosedB[i[0]]
                gB_dic[i[0]] = gB + i[2]
                OpenB.push((i[0], pathB + [i[1]], gB + i[2]), 2*(gB + i[2]))

    return[]


def MM(problem,heuristic=nullHeuristic):
    def PathReverse(p):
        """
        Given a action list, return the reversed version of it for the nodes expanded in the backward direction.
        """
        path = []
        for x in p:
            if x == 'North':
                z = 'South'
                path.append(z)
            if x == 'South':
                z = 'North'
                path.append(z)
            if x == 'West':
                z = 'East'
                path.append(z)
            if x == 'East':
                z = 'West'
                path.append(z)
        return path[::-1]

    gF = 0
    epsilon = 1
    gB = 0
    OpenF = util.PriorityQueue()
    OpenB = util.PriorityQueue()
    hf = heuristic(problem.getStartState(),problem)
    hb = heuristic(problem.goal, problem)
    OpenF.push((problem.getStartState(), [], 0), max(hf,2 * gF))
    OpenB.push((problem.goal, [], 0), max(hb,2 * gB))

    ClosedF = {}                            # dictionary to store the path to reach the node from start node, with the key being its location
    ClosedB = {}                            # dictionary to store the cost to reach the node from goal node, with the key being its location
    gF_dic = {}                             # dictionary to store the cost to reach the node from start node, with the key being its location 
    gB_dic = {}                             # dictionary to store the cost to reach the node from goal node, with the key being its location
    gF_dic[problem.getStartState()] = gF
    gB_dic[problem.goal] = gB

    while (not OpenF.isEmpty()) and (not OpenB.isEmpty()):

        CurrentPopF = OpenF.pop()
        CurrentPopB = OpenB.pop()
        StateF = CurrentPopF[0]
        StateB = CurrentPopB[0]
        gF = CurrentPopF[2]
        gB = CurrentPopB[2]
        pathF = CurrentPopF[1]
        pathB = CurrentPopB[1]

        C = min(gF, gB)                             # check expnding which node is less costlier, i.e. node in the forward direction or the node in backward direction.

        if StateF == StateB:                        # check if the current nodes of forward and backward are meeting
            print('reached goal1')
            return pathF + PathReverse(pathB)
        if StateF in ClosedB:                       # check if the node to be expanded is alredy present in the array which stores the node expanded by the backward search
            pathB = ClosedB[StateF]
            print('reached goal2')
            return pathF + PathReverse(pathB)
        if StateB in ClosedF:                        # check if the node to be expanded is already present in the ClosedF(forward) array
            pathF = ClosedF[StateB]
            print('reached goal3')
            return pathF + PathReverse(pathB)

        if (C == gF):                                  # If the cost of expanding a node in the forward iteration is lesser, then expand node in forward direction
            OpenB.push(CurrentPopB, gB)                # Push back the popped node of backward iteration in the queue
            ClosedF[CurrentPopF[0]] = pathF            # store the popped node's path in dictionary closedF with key as the node's location
            SuccessorsF = problem.getSuccessors(StateF)
            for i in SuccessorsF:                   
                h_f = heuristic(i[0],problem)
                if OpenF.isthere(i[0]) or i[0] in ClosedF:       # check if successor is already present in OpenF or in ClosedF(i.e. already visited nodes)
                    if gF_dic[i[0]] < gF + i[2]:                 # If yes, check if this node's stored cost is less than sum of cost to current node + cost of edge to the successor node 
                        continue
                    if OpenF.isthere(i[0]):                      
                        OpenF.remove_by_value(i[0])               # Remove node from OpenF queue if the successor node is present there. Check function in util.py. 
                    elif i[0] in ClosedF:
                        del ClosedF[i[0]]                         # remove node from ClosedF if the successor node is present there  
                gF_dic[i[0]] = gF + i[2]                          # update the cost to reach the succesor node and then push it to the queue  
                ff = h_f + gF + i[2]                              # f(x) = g(x) + h(x)
                OpenF.push((i[0], pathF + [i[1]], max(ff,2*(gF + i[2]))),max(ff,2*(gF + i[2])))  # choose the cost value which satisfies max(f(x),2*g(x)) as the priority value

                # if OpenB.isthere(i[0]):
                #     U = min(U, gF_dic[i[0]] + gB_dic[1[0]])
        else:
            OpenF.push(CurrentPopF, gF)
            ClosedB[CurrentPopB[0]] = pathB
            SuccessorsB = problem.getSuccessors(StateB)
            for i in SuccessorsB:
                h_b = heuristic(i[0],problem)
                if OpenB.isthere(i[0]) or i[0] in ClosedB:
                    if gB_dic[i[0]] < gB + i[2]:
                        continue
                    if OpenB.isthere(i[0]):
                        OpenB.remove_by_value(i[0])
                    elif i[0] in ClosedB:
                        del ClosedB[i[0]]
                gB_dic[i[0]] = gB + i[2]
                fb = h_b + gB + i[2]                              
                OpenB.push((i[0], pathB + [i[1]], max(fb,2*(gB + i[2]))), max(fb,2*(gB + i[2])))

               
    return[]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bdmm0 = MM0
bdmm = MM
