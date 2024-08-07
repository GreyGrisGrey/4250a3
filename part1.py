import random

# Moves using SARSA and updates the value of the current state it is in.
# Different from Qmove only in when it selects the next action/state pair and in how it updates the policy values.
def SARSAmove(start, next, states, qMatrix, action, alpha, gamma, epsilon):
    # If the next state is a wall
    if next[0] > 4 or next[0] < 0 or next[1] > 4 or next[1] < 0:
        nextState, nextAction = epsilonSelect(epsilon, qMatrix, start)
        qMatrix[start[1]][start[0]][action] += alpha * (-1 - qMatrix[start[1]][start[0]][action] + gamma * (qMatrix[start[1]][start[0]][nextAction]))
        next = start
    # Elif the next state is a terminal state
    elif states[next[1]][next[0]] == "b":
        # "next action" doesn't really have any value upon hitting a terminal state, so that part of the equation is zeroed out
        # should still work fine
        qMatrix[start[1]][start[0]][action] += alpha * (-1 - qMatrix[start[1]][start[0]][action] + gamma * 0)
        next = None
        nextAction = None
        nextState = None
    # Elif the next state is a red state
    elif states[next[1]][next[0]] == "r":
        nextState, nextAction = epsilonSelect(epsilon, qMatrix, [0, 4])
        qMatrix[start[1]][start[0]][action] += alpha * (-20 - qMatrix[start[1]][start[0]][action] + gamma * (qMatrix[4][0][nextAction]))
        next = [0, 4]
    else:
        nextState, nextAction = epsilonSelect(epsilon, qMatrix, next)
        qMatrix[start[1]][start[0]][action] += alpha * (-1 - qMatrix[start[1]][start[0]][action] + gamma * (qMatrix[next[1]][next[0]][nextAction]))
    return next, nextAction, nextState

def Qmove(start, next, states, qMatrix, action, alpha, gamma, epsilon):
    # If the next state is a wall
    if next[0] > 4 or next[0] < 0 or next[1] > 4 or next[1] < 0:
        reward = maxFinder(qMatrix[start[1]][start[0]])
        qMatrix[start[1]][start[0]][action] += alpha * (-1 - qMatrix[start[1]][start[0]][action] + gamma * (qMatrix[start[1]][start[0]][reward]))
        next = start
        nextState, nextAction = epsilonSelect(epsilon, qMatrix, start)
    # Elif the next state is a terminal state
    elif states[next[1]][next[0]] == "b":
        # "next action" doesn't really have any value upon hitting a terminal state, so that part of the equation is zeroed out
        # should still work fine
        reward = maxFinder(qMatrix[next[1]][next[0]])
        qMatrix[start[1]][start[0]][action] += alpha * (-1 - qMatrix[start[1]][start[0]][action] + gamma * 0)
        next = None
        nextAction = None
        nextState = None
    # Elif the next state is a red state
    elif states[next[1]][next[0]] == "r":
        reward = maxFinder(qMatrix[0][4])
        qMatrix[start[1]][start[0]][action] += alpha * (-20 - qMatrix[start[1]][start[0]][action] + gamma * (qMatrix[4][0][reward]))
        next = [0, 4]
        nextState, nextAction = epsilonSelect(epsilon, qMatrix, [0, 4])
    else:
        reward = maxFinder(qMatrix[next[1]][next[0]])
        qMatrix[start[1]][start[0]][action] += alpha * (-1 - qMatrix[start[1]][start[0]][action] + gamma * (qMatrix[next[1]][next[0]][reward]))
        nextState, nextAction = epsilonSelect(epsilon, qMatrix, next)
    return next, nextAction, nextState

# Finds the index of the maximum action on a given state.
def maxFinder(actionList):
    best = max(actionList)
    for i in range(4):
        if best == actionList[i]:
            return i

# Selects the greedy choice or a random choice if the random number generator rolls below epsilon
def epsilonSelect(epsilon, qMatrix, start):
    actions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    if random.random() <= epsilon:
        move = random.randint(0, 3)
    else:
        move = maxFinder(qMatrix[start[1]][start[0]])
    return [start[0] + actions[move][0], start[1] + actions[move][1]], move

        
# The code for SARSA() and QLearning() are basically exactly the same, so only SARSA() will be commented
# The only difference is that SARSA calls SARSAmove at each time step while QLearning calls Qmove
def SARSA():
    # Q(s, a) = Q(s, a) + alpha(R + gamma(Expected Q(s+1, a+1)) - Q(s, a))
    # Initialize Q values, states, constants
    qMatrix = []
    states = []
    gamma = 0.7
    alpha = 0.4
    epsilon = 0.1
    f = open("map1.txt", "r")
    # Creating the policy
    for i in range(5):
        qRow = []
        for j in range(5):
            qSlot = []
            for k in range(4):
                qSlot.append(0)
            qRow.append(qSlot)
        qMatrix.append(qRow)
    # Creating the state matrix
    for i in f:
        stateRow = []
        for j in i:
            if j != "\n":
                stateRow.append(j)
        states.append(stateRow)
    curr = None
    # We use a simple epsilon-greedy policy.
    # It takes 1 million steps, it could afford to take less time but I wanted it accurate to the last decimal place.
    for i in range(1000000):
        if curr == None:
            curr = [0, 4]
            next, action = epsilonSelect(epsilon, qMatrix, curr)
        curr, action, next = SARSAmove(curr, next, states, qMatrix, action, alpha, gamma, epsilon)
    # Prints the policy at the end, it doesn't look very good, but each row corresponds to a row on the map
    # Each list in a row corresponds to a specific state
    # The values correspond to each of these actions in order : Down, Up, Right, Left
    # Black spaces and red spaces don't need values in the policy as the agent never moves from those states.
    for i in qMatrix:
        print(i)

def QLearning():
    # Q(s, a) = Q(s, a) + alpha(R + gamma(Expected Q(s+1, a+1)) - Q(s, a))
    # Initialize Q values, states, constants
    qMatrix = []
    states = []
    gamma = 0.75
    alpha = 0.05
    epsilon = 0.1
    f = open("map1.txt", "r")
    for i in range(5):
        qRow = []
        for j in range(5):
            qSlot = []
            for k in range(4):
                qSlot.append(0)
            qRow.append(qSlot)
        qMatrix.append(qRow)
    for i in f:
        stateRow = []
        for j in i:
            if j != "\n":
                stateRow.append(j)
        states.append(stateRow)
    curr = None
    # We use a simple epsilon-greedy policy
    for i in range(1000000):
        if curr == None:
            curr = [0, 4]
            next, action = epsilonSelect(epsilon, qMatrix, curr)
        curr, action, next = Qmove(curr, next, states, qMatrix, action, alpha, gamma, epsilon)
    for i in qMatrix:
        print(i)

print("SARSA")
print("--------------------")
SARSA()
print("QLearning")
print("--------------------")
QLearning()