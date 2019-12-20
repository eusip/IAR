# standard libraries
import heapq
from copy import deepcopy

# third-party libraries
import numpy as np 

# original maze

ROWS = 6
COLS = 9
START = [2, 0]   
GOAL = [[0, 8]]
OBSTACLES = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

class PriorityQueue():
    """
    This class provides a priority queue for the queue-Dyna system.
    """
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('Pop from an empty priority queue.')

    def empty(self):
        return not self.entry_finder


class Maze():
    """
    This class defines the maze and provides functionality for extending its 
    size.
    """
    # maze map
    # 0,0 ####
    #        #
    #        #
    ###### n,n
    def __init__(self):
        # determine maze resolution
        self.ROWS = ROWS
        self.COLS = COLS
        # define possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN,
                        self.ACTION_LEFT, self.ACTION_RIGHT]
        # start state
        self.START = START
        # goal state
        self.GOAL = GOAL
        # all obstacles
        self.obstacles = OBSTACLES
        self.old_obstacles = None
        self.new_obstacles = None
        # time to change obstacles
        self.obstacle_switch_time = None
        # initial state-action pair values
        # self.stateActionValues = np.zeros((self.rows, self.cols, len(self.actions)))
        # the size of q value
        self.q_size = (self.ROWS, self.COLS, len(self.actions))
        # max steps
        self.max_steps = 1e4
        # track the resolution for this maze
        self.resolution = 1


    def extend_state(self, state, factor):
        """
        This function extends a state to a higher resolution maze.
        Parameters
        ----------
        state : The state in the original maze. \n
        factor : The factor by which a state is scaled (e.g. a factor 
        of 2 will double each state).
        """
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states


    def extend_maze(self, factor):
        """
        This function extends a state to a higher resolution maze.
        Parameters
        ----------
        factor : The factor by which a maze is scaled (e.g. a factor 
        of 2 will double the size of the maze).
        """
        new_maze = Maze()
        new_maze.COLS = self.COLS * factor
        new_maze.ROWS = self.ROWS * factor
        new_maze.START = [self.START[0]
                                * factor, self.START[1] * factor]
        new_maze.GOAL = self.extend_state(self.GOAL[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.ROWS,
                           new_maze.COLS, len(new_maze.actions))
        new_maze.resolution = factor
        return new_maze


    def step(self, state, action):
        """
        Parameters
        ----------
        state : A list containing the grid coordinates of the current 
        state of the agent. \n
        action : The action to be taken by the agent in the associated 
        state.
        Returns
        -------
        [x, y] : New agent state (x - rows, y - cols). \n
        reward : The reward in the new state [x, y].
        """
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.ROWS - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.COLS - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL:
            reward = 100.0
        else:
            reward = 0.0
        return [x, y], reward


class DynaParams():
    """
    This class contains the parameters used to characterize the 
    queue-Dyna system.
    """
    def __init__(self):
        # discount factor
        self.gamma = 0.95
        # probability for exploration
        self.epsilon = 0.1
        # step size
        self.alpha = 0.5
        # n-step planning
        self.planning_steps = 5
        # threshold for priority queue
        self.theta = 1e-4
 

def choose_action(state, q_value, maze, dyna_params):
    """
    This function selects an action by choosing between exploration and 
    exploitation following an epsilon-greedy approach. Exploration has 
    been turned off in replication of the article.
    Parameters
    ----------
    state : A list containing the grid coordinates of the current 
    state of the agent. \n
    q_value : The q-value of the current state. \n
    maze : An instance of the Maze class. \n
    dyna_params : An instance of the DynaParams class.
    Returns
    -------
    An action from the Maze class.
    """
    # if np.random.binomial(1, dyna_params.epsilon) == 1:  # explore
    #     return np.random.choice(maze.actions)
    # else:  # exploit
    #     values = q_value[state[0], state[1], :]
    #     return np.random.choice([action for action, value in \
    #         enumerate(values) if value == np.max(values)])
    # exploitation
    values = q_value[state[0], state[1], :]
    return np.random.choice([action for action, value in
                             enumerate(values) if value == np.max(values)])


class Dyna():
    """
    This class contains the functions necessary to implement the 
    random-update Dyna algorithm.
    """
    def __init__(self, rand=np.random):
        """
        rand : An instance of np.random.RandomState used for sampling.
        """
        self.model = dict()
        self.rand = rand


    def feed(self, state, action, next_state, reward):
        """
        This function takes a state-action pair and if the pair has not 
        yet occured adds it to a state-action pair nested dictionary as 
        a key. The corresponding value is the next state as well as the 
        reward in that state. This step equates to model learning.
        Parameters
        ----------
        state : \n
        action : \n
        next_state : \n
        reward : \n
        """
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]


    def sample(self):
        """
        This function randomly samples from they set of previously 
        visited state-action pairs and returns a state-action pair as 
        well as the next state and its reward. This step equates to 
        model planning.
        Returns
        -------
        list(state) : \n
        action :  \n
        list(next_state) : \n
        reward : 
        """
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward


class QueueDyna(Dyna):
    """
    This class contains the additional functions necessary to implement 
    the largest-first Dyna algorithm.
    """
    def __init__(self, rand=np.random):
        """
        rand : An instance of np.random.RandomState used for sampling.
        """
        Dyna.__init__(self, rand)
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        # track predecessors for every state
        self.predecessors = dict()


    def insert(self, priority, state, action):
        """
        This function inserts a station-action pair into the priority 
        queue along with its priority value. Priority is added as a 
        negative value because the queue is a minimum heap.
        priority : \n
        state : \n
        action : \n
        """
        self.priority_queue.add_item((tuple(state), action), -priority)


    def empty(self):
        return self.priority_queue.empty()


    def pop_queue(self):
        """
        This function pops an item from the top of the priority queue
        and returns a priority, state-action pair, next state and the
        next state's reward. This step equates to model planning.
        Returns
        -------
        priority : \n
        list(state) : \n
        action :  \n
        list(next_state) : \n
        reward : 
        """
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward


    def feed(self, state, action, next_state, reward):
        """
        This function takes a state-action pair and if the pair has not
        yet occured adds it to a state-action pair nested dictionary as
        a key. The corresponding value is the next state as well as the
        reward in that state. This step equates to model learning.
        Parameters
        ----------
        state : \n
        action : \n
        next_state : \n
        reward : \n
        """
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        Dyna.feed(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((tuple(state), action))


    def predecessor(self, state):
        """
        This function returns all the precedessors of a given state.
        Paramters
        ---------
        state : The state for which predecessors are returned.
        Returns
        -------
        precedessors : A list containing all the precedessors of a state. 
        """
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            predecessors.append(
                [list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors


def dyna(q_value, model, maze, dyna_params, start):
    """
    This function plays a single episode of the Dyna algorithm.
    Parameters
    ----------
    q_value : The set of q-values to be updated. \n
    model : An instance of the Dyna model used for Q-planning. \n
    maze : An instance of the Maze class. \n
    dyna_params : An instance of the DynaParams class.
    start : The start state for the maze.
    Returns
    -------
    steps : The number of steps taken during the episode. \n
    backups : The number of backups which took place during the episode.
    """
    state = start  # 2.a
    steps = 0
    backups = 0

    while state not in maze.GOAL:
        # track the steps
        steps += 1
        # get action (2.b)
        action = choose_action(state, q_value, maze, dyna_params)
        # take action (2.c)
        next_state, reward = maze.step(state, action)
        # Q-Learning update (2.d)
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * \
                np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])
        
        # real experience - feed the model with an experience (2.e)
        model.feed(state, action, next_state, reward)

        # simulated experience - planning using random experience (2.f)
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * \
                    np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state  # 2.a
        
        # check if the step limit has been exceeded
        if steps >= maze.max_steps:
            break

    backups = steps * (dyna_params.planning_steps + 1)
    # print('steps: %i, backups: %i' % (steps, backups))

    return steps, backups


def dyna_lf(q_value, model, maze, dyna_params, start):
    """
    This function plays a single episode of the largest-first Dyna 
    algorithm.
    Parameters
    ----------
    q_value : The set of q-values to be updated. \n
    model : An instance of the LFDyna model used for prioritized 
    Q-planning. \n
    maze : An instance of the Maze class. \n
    dyna_params : An instance of the DynaParams class.
    start : The start state for the maze.
    Returns
    -------
    steps : The number of steps taken during the episode. \n
    backups : The number of backups which took place during the episode. 
    """
    state = start  # 2.a
    steps = 0
    backups = 0

    while state not in maze.GOAL:
        # track the steps
        steps += 1
        # get action (2.b)
        action = choose_action(state, q_value, maze, dyna_params)
        # take action (2.c)
        next_state, reward = maze.step(state, action)
        # real experience - feed the model with experience (2.d)
        model.feed(state, action, next_state, reward)
        # get the priority for current state-action pair (2.e)
        priority = np.abs(reward + dyna_params.gamma * \
            np.max(q_value[next_state[0], next_state[1], :]) -
                        q_value[state[0], state[1], action])

        # add item to the priority queue
        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # begin planning (2.f)
        planning_step = 0

        # simulated experience - planning using priority queue
        while not model.empty():
            # get the 4-tuple with highest priority from the priority queue
            priority, state_, action_, next_state_, reward_ = model.pop_queue()
            # update the Q-value for the 4-tuple from the priority queue
            delta = reward_ + dyna_params.gamma * \
                np.max(q_value[next_state_[0], next_state_[1], :]) - \
                q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta
            # evaluate all the predecessors of the 4-tuple from the 
            # priority queue; those that surpass the threshold are added 
            # to the priority queue
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * \
                    np.max(q_value[state_[0], state_[1], :]) -
                                q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            
            planning_step += 1
            
            if planning_step >= dyna_params.planning_steps:
                break

        state = next_state  # 2.a

        # update the # of backups
        backups += planning_step
    
        # check if the step limit has been exceeded
        if steps >= maze.max_steps:
                break
    
    # print('steps: %i, backups: %i' % (steps, backups))

    return steps, backups
    

def dist_from_start(start, state):  
    """
    This function computes the Manhattan distance between a state
    and the start state for maze.
    Paramters
    ---------
    start : The start state for the maze. \n
    state : The state which is being evaluated.
    Returns
    -------
    The Manhattan distance.
    """
    x1, y1 = start
    x2, y2 = state
    # return max(abs(x1+y1-(x2+y2)), abs(x1-y1 -(x2-y2)))
    return abs(x2-x1)+abs(y2-y1)


def dyna_f(q_value, model, maze, dyna_params, start):
    """
    This function plays a single episode of the focused Dyna algorithm.
    Parameters
    ----------
    q_value : The set of q-values to be updated. \n
    model : An instance of the LFDyna model used for prioritized 
    Q-planning. \n
    maze : An instance of the Maze class. \n
    dyna_params : An instance of the DynaParams class.
    start : The start state for the maze.
    Returns
    -------
    steps : The number of steps taken during the episode. \n
    backups : The number of backups which took place during the episode.
    """
    state = start  # 2.a
    steps = 0
    backups = 0

    while state not in maze.GOAL:
        # track the steps
        steps += 1
        # get action (2.b)
        action = choose_action(state, q_value, maze, dyna_params)
        # take action (2.c)
        next_state, reward = maze.step(state, action)
        # real experience - feed the model with experience (2.d)
        model.feed(state, action, next_state, reward)
        # get the priority for current state-action pair (2.e)
        priority = (dyna_params.gamma**dist_from_start(start, state)) * (reward +
                        dyna_params.gamma * np.max(q_value[next_state[0], 
                        next_state[1], :]) - q_value[state[0], state[1], action])
        # add item to the priority queue
        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # begin planning (2.f)
        planning_step = 0

        # simulated experience - planning using priority queue
        while not model.empty():
            # get the 4-tuple with highest priority from the priority queue
            priority, state_, action_, next_state_, reward_ = model.pop_queue()
            # update the Q-value for the 4-tuple from the priority queue
            delta = reward_ + dyna_params.gamma * \
                np.max(q_value[next_state_[0], next_state_[1], :]) - \
                q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta
            # evaluate all the predecessors of the 4-tuple from the
            # priority queue; those that surpass the threshold are added
            # to the priority queue
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = (dyna_params.gamma**dist_from_start(start, state_pre)) * \
                                (reward_pre + dyna_params.gamma * 
                                np.max(q_value[state_pre[0], state_pre[1], :])
                                 - q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    # print('State: %s, Predecessor: %s, ' \
                    #     'Pred distance from start: %i, Priority: %f' %
                    #     (state_, state_pre, dist_from_start(start, state_pre), \
                    #         priority))
                    model.insert(priority, state_pre, action_pre)

            planning_step += 1

            if planning_step >= dyna_params.planning_steps:
                break

        state = next_state  # 2.a

        # update the # of backups
        backups += planning_step

        # check if the step limit has been exceeded
        if steps >= maze.max_steps:
            break

    # print('steps: %i, backups: %i' % (steps, backups))

    return steps, backups


def update_model(maze, model):
    """
    This function updates a model with new action-state pairs after a 
    shortcut is introduced into a maze.
    Parameters
    ----------
    maze : The maze which has been modified.
    model : The model being considered.
    """
    # going into new state
    for state, action in zip([[17, 24], [18, 25], [17, 26]], 
                            [maze.ACTION_RIGHT, maze.ACTION_UP, maze.ACTION_LEFT]):
        model.feed(state, action, [17, 25], 0)
    # going out of new state
    for action, next_state in zip([maze.ACTION_LEFT, maze.ACTION_DOWN, maze.ACTION_RIGHT],
                                [[17, 24], [18, 25], [17, 26]]):
        model.feed([17, 25], action, next_state, 0)
