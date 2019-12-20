# standard libraries
from time import time

# custom libraries
from classes import *

# third-party libraries
from tqdm import tqdm
import matplotlib.pyplot as plt

RUNS = 25  # 25, 50
EPISODES = 250  # 150, 250
MAX_RESOLUTION = 8
OPTIMAL_PATH = 78  # 78, 85
PRINT_INTERVAL = 4  # for printing Q-table


def figure_7(runs, episodes):
    # instantiate maze
    maze = Maze()

    # modify maze to article specifcations
    maze.ROWS = 24
    maze.COLS = 36
    maze.START = [8, 0]

    maze.GOAL = []
    for i in range(0, 4):
        for j in range(32, 36):
            maze.GOAL.append([i, j])

    maze.OBSTACLES = []
    for i in range(4, 16):
        for j in range(8, 12):
            maze.OBSTACLES.append([i, j])

    for i in range(16, 20):
        for j in range(20, 24):
            maze.OBSTACLES.append([i, j])

    for i in range(0, 12):
        for j in range(28, 32):
            maze.OBSTACLES.append([i, j])

    # the size of q value
    maze.q_size = (maze.ROWS, maze.COLS, len(maze.actions))

    # instantiate parameters
    params = DynaParams()

    # set up models for planning
    models = [Dyna, QueueDyna, QueueDyna]
    methods = [dyna, dyna_lf, dyna_f]
    method_names = ['Dyna', 'largest-first Dyna', 'focused Dyna']

    # track the # of steps and backups
    steps = np.zeros((len(method_names), episodes, runs))
    backups = np.zeros((len(method_names), episodes, runs))

    for run in tqdm(range(runs)):
        for nameIndex in range(len(method_names)):
            print('run %d, %s' % (run, method_names[nameIndex]))
            # instantiate a Q-table
            q_value = np.zeros(maze.q_size)
            # instantiate model
            model = models[nameIndex]()
            # play for an episode
            for ep in range(episodes):
                if ep > 0:
                    ep_ = ep-1
                else:
                    ep_ = 0
                s, b = methods[nameIndex](
                    q_value, model, maze, params, maze.START)
                steps[nameIndex, ep, run] += s
                # print(str(steps))
                backups[nameIndex, ep, run] += backups[nameIndex, ep_, run] + b
                # print(str(backups))

        print('steps: \n', steps[:, :, run])
        print('backups: \n', backups[:, :, run])

    # average the number of steps and backups for each episode
    steps_mean = steps.mean(axis=2)
    backups_mean = backups.mean(axis=2)
    print('average steps: \n', steps_mean)
    print('average backups: \n', backups_mean)

    # compute the standard deviation over all runs
    steps_std = steps.std(axis=2)
    # backups_std = backups.std(axis=2)
    print('steps std dev: \n', steps_std)
    # print('backups std dev \n', backups_std)

    # convert NumPy arrays of means and std devs into separate lists
    mean_list = []
    for methodIndex in range(len(method_names)):
        mean_list.append(sorted(tuple(zip(backups_mean[methodIndex, :].tolist(), steps_mean[methodIndex, :].tolist())),
                                key=lambda tup: tup[0]))

        std_list = steps_std[methodIndex, :].tolist()

    # plot the three methods
    for nameIndex in range(len(method_names)):
        # backups, steps = zip(*mean_list[nameIndex])
        # print('average over %i runs - backups: %s, steps: %s' %
        #       (RUNS, backups, steps))
        # zip output
        plt.plot(*zip(*mean_list[nameIndex]), label=method_names[nameIndex])
    plt.xlabel('no. backups')
    plt.ylabel('steps to goal')
    plt.xscale('log')
    plt.legend()
    plt.title('Runs: %d, Episodes: %d' % (runs, episodes))
    plt.savefig('images/figure_7.png')
    # plt.show()
    plt.close()

    # plot the three methods with error bars
    for nameIndex in range(len(method_names)):
        backups, steps = zip(*mean_list[nameIndex])
        # print('average over %i runs - backups: %s, steps: %s' %
        #       (RUNS, backups, steps))
        # print('std dev over %i runs - steps: %s' % (RUNS, std_list))
        # error bar output
        plt.errorbar(backups, steps, yerr=std_list, fmt='o', markersize=4,
                     capsize=4, label=method_names[nameIndex])
    plt.xlabel('no. backups')
    plt.ylabel('steps to goal')
    plt.xscale('log')
    plt.legend()
    plt.title('Runs: %d, Episodes: %d' % (runs, episodes))
    plt.savefig('images/figure_7_error_bars.png')
    # plt.show()
    plt.close()


def figure_8(max_resolution, runs):
    # get the original maze
    original_maze = Maze()

    # instantiate parameters
    params = DynaParams()

    # configure each of the maze resolutions
    mazes = [original_maze.extend_maze(i)
             for i in range(1, MAX_RESOLUTION + 1)]

    # set up models for planning
    models = [Dyna, QueueDyna, QueueDyna]
    methods = [dyna,dyna_lf, dyna_f]
    method_names = ['Dyna', 'largest-first Dyna', 'focused Dyna']

    # track the # of backups
    backups = np.zeros((len(method_names), MAX_RESOLUTION, RUNS))

    for run in range(0, runs):
        for nameIndex, method in enumerate(method_names):
            for mazeIndex, maze in zip(range(len(mazes)), mazes):
                print('run %d, %s, maze size %d' % (
                    run, method_names[nameIndex], maze.ROWS * maze.COLS))
                # instantiate a Q-table
                q_value = np.zeros(maze.q_size)
                # instantiate model
                model = models[nameIndex]()
                # play for an episode
                zero_counter = 0 
                while True:  # try for 250 episodes
                    s, b = methods[nameIndex](
                        q_value, model, maze, params, maze.START)
                    if b == 0:
                        zero_counter += 1
                    if zero_counter >= 10:
                        print('A backup value of zero was returned 10 times. Run for %s has been stopped.' % (method))
                        break
                    backups[nameIndex, mazeIndex,
                            run] += b
                    print(str(backups))
                    # check whether the (relaxed) optimal path is found
                    if s <= 14 * maze.resolution * 1.5:
                        break

                print('backups: \n', backups[:, :, run])

    # compute the mean over all the runs
    backups_mean = backups.mean(axis=2)
    print('average backups: \n', backups_mean)

    # compute the standard deviation over all runs
    backups_std = backups.std(axis=2)
    print('backups std dev: \n', backups_std)

    # convert NumPy array of methods to seperate lists
    mean_list = []
    for methodIndex in range(len(method_names)):
        mean_list.append(sorted(tuple(zip(
            [i for i in range(1, MAX_RESOLUTION + 1)],
            backups_mean[methodIndex, :].tolist())),
            key=lambda tup: tup[0]))
        
        std_list = backups_std[methodIndex, :].tolist()

    # plot the three methods
    for nameIndex in range(len(method_names)):
        # resolution, backups = zip(*mean_list[nameIndex])
        # print('average over %i runs - resolution: %s, backups: %s' % (RUNS, resolution, backups))
        # print('std dev over %i runs - backups: %s' % (RUNS, std_list))
        plt.plot(*zip(*mean_list[nameIndex]), label=method_names[nameIndex])
    plt.xlabel('resolution')
    plt.ylabel('no. backups until optimal solution')
    plt.yscale('log')
    plt.legend()
    plt.title('Runs: %d, Episodes: %i Max. resolution: %i' %
              (runs, EPISODES, MAX_RESOLUTION))
    plt.savefig('images/figure_8.png')
    # plt.show()
    plt.close()

    # plot the three methods with error bars
    for nameIndex in range(len(method_names)):
        resolution, backups = zip(*mean_list[nameIndex])
        # print('average over %i runs - resolution: %s, backups: %s' % (RUNS, resolution, backups))
        # print('std dev over %i runs - backups: %s' % (RUNS, std_list))
        # error bar output
        plt.errorbar(resolution, backups, yerr=std_list, fmt='o', markersize=4,
                     capsize=4, label=method_names[nameIndex])
    plt.xlabel('resolution')
    plt.ylabel('no. backups until optimal solution')
    plt.yscale('log')
    plt.legend()
    plt.title('Runs: %d, Episodes: %i Max. resolution: %i' %
              (runs, EPISODES, MAX_RESOLUTION))
    plt.savefig('images/figure_8_error_bars.png')
    # plt.show()
    plt.close()


def figure_10(runs, episodes):
    # instantiate maze
    maze = Maze()

    # modify maze to article specifications
    maze.ROWS = 30
    maze.COLS = 30
    maze.START = [16, 0]
    maze.GOAL = [[0, 28], [0, 29]]

    maze.OLD_OBSTACLES = [[2, 3], [2, 4], [3, 3],
                          [5, 11], [5, 12], [5, 13], [26, 15]]
    for row in range(12, 14):
        for col in range(12, 21):
            maze.OLD_OBSTACLES.append([row, col])

    for row in range(14, 17):
        for col in range(8, 24):
            maze.OLD_OBSTACLES.append([row, col])

    for row in range(18, 26):
        for col in range(7, 25):
            maze.OLD_OBSTACLES.append([row, col])

    for row in range(0, 17):
        for col in range(25, 28):
            maze.OLD_OBSTACLES.append([row, col])

    maze.OLD_OBSTACLES.append([17, 25])
    maze.NEW_OBSTACLES = deepcopy(maze.OLD_OBSTACLES)
    maze.NEW_OBSTACLES.remove([17, 25])

    # the size of q value
    maze.q_size = (maze.ROWS, maze.COLS, len(maze.actions))

    # max steps per algorithm
    maze.max_steps = 5000

    # instantiate parameters
    params = DynaParams()

    # set up models for planning
    models = [Dyna, QueueDyna, QueueDyna]
    methods = [dyna, dyna_lf, dyna_f]
    method_names = ['Dyna', 'largest-first Dyna', 'focused Dyna']

    # track the # of steps and backups
    steps = np.zeros((len(method_names), episodes, runs))
    backups = np.zeros((len(method_names), episodes, runs))

    # new states and actions for priority queue after removing obstacle
    states = [[17, 24], [18, 25], [17, 26], [17, 25], [17, 25], [17, 25]]
    actions = [maze.ACTION_RIGHT, maze.ACTION_UP,
               maze.ACTION_LEFT, maze.ACTION_LEFT, maze.ACTION_DOWN, maze.ACTION_RIGHT]

    for run in tqdm(range(runs)):
        for nameIndex in range(len(method_names)):
            print('run %d, %s' % (run, method_names[nameIndex]))
            # set original maze
            maze.obstacles = maze.OLD_OBSTACLES
            # instantiate a Q-table
            q_value = np.zeros(maze.q_size)
            # instantiate model
            model = models[nameIndex]()
            print_counter = 0
            while True:  # loop until optimal path (episodes)
                s, _ = methods[nameIndex](
                    q_value, model, maze, params, maze.START)
                path = s
                print('The last path took %i steps.' % (path))
                print_counter += 1
                if print_counter % PRINT_INTERVAL == 0:
                    print('Q-table: \n', str(q_value))
                if path <= OPTIMAL_PATH:
                    break
            # change the obstacles
            maze.obstacles = maze.NEW_OBSTACLES
            # feed new state-action pairs into the model
            update_model(maze, model)
            # insert new state-action pairs into priority queue
            if methods[nameIndex] == dyna_lf:
                for state, action in zip(states, actions):
                    next_state, reward = maze.step(state, action)
                    priority = np.abs(reward + params.gamma *
                                      np.max(q_value[next_state[0],
                                                     next_state[1], :]) -
                                      q_value[state[0], state[1], action])
                    model.insert(priority, state, action)
            if methods[nameIndex] == dyna_f:
                for state, action in zip(states, actions):
                    next_state, reward = maze.step(state, action)
                    priority = (params.gamma**dist_from_start(maze.START, state)) * \
                        (reward + params.gamma *
                         np.max(q_value[next_state[0], next_state[1], :]) -
                         q_value[state[0], state[1], action])
                    model.insert(priority, state, action)
            # play for an episode
            for ep in range(episodes):
                if ep > 0:
                    ep_ = ep-1
                else:
                    ep_ = 0
                s, b = methods[nameIndex](
                    q_value, model, maze, params, maze.START)
                steps[nameIndex, ep, run] += s
                # print(str(steps))
                backups[nameIndex, ep, run] += backups[nameIndex, ep_, run] + b
                # print(str(backups))

            print('steps \n', steps[:, :, run])
            print('backups \n', backups[:, :, run])

    # average the number of steps and backups for each episode
    steps_mean = steps.mean(axis=2)
    backups_mean = backups.mean(axis=2)

    print('average steps \n', steps_mean)
    print('average backups \n', backups_mean)

    # compute the standard deviation over all runs
    steps_std = steps.std(axis=2)
    # backups_std = backups.std(axis=2)

    print('steps std dev \n', steps_std)
    # print('backups std dev \n', backups_std)

    # convert NumPy arrays of means and std devs into separate lists
    mean_list = []
    for methodIndex in range(len(method_names)):
        mean_list.append(sorted(tuple(zip(backups_mean[methodIndex, :].tolist(), steps_mean[methodIndex, :].tolist())),
                                key=lambda tup: tup[0]))

        std_list = steps_std[methodIndex, :].tolist()

    # plot the three methods
    for nameIndex in range(len(method_names)):
        # backups, steps = zip(*mean_list[nameIndex])
        # print('average over %i runs - backups: %s, steps: %s' %
        #       (RUNS, backups, steps))
        # zip output
        plt.plot(*zip(*mean_list[nameIndex]), label=method_names[nameIndex])
    plt.xlabel('no. backups')
    plt.ylabel('steps to goal')
    plt.xscale('log')
    plt.legend()
    plt.title('Runs: %i, Episodes: %i, Optimal Path: <= %i' % (runs, episodes, OPTIMAL_PATH))
    plt.savefig('images/figure_10_path_78.png')
    # plt.show()
    plt.close()

    # plot the three methods with error bars
    for nameIndex in range(len(method_names)):
        backups, steps = zip(*mean_list[nameIndex])
        # print('average over %i runs - backups: %s, steps: %s' %
        #       (RUNS, backups, steps))
        # print('std dev over %i runs - steps: %s' % (RUNS, std_list))
        # error bar output
        plt.errorbar(backups, steps, yerr=std_list, fmt='o', markersize=4,
                     capsize=4, label=method_names[nameIndex])
    plt.xlabel('no. backups')
    plt.ylabel('steps to goal')
    plt.xscale('log')
    plt.legend()
    plt.title('Runs: %i, Episodes: %i, Optimal Path: <= %i' %
              (runs, episodes, OPTIMAL_PATH))
    plt.savefig('images/figure_10_path_78_error_bars.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    start = time()
    # figure_7(RUNS, EPISODES)
    # figure_8(MAX_RESOLUTION, RUNS)
    figure_10(RUNS, EPISODES)
    end = time()
    print("Execution time: " + str((end-start)/60) + " minutes")
