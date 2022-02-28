
'''
Joseph Muffoletto
jrm7925
CS 394R, Prof. Stone
Chapter 2 Programming Assignment (Programming Assignment 1)

Apologies for the previous submissions! I definitely misred the instructions.
'''
import numpy as np
import sys 


#struct which contains the results of one or many runs
class run_data:
    def __init__(self, num_iterations) -> None:
        self.optimal_choices = np.array([0 for i in range(num_iterations)])
        self.average_rewards = np.array([0 for i in range(num_iterations)])

'''
main function
    output_path is the command line argument for where to save the results/output
    responsible for running the sample-average and constant-step data collections
'''
def entry(output_path):
    num_iterations = 10000
    num_runs = 300
    run_dat = run_data(num_iterations)

    # stddev = pow(0.01, 2)
    # sample average
    for j in range(num_runs):
        bandits = [0 for x in range(10)]
        # per step
        action_values = [0 for x in range(10)]
        action_amts = [1 for x in range(10)]
        for i in range(num_iterations):
            action, reward = bandits_iteration(bandits, action_values, i, run_dat)
            action_values[action] = action_values[action] + (1/action_amts[action] * (reward - action_values[action]))
            action_amts[action] += 1


    run_dat.optimal_choices = [i/num_runs for i in run_dat.optimal_choices]
    run_dat.average_rewards = [i/num_runs for i in run_dat.average_rewards]
    # print(bandits)
    # print(action_values)
    # constant step-size parameter
    run_dat_const = run_data(num_iterations)
    alpha = 0.1
    for j in range(num_runs):
        bandits = [0 for x in range(10)]
        action_values_step = [0 for x in range(10)]

        # per step
        # tot_reward = 0
        for i in range(num_iterations):
            action, reward = bandits_iteration(bandits, action_values_step, i, run_dat_const)
            action_values_step[action] = action_values_step[action] + (alpha * (reward - action_values_step[action]))
    
    # print(bandits)
    # print(action_values_step)
    run_dat_const.optimal_choices = [i/num_runs for i in run_dat_const.optimal_choices]
    run_dat_const.average_rewards = [i/num_runs for i in run_dat_const.average_rewards]
    # print(run_dat.optimal_choices)
    # print(run_dat.average_rewards)
    arr = np.stack([run_dat.average_rewards, run_dat.optimal_choices, run_dat_const.average_rewards, run_dat_const.optimal_choices], axis=0)
    np.savetxt(output_path, arr, fmt='%s')



'''
One time step of the bandits problem
Uses e-greedy on its action_value to select the next bandit
Samples from the bandit's reward
returns the chosen action and the earned reward
'''
def bandits_iteration(bandits, action_value, iteration, run_dat: run_data):
    action = -1
    optimal_action = bandits.index(max(bandits))
    optimal_agent_action = action_value.index(max(action_value))
    rand = np.random.random()
    # print(rand)
    if rand <= 0.1: # explore
        action = np.random.randint(0, 10)
    else:
        action = optimal_agent_action

    reward = np.random.normal(bandits[action], 1)
    run_dat.average_rewards[iteration] += reward
    if optimal_action == action:
        run_dat.optimal_choices[iteration] += 1
    
    # update rewards
    for idx in range(len(bandits)):
        noise = np.random.normal(0.0, 0.01)
        bandits[idx] = bandits[idx] + noise

    return action, reward


if __name__ == "__main__":
    entry(sys.argv[1]) #result path