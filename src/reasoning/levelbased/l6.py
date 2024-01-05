from src.reasoning.a_star import a_star_planning
import numpy as np
import random as rd

#####
# LEADER 6 ALGORITHM
#####
# returns the action to lead to a random visible task
def l6_planning(env, agent):
	# 1. Building the state to path
	state = build_state(env.shape,env.components)

	# 2. Choosing a target
	# if no target defined, choose one
	if agent.target is None:
		target_position = l6_choose_target(state)
		agent.target = target_position
	# else maintain the current target
	else:
		target_position = agent.target

	# 3. Planning the action/route to the target
	# if the target exists
	if target_position is not None:
		next_action = a_star_planning(state, env.shape[0], env.shape[1],
							env.action_space, agent.position, target_position)
	# else, take a random action
	else:
		next_action = env.action_space.sample()

	# 4. Verifying if the agent's next action completes a task
	if agent.direction == np.pi/2:
		pos = (agent.position[0],agent.position[1]+1)
	elif agent.direction == 3*np.pi/2:
		pos = (agent.position[0],agent.position[1]-1)
	elif agent.direction == 0:
		pos = (agent.position[0]+1,agent.position[1])
	elif agent.direction == np.pi:
		pos = (agent.position[0]-1,agent.position[1])

	# if it is possible, load
	if pos == target_position:
		agent.target = target_position
		return 4, target_position
	# else, keep moving
	return next_action,target_position

# Returns a random visible task
def l6_choose_target(state):
    # 1. Collecting the visible
    visible_tasks = [(x,y) for x in range(state.shape[0]) 
                        for y in range(state.shape[1]) if state[x,y] == np.inf]

    # 2. Picking a random task
    if len(visible_tasks) != 0:
        return rd.sample(visible_tasks,1)[0]
    else:
        return None

def build_state(shape,components):
	state = np.zeros(shape)
	for ag in components['agents']:
		state[ag.position[0],ag.position[1]] = 1

	for tk in components['tasks']:
		if not tk.completed:
			state[tk.position[0],tk.position[1]] = np.inf

	if 'obstacles' in components:
		for ob in components['obstacles']:
			state[ob[0],ob[1]] = -1
			
	return state
