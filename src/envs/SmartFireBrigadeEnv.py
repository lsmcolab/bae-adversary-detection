from copy import deepcopy
from math import radians
from multiprocessing import Process
import numpy as np
import os

from gymnasium import spaces
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet 

"""
    Load Scenario method
"""



"""
    Support classes and methods
"""
def parallelise(method,dim,components,max_time,drone):
    u_env = SmartFireBrigadeEnv(dim,components,max_time)
    u_env.reset()
    action, _ = method(u_env,drone)
    with open('./tmp/'+drone.index+'.csv','w') as file:
        file.write(str(action))

class Drone(AdhocAgent):

    min_velocity = 0
    max_velocity = 1
    max_steering = 23
    acceleration = 1.1
    deceleration = 0.8

    def __init__(self, index, atype, position):
        super(Drone,self).__init__(index,atype)
        self.position = position

        self.heading = 90
        self.velocity = 0.0

        self.battery = 100
        self.water = 100
    
    def copy(self):
        # 1. Initialising the agent
        cp_drone = Drone(self.index, self.type, deepcopy(self.position))
        cp_drone.heading = deepcopy(self.heading)
        cp_drone.velocity = deepcopy(self.velocity)
        cp_drone.battery = deepcopy(self.battery)
        cp_drone.water = deepcopy(self.water)

        # 2. Copying the parameters
        cp_drone.next_action = self.next_action
        cp_drone.smart_parameters = self.smart_parameters
        return cp_drone

"""
    Customising the SFB Env
"""

def end_condition(state):
    return False

def do_action(env):
    info = {}

    for drone in env.components['drones']:
        action = drone.next_action
        action_name = 'Idle' if action is None else env.action_dict[action]

        if action_name == 'Idle':
            pass
        # Movement
        elif action_name == 'Accelerate':
            drone.velocity = 1#min(drone.velocity*drone.acceleration,drone.max_velocity)\
                #if drone.velocity > 1 else drone.velocity + 0.5
        elif action_name == 'Break':
            drone.velocity = 0#max(drone.velocity*drone.deceleration,drone.min_velocity)\
                #if drone.velocity > 1 else max(drone.velocity - 0.2,drone.min_velocity)
        # Steering
        elif action_name == 'Turn-left':
            drone.heading = (drone.heading + drone.max_steering) % 360
        elif action_name == 'Turn-right':
            drone.heading = (drone.heading - drone.max_steering) % 360
        # Actions
        elif action_name == 'Extinguish':
            pass
        elif action_name == 'Communicate':
            pass
        # Nothing/Something else
        else:
            raise NotImplemented

        # updating position
        drone.position = \
            [max(min(drone.position[0] +\
                        (np.cos(radians(drone.heading))*drone.velocity),env.dim[0]-10),10),
            max(min(drone.position[1] +\
                    (np.sin(radians(drone.heading))*drone.velocity),env.dim[1]-10),10)]
        env.state[drone.index] = drone.position
    
    return env, info

def smartfirebrigade_transition(action, real_env):
    # agent planning
    adhoc_agent_index = real_env.components['drones'].index(real_env.get_adhoc_agent())

    for i in range(len(real_env.components['drones'])):
        if i != adhoc_agent_index:
            # changing the perspective
            copied_env = real_env.copy()
            copied_env.components['adhoc_agent_index'] = copied_env.components['drones'][i].index

            # generating the observable scenario
            obsavable_env = copied_env.observation_space(copied_env)

            # planning the action from agent i perspective
            if real_env.components['drones'][i].type is not None:
                planning_method = real_env.import_method(real_env.components['drones'][i].type + '_planning')
                real_env.components['drones'][i].next_action, real_env.components['drones'][i].target = \
                    planning_method(obsavable_env, real_env.components['agents'][i])
            else:
                real_env.components['drones'][i].next_action, real_env.components['drones'][i].target = \
                    real_env.action_space.sample(), None

        else:
            if isinstance(action,dict):
                real_env.components['drones'][i].next_action = action[real_env.components['drones'][i].index]
            else:
                real_env.components['drones'][i].next_action = action
            real_env.components['drones'][i].target = real_env.components['drones'][i].target

    # environment step
    next_state, info = do_action(real_env)

    # retuning the results
    return next_state, info

# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    ov = [ov for ov in state.values()]
    v = [v for v in next_state.state.values()]
    return 1/np.sqrt((v[0][0]-800)**2+(v[0][1]-400)**2)


# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    return copied_env


"""
    Infiltration Game Environments 
"""

class SFBSpace(spaces.Space):

    def __init__(self,dim,nagents):
        super(SFBSpace,self).__init__(shape=(nagents,2),dtype=dict)
        self.max_x, self.max_y = dim[0], dim[1]
        self.initial_state = {}

    def copy_initial_state(self):
        cp_initial_state = {}
        for k in self.initial_state:
            cp_initial_state[k] = [v for v in self.initial_state[k]]
        return cp_initial_state

class SmartFireBrigadeEnv(AdhocReasoningEnv):

    action_dict = {
        0: 'Idle', 
        1: 'Accelerate',
        2: 'Break',
        3: 'Turn-left',
        4: 'Turn-right',
        5: 'Extinguish',
        6: 'Communicate'
    }

    def __init__(self, dim, components, max_time, display=False):
        ###
        # Env Settings
        ###
        self.dim = dim
        self.nagents = len(components['drones'])
        self.max_time = max_time

        state_set = StateSet(SFBSpace(self.dim,self.nagents), end_condition)
        transition_function = smartfirebrigade_transition
        action_space = spaces.Discrete(len(self.action_dict))
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(SmartFireBrigadeEnv, self).__init__(state_set, \
                                               transition_function, action_space, reward_function, \
                                               observation_space, components)

        # Setting the inital state and components
        self.actions =  {}
        self.state_set.initial_state =  {}
        for d in self.components['drones']:
            self.state_set.initial_state[d.index] = [d.position[0],d.position[1]]
            self.actions[d.index] = 0
        self.state_set.initial_components = self.copy_components(components)
        self.state = self.copy_components(self.state_set.initial_state)
        self.processes = {}
        self.managers = {}

        ###
        # Setting graphical interface
        ###
        self.screen = None
        self.display = display
        self.render_mode = "human"
        self.clock = None

    def reset_renderer(self):
        if not self.display:
            return
        self.screen = None
        self.clock = None
        self.render_mode = "human"
        self.render()

    def listen_action(self):
        for drone in self.components['drones']:
            if drone.index not in self.processes:
                method = self.import_method(drone.type)
                u_env = self.get_observable_env()
                self.processes[drone.index] = Process(target=parallelise,
                    args=(method,u_env.dim,u_env.components,u_env.max_time,drone))
                self.processes[drone.index].start()

            if not self.processes[drone.index].is_alive():
                with open('./tmp/'+drone.index+'.csv') as file:
                    for l in file:
                        self.actions[drone.index] = int(l)
                method = self.import_method(drone.type)
                u_env = self.get_observable_env()
                self.processes[drone.index] = Process(target=parallelise,
                    args=(method,u_env.dim,u_env.components,u_env.max_time,drone))
                self.processes[drone.index].start()
            else:
                self.actions[drone.index] = 0
                
        return self.actions

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.smartfirebrigade.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = SmartFireBrigadeEnv(self.dim, components, self.display)
        copied_env.simulation = self.simulation
        copied_env.screen = self.screen
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state = self.copy_components(self.state)
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        return copied_env

    def sample_state(self, agent):
        # 1. Defining the base simulation
        u_env = self.copy()
        return u_env

    def observation_is_equal(self, obs):
        return True

    def get_actions_list(self):
        actions_list = []
        for key in self.action_dict:
            actions_list.append(key)
        return actions_list

    def get_adhoc_agent(self):
        for agent in self.components['drones']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        raise IndexError("Ad-hoc Index is not in Agents Index Set.")

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]

    def render(self):
        #Render the environment to the screen
        ##
        # Standard Imports
        ##
        if not self.display:
            return

        try:
            import pygame
            from gymnasium.error import DependencyNotInstalled
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
            
        ##
        # Drawing
        ##
        if self.state is None:
            return None

        self.screen_width, self.screen_height = self.dim[0], self.dim[1]
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.dim[0],self.dim[1])
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # background
        self.surf = pygame.Surface((self.screen_width, self.screen_height))

        env_ret = pygame.Rect((0,0),(self.screen_width,self.screen_height))
        env_image = pygame.image.load(os.path.abspath("./imgs/smartfirebrigade/random_forest.jpg"))
        env_image = pygame.transform.scale(env_image, env_ret .size)
        env_image = env_image.convert()
        self.surf.blit(env_image,env_ret)

        # drones
        for d in self.components['drones']:
            image = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.polygon(image, pygame.Color('red'),
                            [(10, 5), (0, 1), (0, 9)])
            image = pygame.transform.rotate(image, -d.heading)
            self.surf.blit(image,d.position)

        ##
        # Displaying
        ##
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )