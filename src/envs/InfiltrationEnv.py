from copy import deepcopy
from importlib import import_module
from enum import IntEnum

from src.reasoning.infiltration.waypoints import Waypoints
# from src.reasoning.infiltration.boid import Boid
import numpy as np
import random as rd

from gymnasium import spaces
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet
from src.spaces.circle import Circle

ENTITY_SIZE_SQ = 25
MAX_SPEED = 3.0
MAX_SPEED_SQ = MAX_SPEED*MAX_SPEED
MAX_FORCE = 0.3
MAX_FORCE_SQ = MAX_FORCE*MAX_FORCE
MAX_STEER = 50*np.pi/180
MAX_STEER_SQ = MAX_STEER*MAX_STEER
MAX_STEER_ROT_CCW = np.array([[np.cos(MAX_STEER), -np.sin(MAX_STEER)], [np.sin(MAX_STEER), np.cos(MAX_STEER)]])
MAX_STEER_ROT_CW = np.array([[np.cos(MAX_STEER), np.sin(MAX_STEER)], [-np.sin(MAX_STEER), np.cos(MAX_STEER)]])

class ForceTypes(IntEnum):
    """Force types."""
    RADIUS = 0
    WEIGHT = 1

class BoidParams(IntEnum):
    """Boid forces"""
    ALIGNMENT = 0
    COHESION = 1
    SEPARATION = 2
    ATTRACTION = 3

    @staticmethod
    def create_obj(ali_r, coh_r, sep_r, ali_w, coh_w, sep_w, att_w):
        """Create tuple of 2 ndarray with given parameters. First axis indices are members of ForceTypes"""
        return (        
            np.array([ali_r, coh_r, sep_r]),
            np.array([ali_w, coh_w, sep_w, att_w])
        )


    @classmethod
    def get_default(cls):
        """Create (2,4) ndarray with default parameters. First axis indices are members of ForceTypes"""
        return cls.create_obj(  cls.get_default_radius(cls.ALIGNMENT), cls.get_default_radius(cls.COHESION), \
                                cls.get_default_radius(cls.SEPARATION), \
                                cls.get_default_weight(cls.ALIGNMENT), cls.get_default_weight(cls.COHESION), \
                                cls.get_default_weight(cls.SEPARATION), cls.get_default_weight(cls.ATTRACTION)
        )

    def get_default_weight(self):
        """Get default weight of force"""
        if self == BoidParams.ALIGNMENT:
            return 1.2
        elif self == BoidParams.COHESION:
            return 0.9
        elif self == BoidParams.SEPARATION:
            return 2.0
        else: # self == BoidParams.ATTRACTION:
            return 1.1

    def get_default_radius(self):
        """Get default radius of force"""
        if self == BoidParams.ALIGNMENT:
            return 70.0
        elif self == BoidParams.COHESION:
            return 70.0
        elif self == BoidParams.SEPARATION:
            return 30.0
        else: # self == BoidParams.ATTRACTION:
            raise ValueError('No default radius for attraction force')

class Difficulties(IntEnum):
    """
        Enum representing the different path settings for the defenders
    """
    EASY = 0
    MEDIUM = 1
    HARD = 2
    RANDOM = 3 # this one for debugging any waypoint learning and ensuring no overfitting

    def __str__(self):
        return self.name

    def get_waypoints(self):
        if self == Difficulties.EASY:
            return np.array([(-100, 50), (200, 0), (-100, -95), (40, -195)])
        elif self == Difficulties.MEDIUM:
            return np.array([(-100, 50), (100, 0), (40, -195)])
        elif self == Difficulties.HARD:
            return np.array([(0, -15), (0, 15)])
        elif self == Difficulties.RANDOM:
            return np.array([(rd.randint(-100, 100), rd.randint(-100, 100)) for _ in np.arange(5)])
        else:
            raise NotImplemented

"""
    Load Scenario method
"""
# def load_default_scenario(method,scenario_id=Difficulties.EASY,display=False):
#     return
    # scenario, scenario_id = load_default_scenario_components(method,scenario_id)

    # dim = scenario['dim']
    # visibility = scenario['visibility']
    # components = {'agents':scenario['agents'],'adhoc_agent_index':scenario['adhoc_agent_index'],'tasks':scenario['tasks']}
    # env = InfiltrationEnv(shape=dim,components=components,visibility=visibility,display=display)
    # return env, scenario_id

# def load_default_scenario_components(method,scenario_id):
#     return
    # if scenario_id >= Difficulties.RANDOM:
    #     print('There is no default scenario with id '+str(scenario_id)+' for the LevelForaging problem. Setting scenario_id to 0 (EASY)')
    #     scenario_id = Difficulties.EASY

    # default_scenarios_component ={
    #     # Scenario 0: 1 attacker, numerous defenders, Easy Waypoints 
    #     'attackers' : [
    #         Agent(index='A',atype=method,position=(1,1),direction=1*np.pi/2,radius=0.25,angle=1,level=1.0), 
    #             ],
    #     'waypoints' : Difficulties.EASY.get_waypoints(),
    #     }

    # return default_scenarios_component, scenario_id



"""
    Support classes
"""
class Flock(AdhocAgent):
    def __init__(self, index, atype, positions, velocities, waypoints, waypoint_idx=1, flock_params=BoidParams.get_default()):
        super(Flock, self).__init__(index, atype)

        # agent parameters
        #
        # self.boids = boids #array of Boid
        self.flock_state = np.array([
            positions,
            velocities,
        ], dtype="float64")
        self.next_action = np.zeros_like(self.flock_state[1])
        self.smart_parameters[atype+"_params"] = flock_params
        self.smart_parameters["waypoints"] = waypoints
        self.smart_parameters["waypoint_idx"] = waypoint_idx
        self.smart_parameters[atype] = None

    @classmethod
    def generate_flock(cls,index,atype,waypoints,flock_params,agent_count):
        if len(waypoints) < 2:
            raise IOError(waypoints,'is an invalid object passed as \'waypoints\' argument.')
        to_waypoint = waypoints[1] - waypoints[0] + np.random.default_rng().uniform(-0.5,0.5,2*agent_count).reshape((agent_count,2))
        to_waypoint /= np.linalg.norm(to_waypoint,axis=1,keepdims=True)
        to_waypoint *= MAX_SPEED
        # to_waypoint = to_waypoint + np.random.default_rng().uniform(-0.5,0.5,2*agent_count).reshape((agent_count,2))
        return Flock(
            index,
            atype,
            waypoints[0]+np.random.default_rng().uniform(-20,20,2*agent_count).reshape((agent_count,2)),
            to_waypoint,
            waypoints,
            1,
            flock_params
        )

    @property
    def current_waypoint(self):
        waypoints = self.smart_parameters["waypoints"]
        waypoint_idx = self.smart_parameters["waypoint_idx"]
        return waypoints[waypoint_idx] if waypoints is not None else self.center_of_mass

    @property
    def center_of_mass(self):
        # calculates center of mass
        return self.positions.mean(0)

    @property
    def positions(self):
        return self.flock_state[0]

    @property
    def velocities(self):
        return self.flock_state[1]

    def get_action(self):
        self.next_action = self.smart_parameters[self.type].get_action()
        return self.next_action

    def do_action(self):
        self.flock_state[1] += self.next_action
        for i, vel in enumerate(self.flock_state[1]):
            vel_mag_sq = np.dot(vel,vel)
            if vel_mag_sq > MAX_SPEED_SQ:
                self.flock_state[1][i] = vel/np.sqrt(vel_mag_sq)*MAX_SPEED
        
        self.flock_state[0] += self.flock_state[1]
        self.next_action = np.zeros_like(self.flock_state[1])

    def update_params(self, flock_params):
        self.smart_parameters[self.type].update_params(flock_params)

    def copy(self):
        # 1. Initialising the agent

        flock = Flock(self.index, self.type, self.flock_state[0], self.flock_state[1], \
            self.smart_parameters["waypoints"], self.smart_parameters["waypoint_idx"], self.smart_parameters[self.type+"_params"])

        # 2. Copying the parameters
        flock.next_action = self.next_action
        flock.smart_parameters = self.smart_parameters
        return flock

class Agent(AdhocAgent):
    """Agent : Main reasoning Component of the Environment. Derives from AdhocAgent Class
    """

    def __init__(self, index, atype, position, velocity):
        super(Agent, self).__init__(index, atype)

        # agent parameters
        self.position = np.array(position,dtype=np.float64)    
        self.velocity = np.array(velocity,dtype=np.float64)

        # smart parameters (i assume this is to estimate waypoints?)
        self.smart_parameters[atype] = atype
        self.smart_parameters["waypoints"] = Waypoints()

    def get_state_repr(self):# returns 2-tuple of ndarray with agent position and velocity
        return [[self.position], [self.velocity]]

    def get_state_repr_ndarray(self):# returns (2,2) ndarray with agent position and velocity
        return np.array(self.get_state_repr(), dtype="float64")

    def set_action(self,action):#todo correct error checking
        self.next_action = action# if action not in [None, np.nan] else np.zeros_like(self.velocity)

    def do_action(self, flock):
        self.smart_parameters["waypoints"].new_observation(flock.flock_state[0])
        self.velocity += self.next_action
        self.position += self.velocity
        self.next_action = np.zeros_like(flock.positions)

    def is_waypoints_learned(self):
        return self.smart_parameters["waypoints"].is_waypoints_estimated

    def copy(self):
        # 1. Initialising the agent
        copied_agent = Agent(self.index, self.type, self.position.copy(), self.velocity.copy())

        # 2. Copying the parameters
        copied_agent.next_action = self.next_action
        copied_agent.smart_parameters = self.smart_parameters
        return copied_agent

"""
    Customising the  Env
"""

def infiltration_transition(action, env):
    # agent planning
    adhoc_agent = env.get_adhoc_agent()
    adhoc_agent_index = env.components['boids'].index(adhoc_agent)
    flock = env.components['boids'][0]# TODO change to approx agent flock model


    for i in range(len(env.components['boids'])):
        agent = env.components['boids'][i]
        # planning the action from agent i perspective
        if i == adhoc_agent_index:
            agent.set_action(action)
            agent.do_action(flock)
            continue
        if agent.type is not None:
            #try:
            module = import_module('src.reasoning.infiltration.'+agent.type)
            # except:
            #     module = import_module('src.reasoning.'+agent.type)
            planning_method = getattr(module, agent.type + '_planning')

            agent.next_action, agent.target = planning_method(env, agent)
        else:
            agent.next_action, agent.target = env.action_space.sample(), None
        
        agent.do_action()

    # environment step
    #next_state, info = do_action(env)

    # if not env.simulation:
    #     for ag in env.components['boids']:

    #next_state = update(env)
    env.state = np.concatenate((flock.flock_state, adhoc_agent.get_state_repr_ndarray(), [[env.components["target"]],[[0.0,0.0]]]),axis=1)#, [[env.state[-1]],[[0.0,0.0]]]

    # 2. Updating its components

    # retuning the results
    return env.state, {}

# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return -1 if np.any(np.square(next_state[-2,0]-next_state[0:-2,0]).sum(0) < ENTITY_SIZE_SQ) else np.inf if np.any(np.square(next_state[-2,0]-next_state[-1,0]).sum(0) < ENTITY_SIZE_SQ) else 0.0

# Changes the actual environment to partial observed environment
def environment_transformation(copied_env): #TODO Make applicable to boids
    if copied_env.simulation or copied_env.visibility == 'full':
        return copied_env

    agent = copied_env.get_adhoc_agent()
    waypoints = None #TODO swap out exact waypoints for estimated waypoints here
    boid_params = None #TODO swap out boids for whatever estimated abstraction we're using   

    if agent is not None and copied_env.visibility == 'partial':
        for i in range(len(copied_env.components['agents'])):
            if copied_env.components['boids'][i] != agent:
                copied_env.components['boids'][i].set_waypoints(waypoints)
                copied_env.components['boids'][i].smart_parameters["boid"].update_params(boid_params)

        copied_env.episode += 1
        return copied_env
    else:
        raise IOError(agent, 'is an invalid agent.')

def is_end_condition(state):
    atk_def_dist = state[0,-2]-state[0,0:-2]
    for i in range(len(atk_def_dist)):
        if np.dot(atk_def_dist[i],atk_def_dist[i]) < ENTITY_SIZE_SQ:
            return True
    return np.square(state[0,-2]-state[0,-1]).sum(0) < ENTITY_SIZE_SQ

"""
    Infiltration Game Environments 
"""


class InfiltrationEnv(AdhocReasoningEnv):

    def __init__(self,shape,components,visibility="full",display=False):
        ###
        # Env Settings
        ###

        # to be able to bodge this env to match how the states currently work
        # and make it all play nice the state is represented as an ndarray with
        # shape (no. of defenders + 2, 3, 2), where the first '+2' is +1 for
        # adhoc agent and + 1 for target. The (..., 2, 2) is for 2 vectors representing
        # position, velocity (which in each state[i] subarray will occupy indices
        # 0 and 1 respectively) each in 2 dimensions. For easy access, the one
        # adhoc attacker will occupy first element in the array and the target the
        # last. 
        # 
        # TODO this is a terrible way of doing things and we should improve the framework
        # as soon as we are reasonably able to better accomodate this scenario
        # (I.E. continuous state space)

        dim=(len(components["boids"])+2,2,2)
        state_set = StateSet(
                            spaces.Box(low=0, high=np.inf, shape=dim, dtype=np.float64),
                            is_end_condition
                            )
        transition_function = infiltration_transition
        action_space = Circle(1.0, np.float32)
        reward_function = reward
        observation_space = environment_transformation
        self.visibility = visibility
        self.shape = shape

        ###
        # Initialising the env
        ###
        super(InfiltrationEnv, self).__init__(state_set, \
                                               transition_function, action_space, reward_function, \
                                               observation_space, components)

        # Setting the inital state
        self.state_set.initial_state = np.concatenate((components["boids"][0].flock_state, components["boids"][1].get_state_repr(), [[components["target"]],[[0.0,0.0]]]), axis=1)
        # for element in components:
        #     if element == 'boids':
        #         for i, boid in enumerate(components[element]):
        #             if isinstance(Flock):
        #                 self.state_set.initial_state[i] = boid.flock_state
        #             else:
        #                 self.state_set.initial_state[i] = boid.state
        #     elif element == 'target':
        #         self.state_set.initial_state[-1][0] = components['target']
        #     elif element == 'waypoints':
        self.waypoints=components["waypoints"]

        # Setting the initial components
        self.state_set.initial_components = self.copy_components(components)

        ###
        # Setting graphical interface
        ###
        self.screen_size = self.shape if display else (0,0)
        self.screen = None
        self.display = display
        self.render_mode = "human"
        self.render_sleep = 0.5
        self.clock = None
        if self.display:
            self.render(self.render_mode)

    def reset_renderer(self):
        if not self.display:
            return
        self.screen_size = self.shape #todo remove magic numbers
        self.screen = None
        self.clock = None
        self.render(self.render_mode)

    def show_state(self):#todo fix
        for y in reversed(range(self.state.shape[1])):
            for x in range(self.state.shape[0]):
                print(self.state[x,y],end=' ')
            print()
    
    def get_flock(self):
        return self.components["boids"]

    def import_method(self, agent_type):
        try:
            module = import_module('src.reasoning.infiltration.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        # components = self.copy_components(self.components)
        copied_env = InfiltrationEnv(self.shape,self.components, self.visibility, self.display)
        copied_env.simulation = self.simulation
        copied_env.screen = self.screen
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.episode = self.episode
        return copied_env

    def get_actions_list(self):
        return [self.action_space.sample()] #todo: for the sake of compatibility for now, update reasoning algos to be able to sample themselves

    def get_adhoc_agent(self):
        for agent in self.components['boids']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        raise IndexError("Ad-hoc Index is not in Agents Index Set.")

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]
        
    def render(self, mode="human"):
        if not self.display:
            return
        ##
        # Standard Imports
        ##
        assert mode in self.metadata["render_modes"]
        from gymnasium.error import DependencyNotInstalled
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    self.screen_size
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface(self.screen_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        ##
        # Drawing
        ##
        if self.state is None:
            return None

        # background
        self.surf = pygame.Surface(self.screen_size)
        self.surf.fill(self.colors['black'])
        self.surf = pygame.transform.flip(self.surf, False, True)

        # boids
        FONT = pygame.font.SysFont(None, 24)
        imgs = [FONT.render(str(self.components['target']), True, 0xFF777777)]

        for pos in self.waypoints:
            gfxdraw.filled_circle(self.surf, int(pos[0]), int(pos[1]), 1, self.colors['yellow'])

        flock=None

        for entity in self.components['boids']:
            vel = entity.velocities if isinstance(entity, Flock) else [entity.velocity]
            pos = entity.positions if isinstance(entity, Flock) else [entity.position]
            for i in range(len(vel)):
                if(entity.index != self.components['adhoc_agent_index']):
                    flock=entity
                    imgs.append(FONT.render(str(pos[i]), True, 0xFF000000))
                direction = np.arctan2(vel[i][1],vel[i][0])
                rotation_matrix = np.array(
                    [[np.cos(direction),-np.sin(direction)],
                    [np.sin(direction), np.cos(direction)]])
                xy1 = np.dot(rotation_matrix,np.array([ 0,+4]).T)
                xy2 = np.dot(rotation_matrix,np.array([+4,+2]).T)
                xy3 = np.dot(rotation_matrix,np.array([ 0, 0]).T)
                gfxdraw.filled_trigon(self.surf,
                    int(pos[i][0]+xy1[0]), int(pos[i][1]+xy1[1]),
                    int(pos[i][0]+xy2[0]), int(pos[i][1]+xy2[1]),
                    int(pos[i][0]+xy3[0]), int(pos[i][1]+xy3[1]), self.colors['red'])

        if flock is not None:
            gfxdraw.filled_circle(self.surf, int(flock.center_of_mass[0]), int(flock.center_of_mass[1]), 1, self.colors['blue'])
            gfxdraw.filled_circle(self.surf, int(flock.current_waypoint[0]), int(flock.current_waypoint[1]), 2, self.colors['magenta'])
        # target
        gfxdraw.filled_circle(self.surf, int(self.components['target'][0]), int(self.components['target'][1]), 1, self.colors['green'])

        ##
        # Displaying
        ##
        self.screen.blit(self.surf, (0, 0))
        vpos=0
        for img in imgs:
            continue
            self.screen.blit(img, (0, vpos))
            vpos += 24
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )