from gymnasium import spaces
import numpy as np
import random as rd

from src.utils.math import euclidean_distance
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display=False):
    scenario, scenario_id = load_default_scenario_components(method,scenario_id)
    components = {  'robot':scenario['robot'],
                    'opponent':scenario['opponent'],
                    'states':scenario['states'] }

    env = TagEnv(components=components,display=display)

    env.name = 'TagEnv'+str(scenario_id)
    return env, scenario_id

def load_default_scenario_components(method,scenario_id):
    default_scenarios_components = [
        {
        # Classic Tag Problem
        # - Citation
        # Pineau, Joelle, Geoff Gordon, and Sebastian Thrun. 
        # "Point-based value iteration: An anytime algorithm for POMDPs." 
        # IJCAI. Vol. 3. 2003.
        'robot': Agent('R',method,(0,0)),
        'opponent': Agent('O','opponent',(5,1)),
        'states':[(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),\
                  (0,1),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1),\
                                                (5,2),(6,2),(7,2),\
                                                (5,3),(6,3),(7,3),\
                                                (5,4),(6,4),(7,4) ]
        },
    ]

    if scenario_id >= len(default_scenarios_components):
        print('There is no default scenario with id '+str(scenario_id)+
                ' for the Tag problem. Setting scenario_id to 0.')
        scenario_id = 0
    else:
        print('Loading scenario',scenario_id,'.')

    return default_scenarios_components[scenario_id], scenario_id


"""
    Support classes
"""
class Agent(AdhocAgent):
    """Agent : Main reasoning Component of the Environment. 
     + Derives from AdhocAgent Class
    """

    def __init__(self, index, atype, position):
        super(Agent, self).__init__(index, atype)
        self.position = position
        self.memory = set()
        self.tagged = False

    def copy(self):
        # 1. Initialising the agent
        x,y = self.position
        copy_agent = Agent(self.index, self.type, (x,y))

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters
        copy_agent.memory = set()
        for elem in self.memory:
            copy_agent.memory.add(elem)
        copy_agent.tagged = self.tagged

        return copy_agent

    def show(self):
        print(self.index, self.type, ':', self.position)
        
    def show_memory(self):
        raise NotImplementedError

"""
    Customising the Level-Foraging Env
"""
def end_condition(state):
    return state.components['opponent'].tagged

def do_action(action,env):
    info = {'movement reward':0,'tag reward':0}

    robot = env.get_robot()
    rpos = robot.position
    robot.memory.add(rpos)

    opponent = env.components['opponent']
    opos = opponent.position

    # ROBOT
    # 1. Calculating the new position
    if action != 4:
        info['movement reward'] = -0.1
        if action == 2:  # N
            rnew_pos = (rpos[0], rpos[1] + 1)
        elif action == 3:  # S
            rnew_pos = (rpos[0], rpos[1] - 1)
        elif action == 1:  # W
            rnew_pos = (rpos[0] - 1, rpos[1]) 
        elif action == 0:  # E
            rnew_pos = (rpos[0] + 1, rpos[1]) 
    else:
        rnew_pos = (rpos[0], rpos[1])
    
    # checking if it is a valid possition
    if rnew_pos not in env.components['states']:
        rnew_pos = rpos

    # 2. Updating position and state
    robot.position = rnew_pos
    env.state = rnew_pos

    # Calculating tag reward
    if action == 4: # Tag
        if (robot.position == opponent.position):
            info['tag reward'] += 1
            opponent.tagged = True
            env.components['opponent'].tagged = True
        else:
            info['tag reward'] -= 1

    # OPPONENT
    if not opponent.tagged:
        # 1. Calculating the new position
        # 80% of moving away from robot
        coin = np.random.uniform(0,1)
        if coin < 0.8:
            possibilities = [
                (opos[0], opos[1] + 1), # N
                (opos[0], opos[1] - 1), # S
                (opos[0] - 1, opos[1]), # W
                (opos[0] + 1, opos[1])  # E
            ]
            # checking if it is a valid possition
            for i in range(len(possibilities)):
                if possibilities[i] not in env.components['states']:
                    possibilities[i] = opos

            distance = [
                euclidean_distance(rpos,possibilities[0]),
                euclidean_distance(rpos,possibilities[1]),
                euclidean_distance(rpos,possibilities[2]),
                euclidean_distance(rpos,possibilities[3])
            ]
            original_distance = euclidean_distance(rpos,opos)
            pos2sample = []
            for i in range(len(possibilities)):
                if distance[i] >= original_distance:
                    pos2sample.append(i)
            onew_pos = possibilities[np.random.choice(pos2sample)]
        else:
            onew_pos = opos
    else:
        onew_pos = opos
        
    # 2. Updating position and state
    opponent.position = onew_pos

    if len(robot.memory) == len(env.components['states']) and \
    robot.memory.issubset(set(env.components['states'])):
        robot.memory = set()
    return env, info

def tag_transition(action, real_env):
    next_state, info = real_env, {}
    next_state, info = do_action(action, real_env)
    return next_state, info

# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return 0

# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    return copied_env

"""
    Tag Environments 
"""
class TagEnv(AdhocReasoningEnv):

    actions = [0,1,2,3,4]

    action_dict = {
        0: 'East',
        1: 'West',
        2: 'North',
        3: 'South',
        4: 'Tag'
    }

    agents_color = {
        'pomcp': 'yellow',
        'ibpomcp':'blue',
        'rhopomcp':'cyan',
    }

    def __init__(self, components, display=False):
        ###
        # Env Settings
        ###
        self.components = components
        dim_x = max([state[0] for state in components['states']])+1
        dim_y = max([state[1] for state in components['states']])+1
        self.dim = [dim_x,dim_y]

        state_set = StateSet(spaces.Tuple(\
            (spaces.Discrete(dim_x),spaces.Discrete(dim_y))), end_condition)
        transition_function = tag_transition
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(TagEnv, self).__init__(state_set, \
            transition_function, action_space, reward_function, \
            observation_space, components)
        self.name = None
        
        # Setting the inital state and components
        agent = self.get_robot()
        self.state_set.initial_state = agent.position
        self.state_set.initial_components = \
            self.copy_components(self.components)

        ###
        # Setting graphical interface
        ###
        self.screen = None
        self.display = display
        self.render_mode = "human"
        self.render_sleep = 0.5
        self.clock = None

    def reset(self):
        # Reset the state of the environment to an initial state
        self.episode = 0
        if self.state_set.initial_state is not None and self.state_set.initial_components is not None:
            self.state = (self.state_set.initial_state[0],self.state_set.initial_state[1])
            self.components = self.copy_components(self.state_set.initial_components)
            agent = self.get_robot()
            if self.display:
                self.reset_renderer()

            return self.observation_space(self.copy())

        else:
            raise ValueError("the initial state from the state set is None.")
        
    def reset_renderer(self):
        if not self.display:
            return
        self.screen = None
        self.clock = None
        self.render(self.render_mode)

    def import_method(self, agent_type):
        from importlib import import_module
        main_type = (agent_type.split('_'))[0]
        try:
            module = import_module('src.reasoning.levelbased.'+main_type)
        except:
            module = import_module('src.reasoning.'+main_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = TagEnv(components, self.display)
        copied_env.simulation = self.simulation
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.name = self.name

        # Setting the initial state
        copied_env.state = (self.state[0],self.state[1])
        copied_env.state_set.initial_state = (self.state_set.initial_state[0],self.state_set.initial_state[1])
        return copied_env

    def get_actions_list(self):
        return [action for action in self.action_dict]
    
    def get_robot(self):
        return self.components['robot']
    
    def get_observation(self):
        if self.components['robot'].position == self.components['opponent'].position:
            return  self.components['opponent'].position
        return None
    
    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]
    
    def hash_state(self):
        return hash((self.components['robot'].position[0],\
            self.components['robot'].position[1]))
    
    def hash_observation(self):
        obs = self.get_observation()
        return hash(str(obs))
    
    def observation_is_equal(self, obs):
        cur_visibility = self.get_observation()
        return cur_visibility == obs
    
    def sample_state(self, agent):
        # 1. Defining the base simulation
        u_env = self.copy()

        # 2. Setting possibilities
        opponent_possible_state = [state for state in self.components['states']]
        opponent_possible_state.remove(agent.position)
        for state in agent.memory:
            if state in opponent_possible_state:
                opponent_possible_state.remove(state)

        u_env.components['opponent'].position = rd.choice(opponent_possible_state)

        # 3. Returning the modified/sampled environment
        return u_env

    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states
    
    def render(self, mode="human"):
        #Render the environment to the screen
        ##
        # Standard Imports
        ##
        if not self.display:
            return

        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
            from gymnasium.error import DependencyNotInstalled
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        dim = self.dim
        max_dim = max(dim)
        if self.screen is None:
            self.screen_size = (dim[0]*800/max_dim,dim[1]*800/max_dim)
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
        
        robot = self.get_robot()
        
        # background
        self.surf = pygame.Surface(self.screen_size)
        self.surf.fill(self.colors['white'])
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # grid
        grid_width, grid_height = (dim[0]*700/max_dim,dim[1]*700/max_dim)
        self.grid_surf = pygame.Surface((grid_width, grid_height))
        self.grid_surf.fill(self.colors['white'])

        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                if (x,y) in self.components['robot'].memory:
                    x_b = int(x*(grid_width/dim[0]))
                    y_b = int(y*(grid_height/dim[1]))
                    gfxdraw.box(self.grid_surf,
                        pygame.Rect(x_b,y_b,(grid_width/dim[0]),(grid_height/dim[1])),
                        self.colors['green'])

        for column in range(-1,dim[1]):
            pygame.draw.line(self.grid_surf,self.colors['black'],
                                (0*grid_width,(column+1)*(grid_height/dim[1])),
                                (1*grid_width,(column+1)*(grid_height/dim[1])),
                                int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
        for row in range(-1,dim[0]):
            pygame.draw.line(self.grid_surf,self.colors['black'],
                                ((row+1)*(grid_width/dim[0]),0*grid_height),
                                ((row+1)*(grid_width/dim[0]),1*grid_height),
                                int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
            
        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                if (x,y) not in self.components['states']:
                    x_b = int(x*(grid_width/dim[0]))
                    y_b = int(y*(grid_height/dim[1]))
                    gfxdraw.box(self.grid_surf,
                        pygame.Rect(x_b,y_b,(grid_width/dim[0]),(grid_height/dim[1])),
                        self.colors['black'])
                    
        # agents
        self.components_surf = pygame.Surface((grid_width, grid_height))
        self.components_surf = self.components_surf.convert_alpha()
        self.components_surf.fill((self.colors['white'][0],self.colors['white'][1],self.colors['white'][2],0))  

        #robot
        x = int(robot.position[0]*(grid_width/dim[0]) + 0.35*(grid_width/dim[0]))
        y = int(robot.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
        r = int(0.35*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
        gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
        x = int(robot.position[0]*(grid_width/dim[0]) + 0.35*(grid_width/dim[0]))
        y = int(robot.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
        r = int(0.3*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
        if robot.type in self.agents_color:
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors[self.agents_color[robot.type]])
        else:
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['lightgrey'])
        # index
        agent_idx = str(robot.index)
        myfont = pygame.font.SysFont("Ariel", int(0.6*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
        label = myfont.render(agent_idx, True, self.colors['black'])
        x = int(robot.position[0]*(grid_width/dim[0]) + 0.25*(grid_width/dim[0]))
        y = int(robot.position[1]*(grid_height/dim[1]) + 0.3*(grid_height/dim[1]))
        label =  pygame.transform.flip(label, False, True)
        self.components_surf.blit(label, (x,y))

        #opponent
        opponent = self.components['opponent']
        x = int(opponent.position[0]*(grid_width/dim[0]) + 0.65*(grid_width/dim[0]))
        y = int(opponent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
        r = int(0.35*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
        gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
        x = int(opponent.position[0]*(grid_width/dim[0]) + 0.65*(grid_width/dim[0]))
        y = int(opponent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
        r = int(0.3*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
        gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['lightgrey'])
        # index
        agent_idx = str(opponent.index)
        myfont = pygame.font.SysFont("Ariel", int(0.6*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
        label = myfont.render(agent_idx, True, self.colors['black'])
        x = int(opponent.position[0]*(grid_width/dim[0]) + 0.45*(grid_width/dim[0]))
        y = int(opponent.position[1]*(grid_height/dim[1]) + 0.3*(grid_height/dim[1]))
        label =  pygame.transform.flip(label, False, True)
        self.components_surf.blit(label, (x,y))
            
        ##
        # Displaying
        ##
        self.grid_surf = pygame.transform.flip(self.grid_surf, False, True)
        self.components_surf = pygame.transform.flip(self.components_surf, False, True)
        self.screen.blit(self.grid_surf, (0.1*self.screen_size[0], 0.1*self.screen_size[1]))
        self.screen.blit(self.components_surf, (0.1*self.screen_size[0], 0.1*self.screen_size[1]))

        ##
        # Text
        ##
        act = self.action_dict[robot.next_action] \
            if robot.next_action is not None else None
        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Episode "+str(self.episode) + \
            " | Action: "+str(act), True, self.colors['black'])
        self.screen.blit(label, (10, 10))
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )