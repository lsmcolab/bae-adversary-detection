from gymnasium import spaces
import numpy as np
import os

from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display=False):
    scenario, scenario_id = load_default_scenario_components(method,scenario_id)

    type_knowledge = scenario['type_knowledge']
    parameter_knowledge = scenario['parameter_knowledge']
    components = {  
        'states':scenario['states'], 'transitions':scenario['transitions'], 'vents':scenario['vents'],
        'agents':scenario['agents'], 'adhoc_agent_index':scenario['adhoc_agent_index'], 'tasks':scenario['tasks']}

    env = AImongUsEnv(components=components,display=display,\
        type_knowledge=type_knowledge,parameter_knowledge=parameter_knowledge)

    env.name = 'AmongUsEnv'+str(scenario_id)
    return env, scenario_id

def load_default_scenario_components(method,scenario_id):
    default_scenarios_components = [
        {
        # Scenario 0: The Skeld
        'states' : ['Cafeteria','Admin','Weapons','O2','Navigation','Shields',
            'Communication','Storage','LowerEngine','Reactor','Security',
            'UpperEngine','Medbay'],
        'transitions' : [
            ['Cafeteria','Cafeteria'],['Cafeteria','UpperEngine'],\
            ['Cafeteria',   'Medbay'],['Cafeteria',    'Weapons'],\
            ['Cafeteria',    'Admin'],['Cafeteria',    'Storage'],\
                ['Admin','Admin'],['Admin','Cafeteria'],['Admin','Storage'],\
            ['Weapons',   'Weapons'],['Weapons','Cafeteria'],['Weapons','02'],\
            ['Weapons','Navigation'],['Weapons',  'Shields'],\
                ['O2','O2'],['O2','Weapons'],['O2','Navigation'],['O2','Shields'],\
            ['Navigation','Navigation'],['Navigation',     'O2'],\
            ['Navigation',   'Weapons'],['Navigation','Shields'],\
                ['Shields',      'Shields'],['Shields','Navigation'],\
                ['Shields',      'Weapons'],['Shields',        'O2'],\
                ['Shields','Communication'],['Shields',   'Storage'],\
            ['Communication','Communication'],['Communication','Shields'],\
            ['Communication',      'Storage'],\
                ['Storage',    'Storage'],['Storage','Communication'],\
                ['Storage',    'Shields'],['Storage',        'Admin'],\
                ['Storage',  'Cafeteria'],['Storage',    'Eletrical'],\
                ['Storage','LowerEngine'],\
                ['Eletrical','Eletrical'],['Eletrical','Storage'],\
                ['Eletrical','LowerEngine'],\
            ['LowerEngine','LowerEngine'],['LowerEngine', 'Electrical'],\
            ['LowerEngine',    'Storage'],['LowerEngine',   'Security'],\
            ['LowerEngine',    'Reactor'],['LowerEngine','UpperEngine'],\
                ['Reactor', 'Reactor'],['Reactor','LowerEngine'],\
                ['Reactor','Security'],['Reactor','UpperEngine'],\
            ['Security','Security'],['Security','LowerEngine'],\
            ['Security', 'Reactor'],['Security','UpperEngine'],\
                ['UpperEngine','UpperEngine'],['UpperEngine','LowerEngine'],\
                ['UpperEngine',    'Reactor'],['UpperEngine',   'Security'],\
                ['UpperEngine',     'MedBay'],['UpperEngine',  'Cafeteria'],\
            ['Medbay','Medbay'],['Medbay','UpperEngine'],['Medbay','Cafeteria']
        ],
        'vents': [ 
            ['Security', 'Eletrical'],['Security',    'Medbay'],\
            ['Eletrical', 'Security'],['Eletrical',   'Medbay'],\
            ['Medbay',    'Security'],['Medbay',   'Eletrical'],\
            ['Cafeteria',  'Shields'],['Shields',  'Cafeteria'],\
            ['Shields',      'Admin'],['Admin',      'Shields']
        ],
        'type_knowledge': False,
        'parameter_knowledge': False,
        'agents' : [
            Agent(index='A',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='2',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='3',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='4',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='1',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='5',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='6',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='X',atype=method,position='Cafeteria',visibility=1.), 
            Agent(index='Z',atype=method,position='Cafeteria',visibility=1.), 
                ],
        'adhoc_agent_index' : 'A',
        'tasks' : { 'Reactor1'   :False,'Reactor2'   :False,\
                    'UpperEngine':False,'LowerEngine':False,\
                    'Security'   :False,'Medbay'     :False,\
                    'Eletrical'  :False,'Storage'    :False,\
                    'Admin'      :False,'O2'         :False,\
                    'Weapons'    :False,'Shields'    :False,\
                    'Cafeteria'  :False,'Navigation' :False},
        },
    ]

    if scenario_id >= len(default_scenarios_components):
        print('There is no default scenario with id '+str(scenario_id)+
                ' for the AImongUs problem. Setting scenario_id to 0.')
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

    def __init__(self, index, atype, position, visibility,estimation_method=None):
        super(Agent, self).__init__(index, atype)

        # agent parameters
        self.position = position
        self.visibility = visibility
        if estimation_method is not None:
            self.smart_parameters['estimation_method'] = estimation_method

    def copy(self):
        # 1. Initialising the agent
        if 'estimation_method' in self.smart_parameters:
            copy_agent = Agent(self.index, self.type,\
                self.position, self.visibility, self.smart_parameters['estimation_method'])
        else:
            copy_agent = Agent(self.index, self.type,\
                self.position, self.visibility )

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters

        return copy_agent

    def set_parameters(self, parameters):
        self.visibility = parameters[0]

    def get_parameters(self):
        return np.array([self.visibility])

    def show(self):
        print(self.index, self.type, ':', self.position, self.visibility)

"""
    Customising the Level-Foraging Env
"""
def end_condition(state):
    return False

def do_action(env):
    info = {}
    return env, info

def aimongus_transition(action, real_env):
    next_state, info = None, {}
    return next_state, info

def reward(state, next_state):
    return 0

def environment_transformation(copied_env):
    return copied_env


"""
    AImongUs Environments 
"""
class AImongUsEnv(AdhocReasoningEnv):

    actions = [0,1,2,3,4]
    action_dict = {
        0: 'East',
        1: 'West',
        2: 'North',
        3: 'South',
        4: 'Load'
    }

    color = [
        'red','green','blue','cyan',\
        'magenta','yellow','brown','white','lightgrey'
    ]

    def __init__(self, components, display=False, \
     type_knowledge=True, parameter_knowledge=True):
        ###
        # Env Settings
        ###
        self.type_knowledge = type_knowledge
        self.parameter_knowledge = parameter_knowledge

        self.reasoning_turn = 'adhoc'

        state_set = StateSet(spaces.Tuple(\
            (spaces.Discrete(10),spaces.Discrete(10))), end_condition)
        transition_function = aimongus_transition
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(AImongUsEnv, self).__init__(state_set, \
            transition_function, action_space, reward_function, \
            observation_space, components)
        self.name = None

        # Checking components integrity

        # Setting the inital state and components
        agent = self.get_adhoc_agent()
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
        copied_env = AImongUsEnv(components, self.display,\
            self.type_knowledge, self.parameter_knowledge)
        copied_env.simulation = self.simulation
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.reasoning_turn = self.reasoning_turn
        copied_env.name = self.name

        # Setting the initial state
        copied_env.state = (self.state[0],self.state[1])
        copied_env.state_set.initial_state = (self.state_set.initial_state[0],self.state_set.initial_state[1])
        return copied_env
    
    def get_adhoc_agent(self):
        for agent in self.components['agents']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        raise IndexError("Ad-hoc Index is not in Agents Index Set.")
    
    def get_actions_list(self):
        return [action for action in self.action_dict]

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
        
        self.screen_width, self.screen_height = 1200, 720
        if self.screen is None:
            self.screen_size = (self.screen_width,self.screen_height)
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

        env_ret = pygame.Rect((0,0),(self.screen_width,self.screen_height))
        env_image = pygame.image.load(os.path.abspath("./imgs/aimongus/theskeld.PNG"))
        env_image = pygame.transform.scale(env_image, env_ret.size)
        env_image = env_image.convert()
        self.surf.blit(env_image,env_ret)

        self.components_surf = pygame.Surface((self.screen_width, self.screen_height))
        self.components_surf = self.components_surf.convert_alpha()
        self.components_surf.fill((self.colors['white'][0],self.colors['white'][1],self.colors['white'][2],0))
        # agents
        map_position_mod = {
            'Cafeteria'    : [630,160],
            'Admin'        : [790,430],
            'Weapons'      : [890,150],
            'O2'           : [830,290],
            'Navigation'   : [1100,340],
            'Shields'      : [910,520],
            'Communication': [760,600],
            'Storage'      : [600,540],
            'Electrical'   : [420,460],
            'LowerEngine'  : [180,510],
            'UpperEngine'  : [180,170],
            'Medbay'       : [420,280],
            'Reactor'      : [90,340],
            'Security'     : [300,340],
        }

        agent_size = [20,30]
        agents_position_mod = [
            [-2*agent_size[0]  , agent_size[1]],
            [-1*agent_size[0]  , agent_size[1]],
            [ 0*agent_size[0]  , agent_size[1]],
            [ 1*agent_size[0]  , agent_size[1]],
            [ 2*agent_size[0]  , agent_size[1]],
            [-1.5*agent_size[0],-agent_size[1]],
            [-0.5*agent_size[0],-agent_size[1]],
            [ 0.5*agent_size[0],-agent_size[1]],
            [ 1.5*agent_size[0],-agent_size[1]]
        ]

        for i in range(len(self.components['agents'])):
            agent = self.components['agents'][i]
            position =  np.random.choice(self.components['states']) # agent.position
            agent_screen_pos = (map_position_mod[position][0] + agents_position_mod[i][0],\
                 map_position_mod[position][1] + agents_position_mod[i][1])

            pygame.draw.rect(self.components_surf,self.colors['darkgrey'],\
                pygame.Rect(
                    (agent_screen_pos[0]-5,agent_screen_pos[1]-5),
                    (agent_size[0]+10,agent_size[1]+10)))
                    
            pygame.draw.rect(self.components_surf,self.colors[self.color[i]],\
                pygame.Rect(agent_screen_pos,(agent_size[0],agent_size[1])))
            
            pygame.draw.rect(self.components_surf,self.colors['darkgrey'],\
                pygame.Rect(
                    (agent_screen_pos[0]+0.35*agent_size[0],agent_screen_pos[1]+0.2*agent_size[1]),
                    (0.7*agent_size[0],0.4*agent_size[1])))
            
            pygame.draw.rect(self.components_surf,self.colors['white'],\
                pygame.Rect(
                    (agent_screen_pos[0]+0.4*agent_size[0],agent_screen_pos[1]+0.25*agent_size[1]),
                    (0.6*agent_size[0],0.3*agent_size[1])))
        ##
        # Displaying
        ##
        self.surf = pygame.transform.flip(self.surf, False, False)
        self.screen.blit(self.surf, (0, 0))
        self.components_surf = pygame.transform.flip(self.components_surf, False, False)
        self.screen.blit(self.components_surf, (0, 0))

        ##
        # Text
        ##
        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Episode "+str(self.episode), True, self.colors['white'])
        self.screen.blit(label, (10, 10))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )