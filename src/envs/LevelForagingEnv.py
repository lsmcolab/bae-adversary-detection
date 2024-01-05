from gymnasium import spaces
import numpy as np
import random as rd
import os

from src.utils.math import euclidean_distance, angle_of_gradient
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet


PRIOR_OBSTACLES_KNOWLEDGE = True
"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display=False):
    scenario, scenario_id = load_default_scenario_components(method,scenario_id)

    dim = scenario['dim']
    type_knowledge = scenario['type_knowledge']
    parameter_knowledge = scenario['parameter_knowledge']
    vision_block = scenario['vision_block']
    components = {  'agents':scenario['agents'],'adhoc_agent_index':scenario['adhoc_agent_index'],
                    'template_types':scenario['template_types'],
                    'tasks':scenario['tasks'],'impostor_index':scenario['impostor_index']}

    components['obstacles'] = scenario['obstacles'] \
        if 'obstacles' in scenario else []

    env = LevelForagingEnv(shape=dim,components=components,display=display,\
        type_knowledge=type_knowledge,parameter_knowledge=parameter_knowledge,\
            vision_block=vision_block)

    env.name = 'LevelForagingEnv'+str(scenario_id)
    return env, scenario_id

def load_default_scenario_components(method,scenario_id):
    default_scenarios_components = [
        {
        # Scenario 0: The Corridor (Rectangle PO Foraging Scenario)
        'dim': (20,2),
        'type_knowledge': True,
        'parameter_knowledge': True,
        'vision_block': True,
        'agents' : [
            Agent(index='A',atype=method,position=(1,1),
                    direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0), 
                ],
        'adhoc_agent_index' : 'A',
        'impostor_index': None,
        'tasks' : [
            Task(index='0',position=(0,0),level=1.0),
            Task(index='4',position=(19,1),level=1.0)
                ],
        'obstacles':[]
        },
        # Scenario 1: The U-shaped Scenario
        {
            'dim': (15, 15),
            'type_knowledge': True,
            'parameter_knowledge': True,
            'vision_block': True,
            'agents': [
            Agent(index='0',atype=method,position=(14,14),direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0),
            ],
            'adhoc_agent_index': '0',
            'impostor_index': None,
            'tasks': [
                    Task(index='0',position=(0,7),level=1.0),
                    Task(index='1',position=(14,0),level=1.0),
                    Task(index='2',position=(14,12),level=1.0),
            ],
            'obstacles': [
            (3, 3) , (3, 4) , (3, 5) , (3, 6) , (3, 7) , (3, 8) , (3, 9) , (3, 10) , (3, 11) , (4, 3) , (4, 4) , (4, 5) , (4, 6) , (4, 7) , (4, 8) , (4, 9) , (4, 10) , (4, 11) , (5, 3) , (5, 4) , (5, 5) , (5, 6) , (5, 7) , (5, 8) , (5, 9) , (5, 10) , (5, 11) , (6, 3) , (6, 4) , (6, 5) , (6, 6) , (6, 7) , (6, 8) , (6, 9) , (6, 10) , (6, 11), (7, 3) , (7, 4) , (7, 5) , (7, 6) , (7, 7) , (7, 8) , (7, 9) , (7, 10) , (7, 11) , (8, 3) , (8, 4) , (8, 5), (8, 6) , (8, 7) , (8, 8) , (8, 9) , (8, 10) , (8, 11) , (9, 3) , (9, 4) , (9, 5) , (9, 6) , (9, 7) , (9, 8), (9, 9) , (9, 10) , (9, 11) , (10, 3) , (10, 4) , (10, 5) , (10, 6) , (10, 7) , (10, 8) , (10, 9) , (10, 10) , (10, 11) , (11, 3) , (11, 4) , (11, 5) , (11, 6) , (11, 7) , (11, 8) , (11, 9) , (11, 10) , (11, 11) , (12, 3) , (12, 4) , (12, 5) , (12, 6) , (12, 7) , (12, 8) , (12, 9) , (12, 10) , (12, 11) , (13, 3) , (13, 4) , (13, 5) , (13, 6) , (13, 7) , (13, 8) , (13, 9) , (13, 10) , (13, 11) , (14, 3) , (14, 4) , (14, 5) , (14, 6) , (14, 7) , (14, 8) , (14, 9) , (14, 10) , (14, 11) ,
                ]
        }, 
        # Scenario 2: The U-Obstacles Scenario
        {
        'dim': (20, 10),
        'type_knowledge': True,
        'parameter_knowledge': True,
        'vision_block': True,
        'agents': [
            Agent(index='0',atype=method,position=(0,5),direction=0,radius=0.2,angle=0.3,level=1.0),
        ],
        'adhoc_agent_index': '0',
        'impostor_index': None,
        'tasks': [
            Task(index='0',position=(4,5),level=1.0),
            Task(index='1',position=(8,4),level=1.0),
            Task(index='2',position=(12,5),level=1.0),
            Task(index='3',position=(19,0),level=1.0),
            Task(index='4',position=(19,9),level=1.0),
        ],
        'obstacles': [
            (3, 2) , (3, 7) , (4, 2) , (4, 7) , (5, 2) , (5, 3) , (5, 4) , (5, 5) , (5, 6) , (5, 7) , (11, 2) , (11, 3) , (11, 4) , (11, 5) , (11, 6) , (11, 7) , (12, 2) , (12, 7) , (13, 2) , (13, 7) ,
        ],
        },
        # Scenario 3: The Warehouse (Square PO Foraging Scenario)
        {
        'dim': (20,20),
        'type_knowledge': True,
        'parameter_knowledge': True,
        'vision_block': True,
        'agents' : [
            Agent(index='A',atype=method,position=(10,1),
                    direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0), 
                ],
        'adhoc_agent_index' : 'A',
        'impostor_index': None,
        'tasks' : [
            Task(index='0',position=(11,0),level=1.0),
            Task(index='1',position=(10,18),level=1.0),
            Task(index='2',position=(9,10),level=1.0),
            Task(index='A0',position=(1,1),level=1.0),
                Task(index='A1',position=(3,3),level=1.0),
                Task(index='A2',position=(1,3),level=1.0),
                Task(index='A3',position=(3,1),level=1.0),
            Task(index='B0',position=(18,1),level=1.0),
                Task(index='B1',position=(18,3),level=1.0),
                Task(index='B2',position=(16,3),level=1.0),
                Task(index='B3',position=(16,1),level=1.0),
            Task(index='C0',position=(18,18),level=1.0),
                Task(index='C1',position=(18,16),level=1.0),
                Task(index='C2',position=(16,16),level=1.0),
                Task(index='C3',position=(16,18),level=1.0),
            Task(index='D0',position=(1,18),level=1.0),
                Task(index='D1',position=(3,18),level=1.0),
                Task(index='D2',position=(3,16),level=1.0),
                Task(index='D3',position=(1,16),level=1.0),
                ],
        'obstacles':[]
        },
        # Scenario 4: The Office (Square with Obstacles PO Foraging Scenario)
        {
        'dim': (15, 10),
        'type_knowledge': True,
        'parameter_knowledge': True,
        'vision_block': True,
        'agents': [
                 Agent(index='0',atype=method,position=(0,9),direction=3*np.pi/2,radius=0.2,angle=0.3,level=1.0),
        ],
        'adhoc_agent_index': '0',
        'impostor_index': None,
        'tasks': [
                 Task(index='0',position=(0,0),level=1.0),
                 Task(index='1',position=(3,3),level=1.0),
                 Task(index='2',position=(5,7),level=1.0),
                 Task(index='3',position=(11,5),level=1.0),
                 Task(index='4',position=(14,0),level=1.0),
        ],
        'obstacles': [
                (2, 2) , (2, 3) , (2, 4) , (2, 5) , (2, 6) , (2, 7) , (2, 8) , (2, 9) , (3, 2) , (3, 6) , (4, 2) , (4, 6) , (5, 2) , (5, 6) , (6, 2) , (6, 3) , (6, 4) , (6, 6) , (6, 7) , (6, 8) , (9, 2) , (9, 3) , (9, 4) , (9, 5) , (9, 6) , (9, 7) , (9, 8) , (10, 2) , (10, 8) , (11, 2) , (11, 4) , (11, 6) , (11, 8) , (12, 2) , (12, 4) , (12, 5) , (12, 6) , (12, 8) , (13, 2) , (13, 8) ,
        ],
        },
        {
        # Scenario 5: FO Adversarial - hand designed scenario: hard to accomplish
        # but good to evaluate the adversary behaviour
        'dim': (5,5),
        'type_knowledge': False,
        'template_types':['l6','adversary'],
        'parameter_knowledge': False,
        'vision_block': True,
        'agents' : [
            Agent(index='A',atype=method,position=(0,1),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
            Agent(index='1',atype='l1',position=(1,0),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=1.), 
            Agent(index='X',atype='mcts_min',position=(3,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
                ],
        'adhoc_agent_index' : 'A',
        'impostor_index': 'X',
        'tasks' : [
            Task(index='0',position=(4,4),level=1.),
                ],
        'obstacles':[]
        },
        {
        # Scenario 6: FO Adversarial - warehouse benchmark scenario with 
        # 1 adhoc agent 1 teammate 1 adversary and 5 tasks
        'dim': (9,9),
        'type_knowledge': False,
        'template_types':['l6','adversary'],
        'parameter_knowledge': False,
        'vision_block': True,
        'agents' : [
            Agent(index='A',atype=method,position=(0,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
            Agent(index='1',atype='l3',position=(0,8),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=1.), 
            Agent(index='X',atype='mcts_min',position=(8,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
                ],
        'adhoc_agent_index' : 'A',
        'impostor_index': 'X',
        'tasks' : [
            Task(index='0',position=(3,3),level=0.5),
            Task(index='1',position=(7,7),level=0.5),
            Task(index='2',position=(7,3),level=0.5),
            Task(index='3',position=(3,7),level=0.5),
            Task(index='4',position=(5,5),level=0.5),
                ],
        'obstacles':[]
        },
        {
        # Scenario 7: FO Adversarial  - warehouse benchmark scenario with 
        # 1 adhoc agent 3 teammate 1 adversary and 5 tasks
        'dim': (9,9),
        'type_knowledge': False,
        'template_types':['l4','l5','l6','adversary'],
        'parameter_knowledge': False,
        'vision_block': True,
        'agents' : [
            Agent(index='A',atype=method,position=(0,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=1.0), 
            Agent(index='1',atype='l1',position=(0,4),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=1.0),  
            Agent(index='2',atype='l2',position=(0,8),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=1.0),  
            Agent(index='3',atype='l3',position=(4,0),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=1.0), 
            Agent(index='X',atype='mcts_min',position=(8,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=1.0),  
                ],
        'adhoc_agent_index' : 'A',
        'impostor_index': 'X',
        'tasks' : [
            Task(index='0',position=(3,3),level=0.5),
            Task(index='1',position=(7,7),level=0.5),
            Task(index='2',position=(7,3),level=0.5),
            Task(index='3',position=(3,7),level=0.5),
            Task(index='4',position=(5,5),level=0.5),
                ],
        'obstacles':[]
        },
        {
        # Scenario 8: FO Adversarial - bigger benchmark scenario for adversaries
        # 1 adhoc agent 3 teammate 1 adversary and 10 tasks
        'dim': (9,9),
        'type_knowledge': False,
        'template_types':['l4','l5','l6','adversary'],
        'parameter_knowledge': False,
        'vision_block': True,
        'agents' : [
            Agent(index='A',atype=method,position=(0,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=0.1), 
            Agent(index='1',atype='l1',position=(0,4),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=0.3), 
            Agent(index='2',atype='l2',position=(0,8),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=0.4), 
            Agent(index='3',atype='l3',position=(4,0),
                    direction=0*np.pi/2,radius=1.,angle=1.,level=0.5), 
            Agent(index='X',atype='mcts_min',position=(8,0),
                    direction=1*np.pi/2,radius=1.,angle=1.,level=0.6), 
                ],
        'adhoc_agent_index' : 'A',
        'impostor_index': 'X',
        'tasks' : [
            Task(index='0',position=(3,3),level=0.1),
            Task(index='1',position=(7,7),level=0.2),
            Task(index='2',position=(7,3),level=0.4),
            Task(index='3',position=(3,7),level=0.6),
            Task(index='4',position=(5,5),level=0.8),
                ],
        'obstacles':[]
        },
    ]

    if scenario_id >= len(default_scenarios_components):
        print('There is no default scenario with id '+str(scenario_id)+
                ' for the LevelForaging problem. Setting scenario_id to 0.')
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

    def __init__(self, index, atype, position, direction,
                                radius, angle, level, estimation_method=None):
        super(Agent, self).__init__(index, atype)

        # agent parameters
        self.position = position
        self.direction = direction
        self.radius = radius
        self.angle = angle
        self.level = level

        self.memory = {'agents':{},'tasks':{},'obstacles':[],'states':set()}
        self.memory_scenario = None

        self.smart_parameters['last_completed_task'] = None
        self.smart_parameters['choose_task_state'] = None
        self.smart_parameters['ntasks'] = None
        if estimation_method is not None:
            self.smart_parameters['estimation_method'] = estimation_method

    def copy(self):
        # 1. Initialising the agent
        x,y = self.position
        copy_agent = Agent(self.index, self.type, (x,y), \
                           self.direction, self.radius, self.angle, self.level)

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters

        return copy_agent

    def set_parameters(self, parameters):
        self.radius = parameters[0]
        self.angle = parameters[1]
        self.level = parameters[2]

    def get_parameters(self):
        return np.array([self.radius,self.angle,self.level])

    def show(self):
        print(self.index, self.type, ':', self.position, self.direction,
         self.radius, self.angle, self.level)
        
    def show_memory(self):
        #print(adhoc_agent.memory)
        for y in reversed(range(len(self.memory_scenario[0]))):
            for x in range(len(self.memory_scenario)):
                if (x,y) == self.position:
                    print("%3s" % ('A'),end='')
                else:
                    print(self.memory_scenario[x][y],end='')
            print()


class Task():
    """Task : These are parts of the 'components' of the environemnt.
    """
    def __init__(self, index, position, level):
        # task parameters
        self.index = index
        self.position = position
        self.level = level

        # task simulation parameters
        self.completed = False
        self.trying = []

    def copy(self):
        # 1. Initialising the copy task
        copy_task = Task(self.index, self.position, self.level)

        # 2. Copying the parameters
        copy_task.completed = self.completed
        copy_task.trying = [a for a in self.trying]

        return copy_task


"""
    Customising the Level-Foraging Env
"""
def end_condition(state):
    return sum([not t.completed for t in state.components['tasks']]) == 0


def who_see(env, position):
    who = []
    for a in env.components['agents']:
        # setting/retriving parameters
        if a.direction is not None:
            direction = a.direction
        else:
            direction = env.sample_direction()

        if a.radius is not None:
            radius = np.sqrt(env.shape[0] ** 2 + env.shape[1] ** 2) * a.radius
        else:
            radius = env.sample_radius()

        if a.radius is not None:
            angle = 2 * np.pi * a.angle
        else:
            angle = env.sample_angle()

        # checking visibility
        if (a.angle == 1. and a.radius == 1.) \
         or is_visible(position, a.position, direction,
         radius, angle, env.components['obstacles'], env.vision_block):
            who.append(a)
    return who


def there_is_task(env, position, direction):
    # 1. Calculating the task position
    if direction == np.pi / 2:
        pos = (position[0], position[1] + 1)
    elif direction == 3 * np.pi / 2:
        pos = (position[0], position[1] - 1)
    elif direction == np.pi:
        pos = (position[0] - 1, position[1])
    elif direction == 0:
        pos = (position[0] + 1, position[1])
    else:
        pos = None

    # 2. If there is a task, return it, else None
    for task in env.components['tasks']:
        if not task.completed and\
         pos == task.position:
            return task
    return None


def new_position_given_action(pos, action, shape):
    # 1. Calculating the new position
    if action == 2:  # N
        new_pos = (pos[0], pos[1] + 1) if pos[1] + 1 < shape[1] \
            else (pos[0], pos[1])
    elif action == 3:  # S
        new_pos = (pos[0], pos[1] - 1) if pos[1] - 1 >= 0 \
            else (pos[0], pos[1])
    elif action == 1:  # W
        new_pos = (pos[0] - 1, pos[1]) if pos[0] - 1 >= 0 \
            else (pos[0], pos[1])
    elif action == 0:  # E
        new_pos = (pos[0] + 1, pos[1]) if pos[0] + 1 < shape[0] \
            else (pos[0], pos[1])
    else:
        new_pos = (pos[0], pos[1])

    return new_pos

# This method returns True if a position is visible, else False
def is_visible(obj, viewer, direction, radius, angle, obstacles, vision_block):
    # 1. Checking visibility
    # - centralising viewer
    c_viewer = [viewer[0]+0.5,viewer[1]+0.5]
    
    # - checking the object edges
    if euclidean_distance([obj[0]+0,obj[1]+0], c_viewer) <= radius or \
       euclidean_distance([obj[0]+1,obj[1]+0], c_viewer) <= radius or \
       euclidean_distance([obj[0]+0,obj[1]+1], c_viewer) <= radius or \
       euclidean_distance([obj[0]+1,obj[1]+1], c_viewer) <= radius:
        if -angle / 2 <= angle_of_gradient([obj[0]+0,obj[1]+0], c_viewer, direction) <= angle / 2 \
         or -angle / 2 <= angle_of_gradient([obj[0]+1,obj[1]+0], c_viewer, direction) <= angle / 2 \
         or -angle / 2 <= angle_of_gradient([obj[0]+0,obj[1]+1], c_viewer, direction) <= angle / 2 \
         or -angle / 2 <= angle_of_gradient([obj[0]+1,obj[1]+1], c_viewer, direction) <= angle / 2:
            if vision_block and obstacle_between(obstacles,obj,viewer):
                return False
            return True
    return False
    
def obstacle_between(obstacles,obj,viewer):
    start_point = [viewer[0]+0.5,viewer[1]+0.5]

    c_viewer = [viewer[0]+0.5,viewer[1]+0.5]
    c_object = [obj[0]+0.5,obj[1]+0.5]

    dist = euclidean_distance(c_object, c_viewer)
    for i in range(1,101):
        current_point = [
            start_point[0]- ((i/100)*(c_viewer[0]-c_object[0])),
            start_point[1]- ((i/100)*(c_viewer[1]-c_object[1]))
        ]
        if (int(current_point[0]),int(current_point[1])) != (obj[0], obj[1]) and \
         (int(current_point[0]),int(current_point[1])) in obstacles:
            return True
    return False

def do_action(env):
    # 1. Position and direction
    # a. defining the agents new position and direction
    positions, directions = {}, {}
    info = {'action reward': 0, 'just_finished_tasks': []}
    for agent in env.components['agents']:
        if agent.next_action != 4 and agent.next_action is not None:
            positions[agent.index] = new_position_given_action(
                                agent.position, agent.next_action, env.shape)
            directions[agent.index] = env.action2direction[agent.next_action]

        else:
            positions[agent.index] = agent.position
            directions[agent.index] = agent.direction

    # b. analysing position conflicts
    # agent x task and obstacle
    for i in range(len(env.components['agents'])):
        for task in env.components['tasks']:
            if positions[env.components['agents'][i].index] == \
            task.position and not task.completed:
                positions[env.components['agents'][i].index] = \
                        env.components['agents'][i].position
        if positions[env.components['agents'][i].index] in env.components['obstacles']:
                positions[env.components['agents'][i].index] = \
                        env.components['agents'][i].position
                        
    # between agents
    # -- ocupied space by another agent
    for i in range(len(env.components['agents'])):
        for j in range(len(env.components['agents'])):
            if i != j:
                index = env.components['agents'][i].index
                if env.components['agents'][j].next_action == 4 and\
                positions[index] == env.components['agents'][j].position:
                    positions[index] = env.components['agents'][i].position

    # -- two agents towards a empty space
    for i in range(len(env.components['agents'])):
        for j in range(len(env.components['agents'])):
            if i != j:
                first_index = env.components['agents'][i].index
                second_index = env.components['agents'][j].index
                if positions[first_index] == positions[second_index]:
                    if rd.uniform(0, 1) < 0.5:
                        positions[first_index] = env.components['agents'][i].position
                    else:
                        positions[second_index] = env.components['agents'][j].position

    # c. updating the simulation agents position
    for i in range(len(env.components['agents'])):
        env.components['agents'][i].position = \
         positions[env.components['agents'][i].index]
        env.components['agents'][i].direction = \
         directions[env.components['agents'][i].index]

    # 2. Tasks 
    # a. verifying the tasks to be completed
    for agent in env.components['agents']:
        if agent.next_action == 4:
            task = there_is_task(env, agent.position, agent.direction)
            if task is not None:
                if agent.level is not None:
                    task.trying.append(agent.level)
                else:
                    task.trying.append(rd.uniform(0, 1))
        else:
            agent.smart_parameters['last_completed_task'] = None

    # b. calculating the reward
    for task in env.components['tasks']:
        if not task.completed:
            if sum(task.trying) >= task.level:
                info['action reward'] += 1
                task.completed = True
                if task not in info['just_finished_tasks']:
                    info['just_finished_tasks'].append(task)

                for ag in who_see(env, task.position):
                    if (ag.target == task.position or ag.target == task.index):
                        if not env.simulation:
                            ag.smart_parameters['last_completed_task'] = task
                            ag.smart_parameters['choose_task_state']=env.copy()
                        ag.target = None

    # c. resetting the task trying
    for task in env.components['tasks']:
        task.trying = []

    if not env.simulation:
        for ag in env.components['agents']:
            ag.smart_parameters['ntasks'] -= len(info['just_finished_tasks'])

    adhoc_agent = env.get_adhoc_agent()
    env.state = (adhoc_agent.position[0],adhoc_agent.position[1])
    return env, info


def levelforaging_transition(action, real_env):
    # agent planning
    adhoc_agent = real_env.get_adhoc_agent()
    adhoc_agent.next_action = action
    adhoc_agent.target = adhoc_agent.target
    next_state, info = real_env, {}

    # Adversarial transition
    if real_env.is_adversarial():
        # in simulation
        if real_env.simulation:
            # teammates reasoning
            if real_env.reasoning_turn == 'adhoc':
                for agent in real_env.components['agents']:
                    if agent.index != adhoc_agent.index and\
                    agent.index != real_env.components['impostor_index']:
                        # changing the perspective
                        copied_env = real_env.copy()
                        copied_env.components['adhoc_agent_index'] = agent.index

                        # generating the observable scenario
                        obsavable_env = copied_env.observation_space(copied_env)

                        # planning the action from agent i perspective
                        if agent.type is not None and agent.type in real_env.get_available_types():
                            planning_method = real_env.import_method(agent.type)
                            agent.next_action, agent.target = \
                                planning_method(obsavable_env, agent)
                        else:
                            # some template
                            agent.type = real_env.sample_available_types()
                            planning_method = real_env.import_method(agent.type)
                            agent.next_action, agent.target = \
                                planning_method(obsavable_env, agent)
                            # random policy
                            #agent.next_action, agent.target = \
                            #    real_env.action_space.sample(), None

                # changing perspectives from adhoc to adversarial
                real_env.reasoning_turn = 'adversarial'
                adhoc_index = real_env.components['adhoc_agent_index']
                impostor_index = real_env.components['impostor_index']
                real_env.components['adhoc_agent_index'] = impostor_index
                real_env.components['impostor_index'] = adhoc_index

            elif real_env.reasoning_turn == 'adversarial':
                # environment step
                next_state, info = do_action(real_env)

                # updating memory
                if not real_env.simulation:
                    next_state.update_memory(adhoc_agent)

                # changing perspectives from adversarial to adhoc
                real_env.reasoning_turn = 'adhoc'
                adhoc_index = real_env.components['impostor_index']
                impostor_index = real_env.components['adhoc_agent_index']
                real_env.components['adhoc_agent_index'] = adhoc_index
                real_env.components['impostor_index'] = impostor_index

            # retuning the results
            return next_state, info

        # in real world
        else:
            for agent in real_env.components['agents']:
                if agent.index == real_env.components['impostor_index']:
                    # changing the perspective
                    copied_env = real_env.copy()
                    copied_env.components['adhoc_agent_index'] = agent.index

                    # generating the observable scenario
                    copied_env.components['type_knowledge'] = False
                    copied_env.components['parameter_knowledge'] = True
                    copied_env.components['vision_block'] = False
                    copied_env.components['impostor_index'] = None
                    obsavable_env = copied_env.observation_space(copied_env)

                    # planning the action from agent i perspective
                    if agent.type is not None:
                        planning_method = real_env.import_method(agent.type)
                        agent.next_action, agent.target = \
                            planning_method(obsavable_env, agent)
                            
                    else:
                        agent.next_action, agent.target = \
                            real_env.action_space.sample(), None
                elif agent.index != adhoc_agent.index:
                    # changing the perspective
                    copied_env = real_env.copy()
                    copied_env.components['adhoc_agent_index'] = agent.index

                    # generating the observable scenario
                    obsavable_env = copied_env.observation_space(copied_env)

                    # planning the action from agent i perspective
                    if agent.type is not None:
                        planning_method = real_env.import_method(agent.type)
                        agent.next_action, agent.target = \
                            planning_method(obsavable_env, agent)
                    else:
                        agent.next_action, agent.target = \
                            real_env.action_space.sample(), None

            # environment step
            next_state, info = do_action(real_env)

            # updating memory
            if not real_env.simulation:
                next_state.update_memory(adhoc_agent)

            # retuning the results
            return next_state, info
    # Foraging transition
    else:
        for agent in real_env.components['agents']:
            # Ad-hoc agent
            if agent.index != adhoc_agent.index:
                # changing the perspective
                copied_env = real_env.copy()
                copied_env.components['adhoc_agent_index'] = agent.index

                # generating the observable scenario
                obsavable_env = copied_env.observation_space(copied_env)

                # planning the action from agent i perspective
                if agent.type is not None and agent.type in real_env.get_available_types():
                    planning_method = real_env.import_method(agent.type)
                    agent.next_action, agent.target = \
                        planning_method(obsavable_env, agent)
                else:
                    # some template
                    agent.type = real_env.sample_available_types()
                    planning_method = real_env.import_method(agent.type)
                    agent.next_action, agent.target = \
                        planning_method(obsavable_env, agent)
                    # random policy
                    #agent.next_action, agent.target = \
                    #    real_env.action_space.sample(), None

        # environment step
        next_state, info = do_action(real_env)

        # updating memory
        if not real_env.simulation:
            next_state.update_memory(adhoc_agent)

        # retuning the results
        return next_state, info
    raise NotImplementedError

# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return 0


# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    agent = copied_env.get_adhoc_agent()
    if copied_env.simulation:
        copied_env.state = (agent.position[0],agent.position[1])
        return copied_env
        
    if agent is not None:
        # 1. Collecting the agent parameter
        shape = copied_env.shape
        radius = np.sqrt(shape[0] ** 2 + shape[1] ** 2) * agent.radius
        angle = 2 * np.pi * agent.angle
        obstacles = copied_env.components['obstacles']

        if agent.angle != 1. or agent.radius != 1.:
            # 2. Removing the invisible agents from environment
            invisible_agents = []
            for i in range(len(copied_env.components['agents'])):
                if copied_env.components['agents'][i].index != agent.index and    \
                not is_visible(copied_env.components['agents'][i].position,    \
                agent.position, agent.direction, radius, angle, obstacles, \
                copied_env.vision_block) and  copied_env.components['agents'].\
                index not in agent.memory['agents']:
                    invisible_agents.append(i)

            for index in sorted(invisible_agents, reverse=True):
                copied_env.components['agents'].pop(index)

            # 3. Removing the invisible tasks from environment
            invisible_tasks = []
            for i in range(len(copied_env.components['tasks'])):
                if copied_env.components['tasks'][i].completed:
                    invisible_tasks.append(i)
                elif copied_env.components['tasks'][i].index not in agent.memory['tasks'] and \
                not is_visible( copied_env.components['tasks'][i].position,  \
                 agent.position, agent.direction, radius, angle, obstacles, \
                 copied_env.vision_block):
                    invisible_tasks.append(i)

            for index in sorted(invisible_tasks, reverse=True):
                copied_env.components['tasks'].pop(index)
            
            # 4. Removing the invisible obstacles from environment
            invisible_obst = []
            if not PRIOR_OBSTACLES_KNOWLEDGE:
                for i in range(len(copied_env.components['obstacles'])):
                    x = copied_env.components['obstacles'][i][0]
                    y = copied_env.components['obstacles'][i][1]
                    if not is_visible( copied_env.components['obstacles'][i],  \
                     agent.position, agent.direction, radius, angle, [], \
                     copied_env) and (x,y) not in agent.memory['obstacles']:
                        invisible_obst.append(i)

            for index in sorted(invisible_obst, reverse=True):
                copied_env.components['obstacles'].pop(index)

        # 5. Cleaning other agents' information
        for i in range(len(copied_env.components['agents'])):
            if copied_env.components['agents'][i] != agent:
                if not copied_env.parameter_knowledge:
                    copied_env.components['agents'][i].radius = None
                    copied_env.components['agents'][i].angle = None
                    copied_env.components['agents'][i].level = None
                    copied_env.components['agents'][i].target = None
                if not copied_env.type_knowledge:
                    copied_env.components['agents'][i].type = None

        copied_env.state = (agent.position[0],agent.position[1])
        return copied_env
    else:
        raise IOError(agent, 'is an invalid agent.')


"""
    Level-Foraging Environments 
"""


class LevelForagingEnv(AdhocReasoningEnv):

    actions = [0,1,2,3,4]

    action_dict = {
        0: 'East',
        1: 'West',
        2: 'North',
        3: 'South',
        4: 'Load'
    }

    action2direction = {
        0: 0,  # East
        1: np.pi,  # West
        2: np.pi / 2,  # North
        3: 3 * np.pi / 2}  # South

    directions = [
        0,  # East
        np.pi,  # West
        np.pi / 2,  # North
        3 * np.pi / 2  # South
    ]

    agents_color = {
        'mcts': 'red',
        'pomcp': 'yellow',
        'ibpomcp':'blue',
        'rhopomcp':'cyan',
        'dqn':'magenta',
        'l1': 'darkred',
        'l2': 'darkgreen',
        'l3': 'darkcyan',
        'adversary':'darkblue'
    }

    def __init__(self, shape, components, display=False, \
     type_knowledge=True, parameter_knowledge=True, vision_block=True):
        ###
        # Env Settings
        ###
        self.type_knowledge = type_knowledge
        self.parameter_knowledge = parameter_knowledge
        self.vision_block = vision_block

        self.shape = shape
        self.reasoning_turn = 'adhoc'

        state_set = StateSet(spaces.Tuple(\
            (spaces.Discrete(shape[0]),spaces.Discrete(shape[0]))), end_condition)
        transition_function = levelforaging_transition
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(LevelForagingEnv, self).__init__(state_set, \
            transition_function, action_space, reward_function, \
            observation_space, components)
        self.name = None
        
        self.max_reward = len(self.components['tasks']) \
            if len(self.components['tasks']) < len(self.components['agents']) \
            else len(self.components['agents'])

        # Checking components integrity
        if 'agents' not in self.components:
            raise ValueError("There is no agent in the environment")
        if 'tasks' not in self.components:
            raise ValueError("There is no task in the environment")
        if 'obstacles' not in self.components:
            self.components['obstacles'] =  []

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
    
    def is_adversarial(self):
        if 'impostor_index' in self.components:
            if self.components['impostor_index'] is not None:
                return True
        return False

    def adversarial_policy(self):
        return np.random.choice([0,1,2,3])
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.episode = 0
        self.sample_index = len(self.components['tasks'])*10

        if self.state_set.initial_state is not None and self.state_set.initial_components is not None:
            self.state = (self.state_set.initial_state[0],self.state_set.initial_state[1])
            self.components = self.copy_components(self.state_set.initial_components)

            agent = self.get_adhoc_agent()
            self.update_memory(agent)

            if self.display:
                self.reset_renderer()

            # Updating agent knowledge about tasks
            for i in range(len(self.components['agents'])):
                self.components['agents'][i].smart_parameters['ntasks'] = len(self.components['tasks'])

            return self.observation_space(self.copy())

        else:
            raise ValueError("the initial state from the state set is None.")

    def respawn_tasks(self):
        empty_positions = self.get_empty_positions()
        rd.shuffle(empty_positions)
        for i in range(len(self.components['tasks'])):
            self.components['tasks'][i].completed = False
            new_position = empty_positions.pop()
            self.components['tasks'][i].position = new_position
        
        for i in range(len(self.components['agents'])):
            self.components['agents'][i].smart_parameters['ntasks'] = \
                len(self.components['tasks'])
        return 

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
        copied_env = LevelForagingEnv(self.shape, components, self.display,\
            self.type_knowledge, self.parameter_knowledge, self.vision_block)
        copied_env.simulation = self.simulation
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.sample_index = self.sample_index
        copied_env.reasoning_turn = self.reasoning_turn
        copied_env.name = self.name

        # Setting the initial state
        copied_env.state = (self.state[0],self.state[1])
        copied_env.state_set.initial_state = (self.state_set.initial_state[0],self.state_set.initial_state[1])
        return copied_env

    def get_actions_list(self):
        return [action for action in self.action_dict]

    def get_feature(self):
        return self.state
    
    def get_max_reward(self):
        return self.max_reward
    
    def get_available_types(self, mode='reactive'):
        if 'template_types' in self.components:
            return self.components['template_types'] 
        
        if mode == 'reactive':
            return ['l1','l2','l3','l4','l5','l6']
        elif mode == 'adversarial':
            return ['adversary']
        elif mode == 'all':
            return ['l1','l2','l3','l4','l5','l6','adversary']
        else:
            raise NotImplemented
    
    def sample_available_types(self, mode='reactive'):
        return np.random.choice(['l1','l2','l3','l4','l5','l6'])

    def get_adhoc_agent(self):
        for agent in self.components['agents']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        raise IndexError("Ad-hoc Index is not in Agents Index Set.")
 
    # This method returns the visible tasks positions
    def get_visible_components(self, agent):
        # 1. Defining the agent vision parameters
        direction = agent.direction
        radius = np.sqrt(self.shape[0] ** 2 + self.shape[1] ** 2) * agent.radius
        angle = 2 * np.pi * agent.angle

        obstacles_ = self.components['obstacles']
        agents, tasks = [], []

        # 2. Looking for agents
        for ag in self.components['agents']:
            x, y = ag.position
            if ag.position != agent.position:
                if (agent.angle == 1. and agent.radius == 1.): 
                    agents.append([ag.index, x, y])
                elif is_visible([x, y], agent.position, direction, radius, \
                 angle, obstacles_, self.vision_block):
                    agents.append([ag.index, x, y])

        # 3. Looking for tasks
        for task in self.components['tasks']:
            x, y = task.position
            if not task.completed:
                if (agent.angle == 1. and agent.radius == 1.):
                    tasks.append([task.index, x, y])
                elif is_visible([x, y], agent.position,
                 direction,radius,angle, obstacles_, self.vision_block):
                    tasks.append([task.index, x, y])
        
        # 4. Looking for obstacles
        obstacles = []
        for obs in self.components['obstacles']:
            x, y = obs
            if PRIOR_OBSTACLES_KNOWLEDGE or (agent.angle == 1. and agent.radius == 1.):
                obstacles.append([x, y])
            elif is_visible([x, y], agent.position, direction, radius, angle,\
             obstacles_, self.vision_block):
                obstacles.append([x, y])

        # 5. Returning the result
        return {'agents':agents, 'tasks':tasks, 'obstacles':obstacles}

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]
        
    def state_is_equal(self, state):
        return state.state[0] == self.state[0] and state.state[1] == self.state[1]

    def get_state_str_representation(self):
        return '('+str(self.state[0])+','+str(self.state[1])+')'

    def get_rlmodel_state(self):
        rl_state = np.zeros(self.shape)
        for ag in self.components['agents']:
            x,y = ag.position
            rl_state[x,y] = 1
        for tk in self.components['tasks']:
            x,y = tk.position
            rl_state[x,y] = 2
        return rl_state
    
    def get_rlmodel_input_shape(self):
        rl_state = self.get_rlmodel_state()
        return rl_state.shape[0]*rl_state.shape[1]

    def hash_state(self):
        return hash((self.state[0],self.state[1]))

    def hash_observation(self):
        obs = self.get_observation()
        return hash(str(obs))

    def get_empty_positions(self):
        empty_spaces = []
        dim_w, dim_h = self.shape

        agents_spaces = [agn.position for agn in self.components['agents']]
        tasks_spaces = [tsk.position if not tsk.completed else None for tsk in self.components['tasks']]
        obstacles_spaces = [obs for obs in self.components['obstacles']]

        for x in range(dim_w):
            for y in range(dim_h):
                if (x,y) not in agents_spaces and \
                  (x,y) not in tasks_spaces and \
                  (x,y) not in obstacles_spaces : 
                    empty_spaces.append((x, y))
        return empty_spaces

    def get_unknown_positions(self, agent):
        empty_spaces = []
        dim_w, dim_h = self.shape

        for x in range(dim_w):
            for y in range(dim_h):
                if agent.memory_scenario[x][y] == '-?-':
                    empty_spaces.append((x, y))
        return empty_spaces

    def get_observation(self):
        return  self.get_visible_components(self.get_adhoc_agent())

    def observation_is_equal(self, obs):
        cur_visibility = self.get_observation()
        
        if PRIOR_OBSTACLES_KNOWLEDGE:
          return (cur_visibility['agents'] == obs['agents']) and \
                (cur_visibility['tasks'] == obs['tasks'])  

        return (cur_visibility['agents'] == obs['agents']) and \
                (cur_visibility['tasks'] == obs['tasks']) and \
                (cur_visibility['obstacles'] == obs['obstacles']) 

    def get_agent_by_index(self,agent_index):
        for ag in self.components['agents']: 
            if ag.index == agent_index:
                return ag
        return None

    def update_memory(self,agent):
        cur_visibility = self.get_visible_components(agent)

        # 1. Updating memory about agents
        for ag in cur_visibility['agents']:
            agent.memory['agents'][ag[0]] = (ag[1],ag[2])

        # 2. Updating memory about tasks
        for tk in cur_visibility['tasks']:
            agent.memory['tasks'][tk[0]] = (tk[1],tk[2])

        # 3. Updating memory about obstacles
        if not PRIOR_OBSTACLES_KNOWLEDGE:
            for obst in cur_visibility['obstacles']:
                if (obst[0],obst[1]) not in agent.memory['obstacles']:
                    agent.memory['obstacles'].append((obst[0],obst[1]))
        elif len(agent.memory['obstacles']) == 0:
            for obst in self.components['obstacles']:
                agent.memory['obstacles'].append((obst[0],obst[1]))
        
        # 4. Updating memory about visible states
        dim_w, dim_h = self.shape
        direction = agent.direction
        radius = np.sqrt(dim_w ** 2 + dim_h ** 2) * agent.radius
        angle = 2 * np.pi * agent.angle

        for x in range(dim_w):
            for y in range(dim_h):
                # maximum/full visibility
                if agent.angle == 1. and agent.radius == 1.:
                    agent.memory['states'].add((x, y))
                # partial visibility
                elif (x,y) not in agent.memory['obstacles'] and \
                is_visible((x, y), agent.position, direction, \
                 radius, angle, obstacles=self.components['obstacles'], \
                 vision_block=self.vision_block):
                    agent.memory['states'].add((x, y))
    
        # 5. Updating memory scenario
        agent.memory_scenario = [['%3s' % ('-?-') \
            for y in range(self.shape[1])] for x in range(self.shape[0])]
        for st in agent.memory['states']:
            x, y = st[0], st[1]
            agent.memory_scenario[x][y] = '%3s' % ('.')
        for ag in agent.memory['agents']:
            x, y = agent.memory['agents'][ag][0], agent.memory['agents'][ag][1]
            agent.memory_scenario[x][y] = '%3s' % ('A'+str(ag))
        for tk in agent.memory['tasks']:
            x, y = agent.memory['tasks'][tk][0], agent.memory['tasks'][tk][1]
            agent.memory_scenario[x][y] = '%3s' % ('T'+str(tk))
        for obs in agent.memory['obstacles']:
            x, y = obs[0], obs[1]
            agent.memory_scenario[x][y] = '%3s' % ('|||')

    def sample_direction(self):
        return rd.choice(self.directions)
    
    def sample_radius(self):
        mod = np.sqrt(self.shape[0] ** 2 + self.shape[1] ** 2) 
        return mod * rd.uniform(0, 1)

    def sample_angle(self):
        mod = 2 * np.pi
        return mod * rd.uniform(0, 1)

    def sample_state(self, agent):
        # 1. Defining the base simulation
        u_env = self.copy()

        # - if the problem is full observable, there is no changes to do
        if agent.radius == 1. and agent.angle == 1.:
            return u_env

        # 2. Setting possibilities
        empty_position = self.get_unknown_positions(agent)
        for _ in range(agent.smart_parameters['ntasks']):
            if len(empty_position) != 0:
                pos = rd.choice(empty_position)
                u_env.components['tasks'].append(\
                    Task('S'+str(self.sample_index),pos,rd.uniform(0,1)))
                empty_position.remove(pos)
                self.sample_index +=1
        # 3. Returning the modified/sampled environment
        return u_env

    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def sample_random_action(self):
        return np.random.choice(self.actions)

    def get_target(self, agent_index, new_type=None, new_parameter=None):
        # changing the perspective
        copied_env = self.copy()
        copied_env.components['adhoc_agent_index'] = agent_index

        # generating the observable scenario
        adhoc_agent = copied_env.get_adhoc_agent()
        adhoc_agent.type = new_type
        adhoc_agent.set_parameters(new_parameter)
        adhoc_agent.target = None

        obsavable_env = copied_env.observation_space(copied_env)

        obsavable_env.components['adhoc_agent_index'] = agent_index
        adhoc_agent = obsavable_env.get_adhoc_agent()
        adhoc_agent.type = new_type
        adhoc_agent.set_parameters(new_parameter)
        adhoc_agent.target = None

        # planning the action from agent i perspective
        planning_method = self.import_method(new_type)
        _, target = \
            planning_method(obsavable_env, adhoc_agent)

        # retuning the results
        for task in self.components['tasks']:
            if task.position == target:
                return task
        return None

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

        dim = self.shape
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

        # background
        self.surf = pygame.Surface(self.screen_size)
        self.surf.fill(self.colors['white'])
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # grid
        grid_width, grid_height = (dim[0]*700/max_dim,dim[1]*700/max_dim)
        self.grid_surf = pygame.Surface((grid_width, grid_height))
        self.grid_surf.fill(self.colors['white'])

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
        
            
        if 'obstacles' in self.components:
            for obs in self.components['obstacles']:
                x = int(obs[0]*(grid_width/dim[0]))
                y = int(obs[1]*(grid_height/dim[1]))
                gfxdraw.box(self.grid_surf,
                    pygame.Rect(x,y,(grid_width/dim[0]),(grid_height/dim[1])),
                    self.colors['black'])

        # agents
        self.components_surf = pygame.Surface((grid_width, grid_height))
        self.components_surf = self.components_surf.convert_alpha()
        self.components_surf.fill((self.colors['white'][0],self.colors['white'][1],self.colors['white'][2],0))
        def my_rotation(ox,oy,px,py,angle):
            angle = angle
            qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
            qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
            return int(qx),int(qy)

        for agent in self.components['agents']:
            direction = agent.direction - np.pi/2
            ox = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            oy = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
            #arms
            w = int(0.85*(grid_width/dim[0]))
            h = int(0.25*(grid_height/dim[1]))
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))

            arms = pygame.Surface((w , h))  
            arms.set_colorkey(self.colors['white'])  
            arms.fill(self.colors['black'])  
            arms = pygame.transform.rotate(arms, np.rad2deg(direction))
            arms_rec = arms.get_rect(center=(ox,oy))
            self.components_surf.blit(arms,arms_rec)
            
            #body
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
            r = int(0.35*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
            r = int(0.3*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            if agent.type in self.agents_color:
                gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors[self.agents_color[agent.type]])
            elif agent.index == 'X':
                gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors[self.agents_color['adversary']])
            else:
                gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['lightgrey'])
            #eyes
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.4*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.15*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.6*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.15*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.4*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['white'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.6*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['white'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.4*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.85*(grid_height/dim[1]))
            r = int(0.07*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.6*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.85*(grid_height/dim[1]))
            r = int(0.07*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            # index
            agent_idx = str(agent.index)
            myfont = pygame.font.SysFont("Ariel", int(0.6*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
            label = myfont.render(agent_idx, True, self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.35*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.3*(grid_height/dim[1]))
            label =  pygame.transform.flip(label, False, True)
            self.components_surf.blit(label, (x,y))

        # box
        adhoc_agent = self.get_adhoc_agent()
        for task in self.components['tasks']:
            if not task.completed:
                rx, ry = task.position[0]*(grid_width/dim[0]),task.position[1]*(grid_height/dim[1])

                task_ret = pygame.Rect((rx+int(0.0*grid_width/dim[0]),ry+int(0.0*grid_height/dim[1])),\
                    (int(1*grid_width/dim[0]),int(1*grid_height/dim[1])))
                task_img = pygame.image.load(os.path.abspath("./imgs/levelbased/task_box.png"))
                task_img = pygame.transform.flip(task_img,False,True)
                task_img = pygame.transform.scale(task_img, task_ret.size)
                task_img = task_img.convert()

                
                dim_w, dim_h = self.shape
                direction = adhoc_agent.direction
                radius = np.sqrt(dim_w ** 2 + dim_h ** 2) * adhoc_agent.radius
                angle = 2 * np.pi * adhoc_agent.angle
                if (adhoc_agent.radius == 1. and adhoc_agent.angle == 1.) \
                 or is_visible(task.position,adhoc_agent.position,direction,
                radius,angle,self.components['obstacles'], self.vision_block):
                    self.components_surf.blit(task_img,task_ret)
                else:
                    self.grid_surf.blit(task_img,task_ret)
        
        # fog
        self.fog_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        self.fog_surf = self.fog_surf.convert_alpha()
        self.fog_surf.fill((self.colors['darkgrey'][0],self.colors['darkgrey'][1],self.colors['darkgrey'][2],100))
        self.fog_surf = pygame.transform.flip(self.fog_surf, False, True)

        # vision
        x = int(adhoc_agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
        y = int(adhoc_agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
        r = int(adhoc_agent.radius*np.sqrt((grid_width)**2+(grid_height)**2))
        self.vision_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        self.vision_surf = self.vision_surf.convert_alpha()
        gfxdraw.pie(self.vision_surf,x,y,r,
            int(np.rad2deg(adhoc_agent.direction-(np.pi*adhoc_agent.angle))),
            int(np.rad2deg(adhoc_agent.direction+(np.pi*adhoc_agent.angle))),
            (self.colors['black'][0],self.colors['black'][1],self.colors['black'][2],200))
        
        start_angle = adhoc_agent.direction-(np.pi*adhoc_agent.angle)
        stop_angle = adhoc_agent.direction+(np.pi*adhoc_agent.angle)
        theta = start_angle
        while theta <= stop_angle:
            pygame.draw.line(self.vision_surf,
                (self.colors['white'][0],self.colors['white'][1],self.colors['white'][2],100),
                    (x,y), (x+r*np.cos(theta),y+r*np.sin(theta)),10)
            theta += (stop_angle-start_angle)/100

        self.vision_surf = pygame.transform.flip(self.vision_surf, False, True)

        ##
        # Displaying
        ##
        self.grid_surf = pygame.transform.flip(self.grid_surf, False, True)
        self.components_surf = pygame.transform.flip(self.components_surf, False, True)
        self.screen.blit(self.grid_surf, (0.1*self.screen_size[0], 0.1*self.screen_size[1]))
        self.screen.blit(self.fog_surf, (0.1*self.screen_size[0], 0.1*self.screen_size[1]))
        self.screen.blit(self.vision_surf, (0.1*self.screen_size[0], 0.1*self.screen_size[1]))
        self.screen.blit(self.components_surf, (0.1*self.screen_size[0], 0.1*self.screen_size[1]))

        ##
        # Text
        ##
        act = self.action_dict[adhoc_agent.next_action] \
            if adhoc_agent.next_action is not None else None
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