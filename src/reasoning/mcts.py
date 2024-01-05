from src.reasoning.node import QNode
import random
import time
from src.reasoning.estimation import type_parameter_estimation

class MCTS(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.max_depth = max_depth
        self.max_it = max_it
        self.c = 0.5
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        ###
        # Further settings
        ###
        target = kwargs.get('target')
        if target is not None:
            self.target = target
            self.initial_target = target
        else: #default
            self.target = 'max'
            self.initial_target = 'max'

        adversary_mode = kwargs.get('adversary')
        if adversary_mode is not None:
            self.adversary = adversary_mode
        else: #default
            self.adversary = False

        multi_tree = kwargs.get('multi_tree')
        if multi_tree is not None:
            self.multi_tree = multi_tree
        else: #default
            self.multi_tree = False

        stack_size = kwargs.get('state_stack_size')
        if stack_size is not None:
            self.state_stack_size = stack_size
        else: #default
            self.state_stack_size = 1

        ###
        # Evaluation
        ###
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def change_paradigm(self):
        if self.target == 'max':
            return 'min'
        elif self.target == 'min':
            return 'max'
        else:
            raise NotImplemented

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = QNode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        return random.choice(state.get_actions_list())

    def rollout(self,node):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.depth += 1

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return reward +\
            self.discount_factor*self.rollout(node)

    def get_rollout_node(self,node):
        tmp_state = node.state.copy()
        depth = node.depth
        return QNode(action=None,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node):
        # 1. Checking the stop condition
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node)

        self.simulation_count += 1
        start_t = time.time()
        
        # 3. Selecting the best action
        action = node.select_action(coef=self.c,mode=self.target)
        self.target = self.change_paradigm() if self.adversary else self.target     

        # 4. Simulating the action
        (next_node, reward) = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        if next_node.action in [c.action for c in node.children]:
            for child in node.children:
                if next_node.action == child.action:
                    child.state = next_node.state.copy()
                    next_node = child
                    break
        else:
            node.children.append(next_node)

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 7. Calculating the reward, quality and updating the node
        R = reward + float(self.discount_factor * self.simulate(next_node))
        node.visits += 1
        node.update(action, R)
        return R

    def search(self, node):
        # Performing the Monte-Carlo Tree Search
        if self.multi_tree:
            # multiple root nodes here as ``node''
            for teammate_index in node:
                base_state = node[teammate_index].state.copy()
                agent = base_state.get_adhoc_agent()
                        
                it = 0
                while it < self.max_it:
                    node[teammate_index].state = agent.smart_parameters['estimation'].\
                        sample_state(base_state,fixed_adv=teammate_index)
                    self.target = self.initial_target
                    self.simulate(node[teammate_index])
                    it += 1
        else:
            base_state = node.state.copy()
            agent = base_state.get_adhoc_agent()
            
            it = 0
            while it < self.max_it:
                # Sampling state (if there is missing information about the env)
                if 'estimation_method' in agent.smart_parameters:
                    node.state = agent.smart_parameters['estimation'].sample_state(base_state)
                    self.target = self.initial_target
                    self.simulate(node)
                else:
                    node.state = base_state.copy()
                    self.target = self.initial_target
                    self.simulate(node)
                it += 1

        self.target = self.initial_target

        if self.multi_tree:
            # expected/belief value
            state = base_state.copy()
            fake_node = QNode(None,state,None,None)
            adv_prob, indexes = agent.smart_parameters['estimation'].get_adversary_estimation(fake_node.state)
            for teammate_index in node:
                for a in node[teammate_index].qtable:
                    adv = indexes.index(teammate_index)
                    fake_node.qtable[a]['qvalue'] += adv_prob[adv]*node[teammate_index].qtable[a]['qvalue']
                    fake_node.qtable[a]['trials'] += node[teammate_index].qtable[a]['trials']
            return fake_node.get_best_action(self.target)
        else:
            # estimated value
            return node.get_best_action(self.target)

    def find_new_root(self,previous_action,previous_root,adversary_last_action=None):
        # 1. If the root doesn't exist yet, create it
        # - NOTE: The root is always represented as an "observation node" since the next node
        # must be an action node.
        new_root, info = None, {'adversary_last_action':None,\
                            'adversary_actions_prob_distribution':None}
        if previous_root is None:
            return new_root, info

        # 2. Else, walk on the tree to find the new one (giving the previous information)
        # a. walking over action nodes
        for child in previous_root.children:
            if child.action == previous_action:
                new_root = child
                break

        # - if we didn't find the action node, create a new root
        if new_root is None:
            return new_root, info

        # b. checking the adversary condition
        if self.adversary:
            adv_target = self.change_paradigm()
            
            max_reward = new_root.state.get_max_reward()
            info['adversary_actions_prob_distribution'] = \
                new_root.get_actions_prob_distribution(adv_target,max_reward=max_reward)
            
            if adversary_last_action:
                info['adversary_last_action'] = adversary_last_action
                for child in new_root.children:
                    if child.action == adversary_last_action:
                        new_root = child
                        break
            else:
                adversary_last_action = new_root.get_best_action(adv_target)
                for child in new_root.children:
                    if child.action == adversary_last_action:
                        new_root = child
                        break

            info['adversary_last_action'] = adversary_last_action

        # - if we didn't find the action node, create a new root
        if new_root is None:
            return new_root, info

        # 3. Definig the new root and updating the depth
        new_root.parent = None
        new_root.update_depth(0)
        return new_root, info
    
    def initialise_root_node(self,agent,state):
        if self.multi_tree:
            root_node = {}
            for ag in state.components['agents']:
                if ag.index != agent.index:
                    root_node[ag.index] = QNode(action=None,state=state,depth=0,parent=None)
        else:
            root_node = QNode(action=None,state=state,depth=0,parent=None)
        return root_node

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action

        # 2. Defining the root of our search tree
        # via initialising the tree
        if 'search_tree' not in agent.smart_parameters:
            root_node = self.initialise_root_node(agent,state)  
            if self.multi_tree:
                info = {}
                for ag in state.components['agents']:
                    if ag.index != agent.index:
                        info[ag.index] = {'adversary_last_action':None, 'adversary_actions_prob_distribution':None}
            else:
                info = {'adversary_last_action':None, 'adversary_actions_prob_distribution':None}
        # or advancing within the existent tree
        else:
            if self.adversary and self.multi_tree:
                root_node, info = {}, {}
                for ag in state.components['agents']:
                    if ag.index != agent.index:
                        root_node[ag.index], info[ag.index] = self.find_new_root(previous_action,\
                            agent.smart_parameters['search_tree'][ag.index], adversary_last_action=ag.next_action)
            elif self.adversary and 'adversary_last_action' in agent.smart_parameters:
                root_node, info = self.find_new_root(previous_action,\
                    agent.smart_parameters['search_tree'], adversary_last_action=\
                    agent.smart_parameters['adversary_last_action'])
            else:
                root_node, info = self.find_new_root(previous_action,\
                agent.smart_parameters['search_tree'])
                
        # if no valid node was found, reset the tree
        if self.multi_tree:
            for ag in root_node:
                if root_node[ag] is None:
                    root_node[ag] = QNode(action=None,state=state,depth=0,parent=None)
        elif root_node is None:
            root_node = self.initialise_root_node(agent,state)

        # 3. Updating possible estimations arguments
        if 'estimation_kwargs' in agent.smart_parameters:
            if self.multi_tree:
                agent.smart_parameters['estimation_kwargs']\
                    ['multi_tree'] = True

                agent.smart_parameters['estimation_kwargs']\
                    ['adversary_last_action'] = {}
                agent.smart_parameters['estimation_kwargs']\
                    ['adversary_actions_prob_distribution'] = {}
                
                for k in info:
                    agent.smart_parameters['estimation_kwargs']\
                    ['adversary_last_action'][k] = info[k]['adversary_last_action']
            
                    agent.smart_parameters['estimation_kwargs']\
                        ['adversary_actions_prob_distribution'][k] =\
                            info[k]['adversary_actions_prob_distribution']
            else:
                agent.smart_parameters['estimation_kwargs']\
                    ['multi_tree'] = False
                agent.smart_parameters['estimation_kwargs']\
                    ['adversary_last_action'] = info['adversary_last_action']
                agent.smart_parameters['estimation_kwargs']\
                    ['adversary_actions_prob_distribution'] =\
                        info['adversary_actions_prob_distribution']
        
        # 4. Updating state information
        if self.multi_tree:
            base_state = None
            for ag in root_node:
                root_node[ag].state = state.copy()
                base_state = state.copy()
        else:
            root_node.state = state.copy()
         
        # 5. Initialising/Updating type and parameters estimation
        if 'estimation_method' in agent.smart_parameters:
            if self.multi_tree:
                root_node[ag].state, agent.smart_parameters['estimation'] = \
                type_parameter_estimation(\
                    base_state,agent,\
                    type=agent.smart_parameters['estimation_method'],\
                    **agent.smart_parameters['estimation_kwargs'])
            else:
                root_node.state, agent.smart_parameters['estimation'] = \
                type_parameter_estimation(\
                    root_node.state,agent,\
                    type=agent.smart_parameters['estimation_method'],\
                    **agent.smart_parameters['estimation_kwargs'])

        # 6. Searching for the best action within the tree
        best_action = self.search(root_node)

        # 7. Returning the best action
        return best_action, root_node, {'nrollouts': self.rollout_count,'nsimulations':self.simulation_count}

def mcts_planning(env, agent, max_depth=25, max_it=500, **kwargs):   
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    mcts = MCTS(max_depth, max_it, kwargs) if 'mcts' not \
     in agent.smart_parameters else agent.smart_parameters['mcts']
    next_action, search_tree, info = mcts.planning(copy_env,agent)
    #search_tree.show_qtable()

    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    agent.smart_parameters['count'] = info
    return next_action,None

def mcts_min_planning(env, agent, max_depth=25, max_it=500, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    kwargs['target'] = 'min'
    mcts = MCTS(max_depth, max_it, kwargs) if 'mcts' not \
     in agent.smart_parameters else agent.smart_parameters['mcts']
    next_action, search_tree, info = mcts.planning(copy_env,agent)
    #search_tree.show_qtable()
    
    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    agent.smart_parameters['count'] = info

    # - model creation and update step
    #from ..log import TrainedModel
    #model = TrainedModel('AdversaryModel_'+env.name)
    #model.update(env,search_tree.actions,search_tree.qtable)

    return next_action,None

def mcts_multi_tree_planning(env, agent, max_depth=25, max_it=500, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    kwargs['multi_tree'] = True
    mcts = MCTS(max_depth, max_it, kwargs) if 'mcts' not \
     in agent.smart_parameters else agent.smart_parameters['mcts']
    next_action, search_tree, info = mcts.planning(copy_env,agent)
    
    #for k in search_tree:
    #    print('==============')
    #    search_tree[k].show_qtable()
    #    print('==============\n')
    
    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    agent.smart_parameters['count'] = info
    return next_action,None