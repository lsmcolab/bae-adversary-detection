import math
import numpy as np
import random

from src.reasoning.qlearn import \
    create_qtable, create_etable, entropy, \
    ucb_select_action, iucb_select_action

"""
    Traditional tree search nodes
"""
class Node(object):

    def __init__(self, state, depth, parent=None):
        self.state = state
        self.depth = depth
        self.parent = parent
        self.children = []
        self.visits = 0 

    def add_child(self,state):
        child = Node(state,self.depth+1,self)
        self.children.append(child)
        return child

    def remove_child(self,child):
        for c in self.children:
            if c == child:
                self.children.remove(child)
                break

    def update_depth(self,new_depth):
        self.depth = new_depth
        for c in self.children:
            c.update_depth(new_depth+1)

"""
    Quality search nodes
"""
class QNode(Node):

    def __init__(self,action, state, depth, parent=None):
        super(QNode,self).__init__(state,depth,parent)
        self.value = 0
        self.action = action
        self.actions = state.get_actions_list()
        self.qtable = create_qtable(self.actions)

    def update(self, action, result):
        self.qtable[str(action)]['trials'] += 1
        self.qtable[str(action)]['sumvalue'] += result
        self.qtable[str(action)]['qvalue'] += \
            (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])

        self.value += (result-self.value)/self.visits

    def select_action(self,coef=0.5,mode='ucb'):
        # UCB
        if mode == 'ucb' or mode == 'max' or mode == 'ucb-max':
            return ucb_select_action(self,c=coef,mode=mode)
        elif mode == 'min'  or mode == 'ucb-min':
            return ucb_select_action(self,c=coef,mode=mode)
        # I-UCB
        elif mode == 'iucb' or mode == 'iucb-max':
            return iucb_select_action(self,alpha=coef,mode='max')
        elif mode == 'iucb-min':
            return iucb_select_action(self,alpha=coef,mode='min')
        # Not Implemented
        else:
            print('Invalid best action mode:',mode)
            raise NotImplemented

    def get_actions_prob_distribution(self, mode='max', max_reward=1):
        prob_distribution = {}
        
        norm = 0.0
        for a in self.qtable:
            if mode == 'max':
                prob_distribution[a] = self.qtable[a]['qvalue']
            elif mode == 'min':
                prob_distribution[a] = (max_reward-self.qtable[a]['qvalue'])
            else:
                raise NotImplemented
            norm += prob_distribution[a]
        
        if norm == 0.0:
            for a in prob_distribution:
                prob_distribution[a] = 1/len(prob_distribution)
        else:
            for a in prob_distribution:
                prob_distribution[a] /= norm

        return prob_distribution

    def get_best_action(self,mode='max'):
        # 1. Intialising the support variables
        # - maximisation
        if mode == 'max' or mode == 'ucb':
            target = 'max'
            best_action, bestQ = None, -100000000000
        # - minimisation
        elif mode == 'min' or mode == 'ucb-min':
            target = 'min'
            best_action, bestQ = None, 100000000000
        # - not implemented
        else:
            print('Invalid best action mode:',mode)
            raise NotImplemented

        # 2. Looking for the best action (max qvalue action)
        for a in self.actions:
            if target == 'max' and  \
             self.qtable[str(a)]['qvalue'] > bestQ  and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = self.qtable[str(a)]['qvalue']
                best_action = a
            elif target == 'min' and \
             self.qtable[str(a)]['qvalue'] < bestQ and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = self.qtable[str(a)]['qvalue']
                best_action = a

        # 3. Checking if a tie case exists
        tieCases = []
        for a in self.actions:
            if self.qtable[str(a)]['qvalue'] == bestQ:
                tieCases.append(a)

        if len(tieCases) > 1:
            # trying tie break by number of visits
            trials = [self.qtable[str(a)]['trials'] for a in tieCases]
            max_trial = max(trials)
            trialTieCases = []
            for a in tieCases:
                if self.qtable[str(a)]['trials'] == max_trial:
                    trialTieCases.append(a)

            if len(trialTieCases) > 1:
                best_action = random.choice(trialTieCases)
            else:
                best_action = trialTieCases[0]

        # 4. Returning the best action
        if best_action is None:
            best_action = random.sample(self.actions,1)[0]
        
        return best_action
    
    def size_in_memory(self):
        from pympler import asizeof
        node_size = asizeof.asizeof(self)
        return node_size*(10**(-6))

    def show_qtable(self):
        print('%8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1]), reverse=True)
        
        for a in action_dict:
            if hasattr(self.state, 'action_dict'):
                print('%8s %8.4f %8.4f %8d' % (self.state.action_dict[a],self.qtable[str(a)]['qvalue'],\
                                            self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
            else:
                print('%8s %8.4f %8.4f %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                            self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')

class ANode(QNode):

    def __init__(self,action, state, depth, parent=None):
        super(ANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,observation):
        state = self.state.copy()
        child = ONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

    
    def get_child(self,observation):
        for child in self.children:
            if child.observation == observation:
                return child
        return None

class ONode(QNode):

    def __init__(self,observation, state, depth, parent=None):
        super(ONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.particles_set = {}
        
    def add_child(self,state,action):
        child = ANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def get_child(self,action):
        for child in self.children:
            if child.action == action:
                return child
        return None

def find_new_PO_root(current_state, previous_action, current_observation, 
 agent, previous_root, adversary=False):
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # b. walking over observation nodes
    for child in action_node.children:
        if child.state.observation_is_equal(current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # c. checking the adversary condition
    if adversary:
        action_node, observation_node = None, None
        for child in new_root.children:
            if child.action == agent.smart_parameters['adversary_last_action']:
                action_node = child
                break
        # - if we didn't find the action node, create a new root
        if action_node is None:
            new_root = ONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root

        for child in action_node.children:
            if child.state.observation_is_equal(\
             agent.smart_parameters['adversary_last_observation']):
                observation_node = child
                break
        # - if we didn't find the action node, create a new root
        if observation_node is None:
            new_root = ONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root

def particle_revigoration(env,agent,root,k):
    # 1. Copying the current root particle filter
    current_particle_filter = []
    for particle in root.particle_filter:
        current_particle_filter.append(particle)
    
    # 2. Reinvigorating particles for the new particle filter or
    # picking particles from the uniform distribution
    root.particle_filter = []
    if len(current_particle_filter) > 0: # particle ~ F_r
        while(len(root.particle_filter) < k):
            particle = random.sample(current_particle_filter,1)[0]
            root.particle_filter.append(particle)
    else: # particle ~ U
        while(len(root.particle_filter) < k):
            particle = env.sample_state(agent)
            root.particle_filter.append(particle)

"""
    IB-POMCP nodes
"""
class IANode(ANode):

    def __init__(self,action, state, depth, parent=None):
        super(IANode,self).__init__(action,state,depth,parent)
        self.action = action

    def add_child(self,observation):
        state = self.state.copy()
        child = IONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class IONode(ONode):

    def __init__(self,observation, state, depth, parent=None):
        super(IONode,self).__init__(None,state,depth,parent)
        self.observation = observation
        
        self.particle_filter = []   

        self.observation_distribution = {}
        self.entropy = 0.0
        self.cumentropy = 0.0
        self.max_entropy = 1.0

        self.etable = create_etable(self.actions) 
        
    def add_child(self,state,action):
        child = IANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def add_to_observation_distribution(self,observations_state):
        # adding info to state distribution
        for state in observations_state:
            key = state.hash_observation()
            if key in self.observation_distribution:
                self.observation_distribution[key] += 1
            else:
                self.observation_distribution[key] = 1
    
    def get_alpha(self):
        adjust_value = 0.2
        if self.visits == 0:
            return 1 - adjust_value

        # Old Version
        decaying_factor = math.e*math.log(self.visits)/self.visits
        entropy_value_trend = self.cumentropy/\
                            (self.visits*self.max_entropy)
        norm_entropy = decaying_factor*entropy_value_trend

        return (1 - 2*adjust_value)*(norm_entropy) +adjust_value

    def update(self, action, result):
        # Q-table update
        self.qtable[str(action)]['trials'] += 1

        self.qtable[str(action)]['sumvalue'] += result

        self.qtable[str(action)]['qvalue'] += \
            (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])
        self.value += (result-self.value)/self.visits

        # Observation-table update
        result_entropy = entropy(self.observation_distribution)
        self.etable[str(action)]['trials'] += 1

        self.etable[str(action)]['cumentropy'] += result_entropy
        self.cumentropy += result_entropy

        self.etable[str(action)]['entropy'] += \
            (float(result_entropy) - self.etable[str(action)]['entropy']) /\
                 float(self.etable[str(action)]['trials'])
        self.entropy += (result_entropy - self.entropy) / self.visits

        if result_entropy > self.etable[str(action)]['max_entropy']:
            self.etable[str(action)]['max_entropy'] = result_entropy 

        if result_entropy > self.max_entropy:
            self.max_entropy = self.entropy

    
    def get_best_action(self,alpha,mode='iucb-max'):
        # 1. Intialising the support variables
        # - maximisation
        if mode == 'iucb-max' or mode == 'iucb':
            target = 'max'
            best_action, bestQ = None, -100000000000
        # - minimisation
        elif mode == 'iucb-min':
            target = 'min'
            best_action, bestQ = None, 100000000000
        # - not implemented
        else:
            print('Invalid best action mode:',mode)
            raise NotImplemented

        # 2. Looking for the best action (max qvalue action)
        actionsQ = {}
        for a in self.actions:
            actionsQ[str(a)] = (1-alpha)*self.qtable[str(a)]['qvalue'] + \
             (alpha)*(self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy'])

            # maximisation case
            if target == 'max' and  actionsQ[str(a)] > bestQ  and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = actionsQ[str(a)]
                best_action = a

            # minimisation case
            elif target == 'min' and actionsQ[str(a)] < bestQ and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = actionsQ[str(a)]
                best_action = a

        # 3. Checking if a tie case exists
        tieCases = []
        for a in self.actions:
            if actionsQ[str(a)] == bestQ:
                tieCases.append(a)

        if len(tieCases) > 1:
            # 3.1. trying tie break by number of visits
            trials = [self.qtable[str(a)]['trials'] for a in tieCases]
            max_trial = max(trials)
            trialTieCases = []
            for a in tieCases:
                if self.qtable[str(a)]['trials'] == max_trial:
                    trialTieCases.append(a)

            # 3.2. checking if a tie case persists
            if len(trialTieCases) > 1:
                best_action = random.choice(trialTieCases)
            else:
                best_action = trialTieCases[0]
        else:
            best_action = tieCases[0]

        # 4. Returning the best action
        if(best_action==None):
            best_action = random.sample(self.actions,1)[0]
            
        return best_action

    def show_qtable(self):
        print('%8s %8s %8s %8s | %8s %8s %8s' % \
            ('Action','Q-Value','SumValue','Trials','Entropy','MaxEntropy','NormEntropy'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],\
                self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy'],\
                    self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1],action_dict[x][2]), reverse=True)
        
        for a in action_dict:
            if hasattr(self.state, 'action_dict'):
                print('%8s %8.4f %8.4f %8d | %8.4f %8.4f %8.4f' \
                    % (self.state.action_dict[a],self.qtable[str(a)]['qvalue'],\
                    self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials'],\
                    self.etable[str(a)]['entropy'],self.etable[str(a)]['max_entropy'],\
                    self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy']))
            else:
                print('%8s %8.4f %8.4f %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                            self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')

"""
    rho-POMCP nodes
"""
class RhoANode(ANode):

    def __init__(self,action, state, depth, parent=None):
        super(RhoANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,observation):
        state = self.state.copy()
        child = RhoONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class RhoONode(ONode):

    def __init__(self,observation, state, depth, parent=None):
        super(RhoONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.cummulative_bag = {}
    
    def add_child(self,state,action):
        child = RhoANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def add_to_cummulative_bag(self,particle,action):
        obs_p = particle.get_obs_p(action)[1]
        hash_key = particle.hash_state()
        if hash_key not in self.cummulative_bag:
            self.cummulative_bag[hash_key] =  [particle,obs_p]
        else:
            self.cummulative_bag[hash_key] =  [particle,self.cummulative_bag[hash_key][1] + obs_p]