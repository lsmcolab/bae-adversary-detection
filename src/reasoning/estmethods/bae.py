import numpy as np
import random as rd

import warnings
warnings.filterwarnings("ignore")

class BAE(object):

    def __init__(self, env, template_types, parameters_minmax, estimation_method = None):
        """
        Bayesian Adversary Estimation (BAE) class [1].

        ...

        Attributes
        ----------
        env : AdhocReasoningEnv object or extensions
            the environment to extract the current estimation information.
        teamplate_types : list
            a list of strings with the types to consider in the estimation.
        parameters_minmax : list
            a list of tuples with the range (min,max) for each parameter to
            consider in the estimation.
            
        Methods
        -------
        `check_teammates_estimation_set(self,env)`
            Initialise and update the set of teammates for estimation. If you
            are playing in a partially observable scenario, the teammates will
            be added in an online-manner.
        `update(self,env)`
            Update the estimation values for type and parameters according to
            the current state and environment setting.
        `is_in_the_previous_state(self, agent)`
            Check if an agent or teammate was in the previous analysed state.
        `get_estimation(self,env)`
            Get the complete estimation information for the target environment.
        `get_adversary_estimation(self,env)'
            Get the adversary estimation information for the target environment.
        `get_type_with_highest_probability(self,teammate_index)`
            Get the type with highest estimated probability for a defined 
            teammate.
        `get_parameter_for_selected_type(self, teammate, selected_type)`
            Get the last estimated parameters for a teammate and a selected 
            type.
        `weighted_sample_type(self,teammate_index)`
            Samples types for a teammate weighting types considering its 
            current estimation probabilities.
        `sample_impostor(self, env)`
            Sample a teammate to be categorised as impostor.
        `sample_state(self, env)`
            Sample a state (modified environment) to be used in the simulation 
            and estimation process of a reasoning method, but one of the 
            teammates will be categorised as impostor.
        `show_estimation(self, env)`
            Prints BAE's estimation type probabilities and parameter estimation
            to sys.stdout in a table format.
        
        References
        ----------
        [1] to appear
        """
        # initialising the BAE parameters
        self.template_types = template_types
        self.nparameters = len(parameters_minmax)
        self.parameters_minmax = parameters_minmax

        self.set_default()
        
        if estimation_method is None:
            self.estimation_method_name = None
            self.estimation_method = None
        else:
            self.estimation_method_name = estimation_method.upper()
            if self.estimation_method_name == 'AGA':
                from src.reasoning.estmethods import aga
                self.estimation_method = aga.AGA(env,self.template_types,\
                    self.parameters_minmax,self.grid_size,self.reward_factor,\
                    self.step_size,self.decay_step,self.degree,self.univariate)
            elif self.estimation_method_name == 'ABU':
                from src.reasoning.estmethods import abu
                self.estimation_method = abu.ABU(env,self.template_types,\
                    self.parameters_minmax,self.grid_size,self.reward_factor,self.degree)
            elif self.estimation_method_name == 'OEATE':
                from src.reasoning.estmethods import oeate
                self.estimation_method = oeate.OEATE(env,self.template_types,\
                    self.parameters_minmax,self.N,self.xi,self.mr,self.d,self.normalise,self.mode)
                
        self.previous_state = None

        # initialising the estimation for the agents
        self.teammate = {}
        self.check_teammates_estimation_set(env)

    def set_default(self):
        # AGA and ABU
        self.grid_size, self.reward_factor, self.degree = 100, 0.04, 2

        # AGA only
        self.step_size, self.decay_step, self.univariate = 0.01, 0.999, True
        
        # OEATE only
        self.N, self.xi, self.mr, self.d = 100, 2, 0.2, 100
        self.normalise, self.mode = np.mean, 'weight'

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            tindex = teammate.index
            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}

                # for each type
                for type in self.template_types:
                    # create the estimation set and the bag of estimators
                    self.teammate[tindex][type] = {}
                    if env.type_knowledge is False:
                        self.teammate[tindex][type]['probability_history'] = [1/len(self.template_types)]
                    else:
                        self.teammate[tindex][type]['probability_history'] = [1.0] if type == teammate.type else [0.0]

                    if env.parameter_knowledge is False:
                        self.teammate[tindex][type]['parameter_estimation_history'] = \
                            [[rd.uniform(self.parameters_minmax[n][0],self.parameters_minmax[n][1]) for n in range(self.nparameters)]]
                    else:
                        self.teammate[tindex][type]['parameter_estimation_history'] = [teammate.get_parameters()]
                
                self.teammate[tindex]['adversary'] = {}
                self.teammate[tindex]['adversary']['probability_history'] = [1/(len(env.components['agents'])-1)]
                self.teammate[tindex]['adversary']['parameter_estimation_history'] = []
                if env.parameter_knowledge:
                    self.teammate[tindex]['adversary']['parameter_estimation_history'].append(teammate.get_parameters())
                else:
                    self.teammate[tindex]['adversary']['parameter_estimation_history'].append(\
                        np.array([rd.uniform(self.parameters_minmax[n][0],self.parameters_minmax[n][1])\
                            for n in range(self.nparameters)]))
            

    def update(self,env,adversary_actions_prob_distribution,multi_tree):
        self.check_teammates_estimation_set(env)
        adhoc_agent = env.get_adhoc_agent()

        ###
        # CHECKING PREVIOUS STATE
        ###
        # BAE estimation requires, at least, one previous state in the history to start the estimation
        if self.previous_state is None or adversary_actions_prob_distribution is None:
            self.previous_state = env.copy()
            return self
        elif multi_tree:
            for agent in env.components['agents']:
                if agent.index != adhoc_agent.index and \
                (agent.index not in adversary_actions_prob_distribution or \
                adversary_actions_prob_distribution[agent.index] is None):
                    self.previous_state = env.copy()
                    return self

        ###
        # RUNNING PARAMETER AND TYPE ESTIMATION
        ###
        if self.estimation_method_name is not None:
            if self.estimation_method_name == 'OEATE':
                self.estimation_method.run(env)
            else:
                self.estimation_method.update(env)
    
        #####
        # START OF BAE ESTIMATION
        #####
        adversary_prob = 0.0
        for agent in env.components['agents']:
            # - if the agent is not the adhoc agent and it was seen in the previous state
            if agent.index != adhoc_agent.index and self.is_in_the_previous_state(agent):
                for type in self.template_types:
                    if self.estimation_method_name is not None:
                        self.teammate[agent.index][type]['probability_history'].append(\
                            self.estimation_method.teammate[agent.index][type]['probability_history'][-1])
                        self.teammate[agent.index][type]['parameter_estimation_history'].append(\
                            self.estimation_method.teammate[agent.index][type]['parameter_estimation_history'][-1])
                    else:
                        self.teammate[agent.index][type]['probability_history'].append(\
                            self.teammate[agent.index][type]['probability_history'][-1])
                        self.teammate[agent.index][type]['parameter_estimation_history'].append(\
                            self.teammate[agent.index][type]['parameter_estimation_history'][-1])

                # bayes update
                if multi_tree:
                    PBA = adversary_actions_prob_distribution[agent.index][str(agent.next_action)] + (10**(-10))
                    PA = self.teammate[agent.index]['adversary']['probability_history'][-1] + (10**(-10))
                    self.teammate[agent.index]['adversary']['probability_history'].append(PBA*PA)
                    adversary_prob += self.teammate[agent.index]['adversary']['probability_history'][-1]
                else:
                    PBA = adversary_actions_prob_distribution[str(agent.next_action)] + (10**(-10))
                    PA = self.teammate[agent.index]['adversary']['probability_history'][-1] + (10**(-10))
                    self.teammate[agent.index]['adversary']['probability_history'].append(PBA*PA)
                    adversary_prob += self.teammate[agent.index]['adversary']['probability_history'][-1]
        
        for agent in env.components['agents']:
            # - if the agent is not the adhoc agent and it was seen in the previous state
            if agent.index != adhoc_agent.index and self.is_in_the_previous_state(agent):
                if adversary_prob != 0.0:
                    self.teammate[agent.index]['adversary']['probability_history'][-1] /= adversary_prob
                else:
                    self.teammate[agent.index]['adversary']['probability_history'][-1] = 1/(len(env.components['agents'])-1)
        #####
        # END OF BAE ESTIMATION
        #####
        # Updating the previous state variable
        self.previous_state = env.copy()

        return self

    def is_in_the_previous_state(self, agent):
        """
        Check if an agent or teammate was in the previous analysed state.

        Parameters
        ----------
        agent: AdhocAgent object or extension
            target agent to verify.

        Returns
        -------
        check : bool
            Returns True if the agent was in the previous state, else False.
        """
        for i in range(0, len(self.previous_state.components["agents"])):
            if (self.previous_state.components["agents"][i].index == agent.index):
                return True
        return False

    def get_estimation(self,env):
        """
        Get the complete estimation information for the target environment.

        Parameters
        ----------
        env : AdhocReasoningEnv object or extensions
            the problem's environment.

        Returns
        -------
        type_probabilities : str
            the probability of types for each agent in the environment.
        estimated_parameters :
            the estimated parameters for each agent and type in the environment.
        indexes : str
            sorted indexes of each agent in the result vectors.
        """
        type_probabilities, estimated_parameters, indexes = [], [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index not in self.teammate.keys() and teammate.index != adhoc_agent.index: 
                indexes.append(teammate.index)   
                type_prob = np.array([-1 for i in range(0,len(self.template_types)+1)]) 
                parameter_est = [np.array([-1 for i in range(0,self.nparameters)]) for type in range(len(self.template_types)+1)]

                type_probabilities.append(list(type_prob))
                estimated_parameters.append(parameter_est)

            elif teammate.index != adhoc_agent.index:
                indexes.append(teammate.index)   
                # type result
                type_prob = []
                for type in self.template_types:
                    type_prob.append(self.teammate[teammate.index][type]['probability_history'][-1])
                type_prob.append(self.teammate[teammate.index]['adversary']['probability_history'][-1])
                type_probabilities.append(list(type_prob))

                # parameter result
                parameter_est = []
                for type in self.template_types:
                    parameter_est.append(self.teammate[teammate.index][type]['parameter_estimation_history'][-1])
                parameter_est.append(self.teammate[teammate.index]['adversary']['parameter_estimation_history'][-1])
                estimated_parameters.append([[p for p in ag] for ag in parameter_est])

        return type_probabilities, estimated_parameters, indexes

    def get_adversary_estimation(self,env):
        """
        Get the adversary estimation information for the target environment.

        Parameters
        ----------
        env : AdhocReasoningEnv object or extensions
            the problem's environment.

        Returns
        -------
        adversary_probabilities : str
            the probability of being an adversary for each agent in the 
            environment.
        indexes : str
            sorted indexes of each agent in the adversary probabilities vector.
        """
        adversary_probabilities, indexes = [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index not in self.teammate.keys() and teammate.index != adhoc_agent.index: 
                indexes.append(teammate.index)   
                type_prob = np.array([-1 for i in range(0,len(self.template_types)+1)]) 

                adversary_probabilities.append(list(type_prob)[-1])

            elif teammate.index != adhoc_agent.index:
                indexes.append(teammate.index)   
                # type result
                type_prob = []
                for type in self.template_types:
                    type_prob.append(self.teammate[teammate.index][type]['probability_history'][-1])
                type_prob.append(self.teammate[teammate.index]['adversary']['probability_history'][-1])
                adversary_probabilities.append(list(type_prob)[-1])

        return adversary_probabilities, indexes

    def get_type_with_highest_probability(self,teammate_index):
        """
        Get the type with highest estimated probability for a defined teammate.

        Parameters
        ----------
        teammate_index : str
            target teammate's index.

        Returns
        -------
        likely_type : str
            type with highest probability between all possible types.
        """
        last_types_probabilites = [self.teammate[teammate_index][type]['probability_history'][-1] for type in self.template_types]
        last_types_probabilites.append(self.teammate[teammate_index]['adversary']['probability_history'][-1])
        likely_type = self.template_types[last_types_probabilites.index(max(last_types_probabilites))]
        return  likely_type
    
    def get_parameter_for_selected_type(self, teammate, selected_type):
        """
        Get the last estimated parameters for a teammate and a selected type.

        Parameters
        ----------
        teammate : AdhocAgent object or extension
            target teammate.
        selected_type : str
            selected type to get the parameters.

        Returns
        -------
        parameter_est : list
            list of parameters in (min,max) range for the target teammate and
            the selected type.
        """
        parameter_est = self.teammate[teammate.index][selected_type]['parameter_estimation_history'][-1]
        return parameter_est

    def weighted_sample_type(self,teammate_index):
        """
        Samples types for a teammate weighting types considering its current 
        estimation probabilities.

        Parameters
        ----------
        teammate_index: str
            target teammate index to sample a type.

        Returns
        -------
        sampled_type : str
            weighted sampled type for the target teammate from the defined
            template types.
        """
        last_types_probabilites = [self.teammate[teammate_index][type]['probability_history'][-1] for type in self.template_types]
        sampled_type = rd.choices(self.template_types, last_types_probabilites)[0]
        return sampled_type

    def sample_impostor(self, env):
        """
        Sample a teammate to be categorised as impostor.

        Parameters
        ----------
        env : AdhocReasoningEnv object or extensions
            the problem's environment.

        Returns
        -------
        env : AdhocReasoningEnv object or extensions
            the modified environment with one agent being an impostor.
        """
        adhoc_agent = env.get_adhoc_agent()
        type_prob = []
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                type_prob.append(self.teammate[teammate.index]['adversary']['probability_history'][-1])
        
        teammates_index = [k for k in self.teammate.keys()]
        prob_sum = sum(type_prob)
        if prob_sum == 0.0:
            type_prob = [ 1/len(env.components['agents']) for teammate in env.components['agents']]
        else:
            type_prob = np.array(type_prob)
            type_prob /= prob_sum

        sampled_type = rd.choices(teammates_index,list(type_prob),k=1)
        return sampled_type[0]

    def sample_state(self, env, fixed_adv=None):
        """
        Sample a state (modified environment) to be used in the simulation and
        estimation process of a reasoning method, but one of the teammates will
        be categorised as impostor.

        Parameters
        ----------
        env : AdhocReasoningEnv object or extensions
            the problem's environment.
        fixed_adv : str (optional)
            specific agent to be the adversary while sampling the state.

        Returns
        -------
        env : AdhocReasoningEnv object or extensions
            the modified environment considering the sampled estimation.
        """
        adhoc_agent = env.get_adhoc_agent()
        impostor = self.sample_impostor(env) if fixed_adv is None else fixed_adv
        for teammate in env.components['agents']:
            if teammate.index == impostor:
                selected_type = 'adversary'
                selected_parameter = self.get_parameter_for_selected_type(teammate,selected_type)
                
                teammate.type = selected_type
                teammate.set_parameters(selected_parameter)

            elif teammate.index != adhoc_agent.index:
                selected_type = self.weighted_sample_type(teammate.index)
                selected_parameter = self.get_parameter_for_selected_type(teammate,selected_type)

                teammate.type = selected_type
                teammate.set_parameters(selected_parameter)

        return env

    def show_estimation(self, env, mode='both'):
        """
        Prints BAE's estimation type probabilities and parameter estimation to 
        sys.stdout in a table format.

        Parameters
        ----------
        env : AdhocReasoningEnv object or extensions
            the problem's environment. 
        mode : str
            if mode is equal to 'type' or 'parameter', it shows the estimation 
            for only the type or parameter estimation. If it is equal to 'both',
            it shows both estimations. Default is 'both'.
        """
        type_probabilities, estimated_parameters, indexes = self.get_estimation(env)
        types = [t for t in self.template_types]
        types.append('adversary (X)')

        if mode == 'both' or mode == 'type':
            print('|%10s|' %('Type'),end='')
            for type in types:
                print('|%10s|' %(str(type)),end='')
            print('')

            for i in range(len(type_probabilities)):
                print('|%10s|' %('Agent '+str(indexes[i])), end='')
                for j in range(len(type_probabilities[i])):
                    print('|%.8f|' %(type_probabilities[i][j]), end='')
                print('')
            print('--------------------------------------------------------')
        
        if  mode == 'both' or mode == 'parameter':
            print('|%10s|' %('Parameter'),end='')
            for type in types:
                print('|%15s|' %(str(type)),end='')
            print('')

            for i in range(len(estimated_parameters)):
                print('|%10s|' %('Agent '+str(indexes[i])), end='')
                for j in range(len(estimated_parameters[i])):
                    print('|',end='')
                    for k in range(len(estimated_parameters[i][j])):
                        print('%.2f ' %(estimated_parameters[i][j][k]), end='')
                    print('|',end='')
                print('')