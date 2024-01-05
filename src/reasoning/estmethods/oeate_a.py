import gc
import numpy as np
import random as rd

class Estimator(object):

    def __init__(self, parameters_minmax , type, d):
        # parameters and types
        self.minmax = parameters_minmax
        self.d = d
        self.parameters = []
        for i in range(len(parameters_minmax)):
            min_, max_ = d*parameters_minmax[i][0], (d+1)*parameters_minmax[i][1]
            self.parameters.append(np.random.randint(min_,max_)/d)
        self.type = type

        # evaluation
        self.success_counter = 0
        self.failure_counter = 0
        self.predicted_task = None
        self.time2completion = 0

    def predict_task(self, env, teammate):
        self.predicted_task = env.get_target(teammate.index, self.type, self.parameters)
        return self.predicted_task

    def copy(self):
        copied_estimator = Estimator(self.minmax,self.type,self.d)
        copied_estimator.parameters = np.array([p for p in self.parameters])
        copied_estimator.success_counter = self.success_counter
        copied_estimator.failure_counter = self.failure_counter
        copied_estimator.predicted_task = None
        return copied_estimator


class OEATE_A(object):

    def __init__(self, initial_state, template_types, parameters_minmax, N=100, xi=2, mr=0.2, d=100, normalise=np.mean, mode='weight'):
        # initialising the oeata parameters
        self.template_types = template_types
        self.nparameters = len(parameters_minmax)
        self.parameters_minmax = parameters_minmax
        self.N = N
        self.xi = xi
        self.mr = mr
        self.d = d
        self.normalise = normalise
        self.mode = mode

        # initialising the oeata teammates estimation set
        self.teammate = {}
        self.check_teammates_estimation_set(initial_state)

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            tindex = teammate.index
            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}
                self.teammate[tindex]['suspicious'] = 0
                self.teammate[tindex]['history'] = []

                # for each type
                for type in self.template_types:
                    # create the estimation set and the bag of estimators
                    self.teammate[tindex][type] = {}
                    self.teammate[tindex][type]['estimation_set'] = []
                    self.teammate[tindex][type]['bag_of_estimators'] = []

                    # initialize the estimation set with N estimators
                    while len(self.teammate[tindex][type]['estimation_set']) < self.N:
                        self.teammate[tindex][type]['estimation_set'].append(\
                            Estimator(self.parameters_minmax, type, self.d))
                        
                        
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

    def run(self, env):
        # checking if there are new teammates in the environment
        self.check_teammates_estimation_set(env)

        # check the tasks were completed
        just_finished_tasks = set([teammate.smart_parameters['last_completed_task'] \
            for teammate in env.components['agents'] if teammate.smart_parameters['last_completed_task'] != None])

        # running the oeata procedure
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                # if the agent accomplished a task
                # 1. Evaluation
                self.evaluation(teammate,just_finished_tasks)

                # 2. Generation
                self.generation(teammate)

                # 3. Update
                self.update(env, teammate, just_finished_tasks)
        
        type_probabilities, estimated_parameters, indexes = self.get_estimation(env)
        total_suspicious = 0
        for idx in range(len(indexes)):
            total_suspicious += self.teammate[indexes[idx]]['suspicious']

        for idx in range(len(indexes)):
            for type in range(len(self.template_types)):
                self.teammate[indexes[idx]][self.template_types[type]]['probability_history'].append(\
                    type_probabilities[idx][type])
                self.teammate[indexes[idx]][self.template_types[type]]['parameter_estimation_history'].append(\
                    estimated_parameters[idx][type])
                
            if total_suspicious != 0:
                self.teammate[indexes[idx]]['adversary']['probability_history'].append(\
                    self.teammate[indexes[idx]]['suspicious']/total_suspicious)
            else:
                self.teammate[indexes[idx]]['adversary']['probability_history'].append(\
                    self.teammate[indexes[idx]]['adversary']['probability_history'][-1])
        
        return self

    def evaluation(self, teammate, just_finished_tasks):

        for type in self.template_types:
            removed = 0
            for i in range(self.N):
                # verifying if the estimator correctly estimates the task
                # CORRECT
                if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                 self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index in just_finished_tasks:
                    # adding to the bag of estimators
                    self.teammate[teammate.index][type]['bag_of_estimators'].append(self.teammate[teammate.index][type]['estimation_set'][i].copy())

                    # updating the success counter
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter = 0
                    self.teammate[teammate.index][type]['estimation_set'][i].success_counter += 1
                
                # INCORRECT
                else:
                    #updating the failure counter
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter += 1

                    # checking if the task must be removed
                    f_e = self.teammate[teammate.index][type]['estimation_set'][i].failure_counter
                    if  f_e == 5:
                        removed += 1
                        self.teammate[teammate.index][type]['estimation_set'][i].parameters = None
                        gc.collect()

    def generation(self, teammate):
        for type in self.template_types:
            # 1. Calculating the number of mutations
            n_removed_estimators = sum([self.teammate[teammate.index][type]['estimation_set'][i].parameters is None for i in range(self.N)])
            if len(self.teammate[teammate.index][type]['bag_of_estimators']) > 0:
                nmutations = int(self.mr*n_removed_estimators)
            else:
                nmutations = 0

            # 2. Generating new estimators
            for i in range(self.N):
                if self.teammate[teammate.index][type]['estimation_set'][i].parameters is None:
                    # from the bag
                    if nmutations < 0 and len(self.teammate[teammate.index][type]['bag_of_estimators']) > 1:
                        new_estimator = Estimator(self.parameters_minmax, type, self.d)
                        for n in range(self.nparameters):
                            sampled_estimator = rd.sample(self.teammate[teammate.index][type]['bag_of_estimators'],1)[0]
                            new_estimator.parameters[n] = sampled_estimator.parameters[n]
                        self.teammate[teammate.index][type]['estimation_set'][i] = new_estimator
                    # from the uniform distribution
                    else:
                        self.teammate[teammate.index][type]['estimation_set'][i] = Estimator(self.parameters_minmax, type, self.d)
                        nmutations -= 1
                    
    def update(self, env, teammate, just_finished_tasks):
        # updating the estimator-predicted task
        just_finished_indexes = [task.index for task in just_finished_tasks]
        
        for type in self.template_types:
            total_success = 0
            for i in range(self.N):
                # if the selected task was accomplished by other agent
                if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                 self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index in just_finished_indexes:
                    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = \
                         self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                    self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()
                    self.teammate[teammate.index][type]['estimation_set'][i].time2completion = 0

                # or if the estimator did not select any task
                elif self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is None:
                    #print("Task is None for an estimator of ", teammate.index)
                    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task =\
                        self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                    self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()
                    self.teammate[teammate.index][type]['estimation_set'][i].time2completion = 0
                    
            
                # else update the time to completion
                else:
                    self.teammate[teammate.index][type]['estimation_set'][i].time2completion += 1

                # adversary
                c_e = self.teammate[teammate.index][type]['estimation_set'][i].success_counter
                total_success += c_e

        if total_success == 0:
            self.teammate[teammate.index]['suspicious'] += 1
                    
    def get_estimation(self, env):
        type_probabilities, estimated_parameters, indexes = [], [], []

        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index not in self.teammate.keys() and teammate.index != adhoc_agent.index:
                indexes.append(teammate.index)   
                print("Haven't seen ", teammate.index)
                type_prob = np.array([-1 for i in range(0,len(self.template_types))]) 
                parameter_est = []
                for type in self.template_types:
                    parameter_est.append(np.array([-1 for i in range(0,self.nparameters)]))
                type_probabilities.append(list(type_prob))
                estimated_parameters.append(parameter_est)
                continue

            if teammate.index != adhoc_agent.index: 
                indexes.append(teammate.index)   
                # type result
                type_prob = np.zeros(len(self.template_types))
                for i in range(len(self.template_types)):
                    type = self.template_types[i]
                    type_prob[i] = np.sum([self.teammate[teammate.index][type]['estimation_set'][i].success_counter \
                                        if self.teammate[teammate.index][type]['estimation_set'][i].success_counter > 0 else 0 for i in range(self.N)])

                sum_type_prob = np.sum(type_prob)
                if sum_type_prob != 0:
                    type_prob /= sum_type_prob
                else:
                    type_prob += 1
                    type_prob /= len(self.template_types)

                type_probabilities.append(list(type_prob))
                type_probabilities[-1].append(self.teammate[teammate.index]['adversary']['probability_history'][-1])

                # parameter result
                parameter_est = []
                for i in range(len(self.template_types)):
                    type = self.template_types[i]
                    success_sum = sum([e.success_counter for e in self.teammate[teammate.index][type]['estimation_set']])
                    
                    # if we have made estimations
                    if success_sum > 0:
                        # unif estimation
                        if self.mode == 'unif':
                            parameter_est.append(\
                             self.normalise([np.array(e.parameters) \
                              for e in self.teammate[teammate.index][type]['estimation_set']],axis=0)\
                            )
                        elif self.mode == 'weight':
                            # weight estimation
                            parameter_est.append(\
                            np.sum([e.success_counter*np.array(e.parameters)/success_sum \
                            for e in self.teammate[teammate.index][type]['estimation_set']],axis=0)\
                            )
                        else:
                            raise NotImplemented

                    # else, mean
                    else:
                        parameter_est.append(\
                         np.mean([np.array(e.parameters) \
                          for e in self.teammate[teammate.index][type]['estimation_set']],axis=0)\
                        )
   
                estimated_parameters.append(parameter_est)

        return type_probabilities, estimated_parameters, indexes

    def sample_state(self, env):
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                selected_type = self.sample_type_for_agent(teammate)
                selected_parameter = self.get_parameter_for_selected_type(teammate,selected_type)

                teammate.type = selected_type
                teammate.set_parameters(selected_parameter)

        return env

    def sample_type_for_agent(self, teammate):
        type_prob = np.zeros(len(self.template_types))
        for i in range(len(self.template_types)):
            type = self.template_types[i]
            type_prob[i] = np.sum([self.teammate[teammate.index][type]['estimation_set'][i].success_counter \
                                if self.teammate[teammate.index][type]['estimation_set'][i].success_counter > 0 else 0 for i in range(self.N)])

        sum_type_prob = np.sum(type_prob)
        if sum_type_prob != 0:
            type_prob /= sum_type_prob
        else:
            type_prob += 1
            type_prob /= len(self.template_types)
        
        sampled_type = rd.choices(self.template_types,type_prob,k=1)
        return sampled_type[0]

    def get_parameter_for_selected_type(self, teammate, selected_type):
        success_sum = sum([e.success_counter for e in self.teammate[teammate.index][selected_type]['estimation_set']])
                    
        # unif set
        if self.mode == 'unif':
            sample_set = self.teammate[teammate.index][selected_type]['estimation_set']
        # weight set
        elif self.mode == 'weight':
            if success_sum > 0:
                sample_set = [e for e in self.teammate[teammate.index][selected_type]['estimation_set'] for i in range(e.success_counter)]
            else:
                sample_set = self.teammate[teammate.index][selected_type]['estimation_set']
        else:
            raise NotImplemented

        parameter_est = rd.sample(sample_set,1)[0].parameters
        return parameter_est

    def show_estimation(self, env, mode='both'):
        type_probabilities, estimated_parameters, indexes = self.get_estimation(env)
        types = [t for t in self.template_types]

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
        
        if  mode == 'both' or mode == 'parameter':
            print('|%10s| ==========' %('Parameters'))
            for i in range(len(estimated_parameters)):
                print('|xxxxxxxxxx| Agent %3s:' %(indexes[i]))
                for j in range(len(types)-1):
                    print('|xxxxxxxxxx|xxxxx| Type %3s:' %(types[j]), estimated_parameters[i][j])