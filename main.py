##
## Main code: template to run a single (default) experiment on AdLeap-MAS
##
# 1. Setting the environment
method = 'mcts'                 # choose your method (we used only mcts in this paper)
scenario_id = 5                 # define your scenario configuration. Options: [5,6,7,8]
estimation_method = 'bae'       # choosing your estimation method. Options: ['bae','aga','abu']

display = True                  # choosing to turn on or off the display


# 2. Creating the environment
# a. importing necessary modules
import time
from src.log import EstimationLogFile
from src.envs.LevelForagingEnv import load_default_scenario

# b. creating the environment
env, scenario_id = load_default_scenario(method,scenario_id,display=display)
state = env.reset()

if estimation_method.upper() == 'BAE' or estimation_method.upper() == 'OEATE_A':
    template_types = env.components['template_types']
    if 'adversary' in template_types:
        template_types.remove('adversary')
        env.state_set.initial_components = env.copy_components(env.components)
    state = env.reset()
        
    estimation_kwargs = { 
        'template_types':template_types,\
        'parameters_minmax':[(0.5,1),(0.5,1),(0.5,1)],
        'adversary_last_action':None
        }
    
    if estimation_method.upper() == 'BAE':
        from src.reasoning.estmethods import bae
        supmethod = bae.BAE(env,estimation_kwargs['template_types'],\
                    estimation_kwargs['parameters_minmax']).estimation_method_name
        if supmethod is None:
            method_name = estimation_method
        else:
            method_name = estimation_method+'_'+supmethod
    else:
        method_name = estimation_method
else:
    template_types = env.components['template_types']
    estimation_kwargs = { 
        'template_types':template_types,\
        'parameters_minmax':[(0.5,1),(0.5,1),(0.5,1)],
        'adversary_last_action':None
        }
    
    method_name = estimation_method

log = EstimationLogFile('LevelForagingEnv',scenario_id,method,method_name,0,\
    estimation_kwargs['template_types'],estimation_kwargs['parameters_minmax'])

###
# ADLEAP-MAS MAIN ROUTINE
###
done, max_episode = False, 200
while env.episode < max_episode:
    #print('|||| Episode',env.episode)
    # 1. Importing agent method
    adhoc_agent = env.get_adhoc_agent()
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    # Adversarial problems
    start = time.time()
    if env.is_adversarial():
        adhoc_agent.smart_parameters['estimation_method'] = estimation_method
        adhoc_agent.smart_parameters['estimation_kwargs'] = estimation_kwargs
        action, target = method(state, adhoc_agent, adversary = True, mode='max')
    # Foraging problems
    else:
        action, target = method(state, adhoc_agent)
    end = time.time()
    memory_usage = adhoc_agent.smart_parameters['search_tree'].size_in_memory()

    # 3. Taking a step in the environment
    state, reward, done, info = env.step(action)

    # if you want to visualize the ad hoc agent memory abou the environment,
    # remove the bellow comment to print it 
    #adhoc_agent.show_memory()

    # if you want to visualize the ad hoc agent estimation about the environment,
    # remove the bellow comment to print it 
    #adhoc_agent.smart_parameters['estimation'].show_estimation(env)

    typeestimation, parametersestimation, _ =\
        adhoc_agent.smart_parameters['estimation'].get_estimation(env)
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations'],
            'typeestimation':typeestimation,
            'parametersestimation':parametersestimation,
            'memoryusage':memory_usage,}
    log.write(data)

    if done:
        env.respawn_tasks()

env.close()
###
# THE END - That's all folks :)
###
