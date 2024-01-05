import random as rd
import numpy as np
from copy import *
from src.envs.SmartFireBrigadeEnv import SmartFireBrigadeEnv
from src.envs.CaptureEnv import CaptureEnv
from src.envs.LevelForagingEnv import LevelForagingEnv

def type_parameter_estimation(env,adhoc_agent, type='uniform',**kwargs):
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork environment from the perspective of an ad hoc agent.

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    type : str, optional
        the estimation method's name to be used (default is uniform). The
        available options are: ['uniform', 'aga', 'abu', 'oeate', 'pomcp', 'bae']
    **kargs : 
        for additional key-arguments, double check the estimation methods 
        documentation `./src/reasoning/estimation/*.py` or look at
        `*_estimation` methods here

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 

    Examples
    --------
    Considering the execution of an example of Level-based Foraging Environment
    using the AGA as our estimation method

    >>> type_parameter_estimation(env, adhoc_agent, *args)
    (<src.envs.LevelForagingEnv.LevelForagingEnv object at 0x000001EFBC527EB0>, 
    ... <src.reasoning.bae.AGA object at 0x000001EFBC527FD0>
    """
    if type.upper() == 'UNIFORM':
        return uniform_estimation(env)
    elif type.upper() == 'AGA':
        return aga_estimation(env,adhoc_agent,**kwargs)
    elif type.upper() == 'ABU':
        return abu_estimation(env,adhoc_agent,**kwargs)
    elif type.upper() == 'OEATE':
        return oeate_estimation(env,adhoc_agent,**kwargs)
    elif type.upper() == 'OEATE_A':
        return oeate_a_estimation(env,adhoc_agent,**kwargs)
    elif type.upper() == 'POMCE':
        return pomcp_estimation(env,adhoc_agent,**kwargs)
    elif type.upper() == 'BAE':
        return bae_estimation(env,adhoc_agent,**kwargs)
    else:
        raise NotImplementedError


def aga_estimation(env, adhoc_agent,\
 template_types, parameters_minmax, grid_size=100, reward_factor=0.04, \
 step_size=0.01, decay_step=0.999, degree=2, univariate=True, **kwargs):
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork environment from the perspective of an ad hoc agent using AGA 
    estimation method [1].

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    template_types : list
        a list of strings with the types to consider in the estimation
    parameters_minmax : list
        a list of tuples with the range (min,max) for each parameter to
        consider in the estimation.
    grid_size : int, optional
        the size for the AGA estimation grid (default is 100)
    reward_factor :float, optional
        the value for AGA reward factoring (default is 0.04)
    step_size : float, optional
        the value for AGA estimation's step size (default is 0.01)
    decay_step : float, optional
        the value for AGA's decaying factor at each step (default is 0.999)
    degree : int, optional
        the value for AGA's estimation degree (default is 2)
    univariate : bool, optional
        a boolean variable to enable an univariate or multivariate estimation;
        if `True`, an univariate estimation will be performed (default is True)

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 

    
    See Also
    --------
    abu_estimation : Performs and returns the type and parameter estimation in 
    an ad hoc teamwork environment from the perspective of an ad hoc agent using
    ABU estimation method [1].
    
    Notes
    -----
    The AGA estimation algorithm is an alternative method to ABU application. 
    Both methods were proposed by Stefano V. Albrecht and Peter Stone (2017) [1].

    References
    ----------
    [1] Stefano V. Albrecht and Peter Stone. 2017. Reasoning about 
    Hypothetical Agent Behaviours and their Parameters. In Proceedings of the 
    16th Conference on Autonomous Agents and MultiAgent Systems (AAMAS '17). 
    International Foundation for Autonomous Agents and Multiagent Systems, 
    Richland, SC, 547–555.

    Examples
    --------
    Considering the execution of an example of Level-based Foraging Environment
    using the AGA as our estimation method

    >>> aga_estimation(env, adhoc_agent, template_types, parameters_minmax,\\
    >>>     grid_size=100, reward_factor=0.04, step_size=0.01,\\
    >>>     decay_step=0.999, degree=2, univariate=True)
    (<src.envs.LevelForagingEnv.LevelForagingEnv object at 0x000001EFBC527EB0>, 
    ... <src.reasoning.aga.AGA object at 0x000001EFBC527FD0>
    """
    #####
    # AGA INITIALISATION
    #####
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from src.reasoning.estmethods.aga import AGA
        adhoc_agent.smart_parameters['estimation'] = \
            AGA(env,template_types,parameters_minmax,grid_size,\
                reward_factor,step_size,decay_step,degree,univariate)
        
    #####    
    # AGA PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.\
        smart_parameters['estimation'].update(env)

    #####
    # AGA - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            if env.type_knowledge:
                selected_type = teammate.type
            else:
                selected_type = adhoc_agent.smart_parameters['estimation'].\
                    sample_type_for_agent(teammate)
                
            if env.parameter_knowledge:
                selected_parameter = teammate.get_parameters()
            else:
                selected_parameter = adhoc_agent.smart_parameters['estimation'].\
                    get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def abu_estimation(env, adhoc_agent, \
 template_types, parameters_minmax, grid_size=100, reward_factor=0.04, degree=2,
 **kwargs):
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork environment from the perspective of an ad hoc agent using AGA 
    estimation method [1].

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    template_types : list
        a list of strings with the types to consider in the estimation
    parameters_minmax : list
        a list of tuples with the range (min,max) for each parameter to
        consider in the estimation.
    grid_size : int, optional
        the size for the ABU estimation grid (default is 100)
    reward_factor :float, optional
        the value for ABU reward factoring (default is 0.04)
    degree : int, optional
        the value for ABU's estimation degree (default is 2)

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 

    
    See Also
    --------
    aga_estimation : Performs and returns the type and parameter estimation in 
    an ad hoc teamwork environment from the perspective of an ad hoc agent using
    AGA estimation method [1].
    
    Notes
    -----
    The ABU estimation algorithm is an alternative method to AGA application. 
    Both methods were proposed by Stefano V. Albrecht and Peter Stone (2017) [1].

    References
    ----------
    [1] Stefano V. Albrecht and Peter Stone. 2017. Reasoning about 
    Hypothetical Agent Behaviours and their Parameters. In Proceedings of the 
    16th Conference on Autonomous Agents and MultiAgent Systems (AAMAS '17). 
    International Foundation for Autonomous Agents and Multiagent Systems, 
    Richland, SC, 547–555.

    Examples
    --------
    Considering the execution of an example of Level-based Foraging Environment
    using the ABU as our estimation method

    >>> abu_estimation(env, adhoc_agent, template_types, parameters_minmax, \\
    >>>     grid_size=100, reward_factor=0.04, degree=2):
    (<src.envs.LevelForagingEnv.LevelForagingEnv object at 0x000001EFBC527EB0>, 
    ... <src.reasoning.abu.ABU object at 0x000001EFBC527FD0>
    """
    #####
    # ABU INITIALISATION
    #####
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from src.reasoning.estmethods.abu import ABU
        adhoc_agent.smart_parameters['estimation'] = \
            ABU(env,template_types,parameters_minmax,\
                grid_size,reward_factor,degree)
        
    #####    
    # ABU PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.\
        smart_parameters['estimation'].update(env)

    #####
    # ABU - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            if env.type_knowledge:
                selected_type = teammate.type
            else:
                selected_type = adhoc_agent.smart_parameters['estimation'].\
                    sample_type_for_agent(teammate)  
                
            if env.parameter_knowledge:
                selected_parameter = teammate.get_parameters()
            else:
                selected_parameter = adhoc_agent.smart_parameters['estimation'].\
                    get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def oeate_estimation(env, adhoc_agent,template_types, parameters_minmax,\
     N=100, xi=2, mr=0.2, d=100, normalise=np.mean, **kwargs):
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork environment from the perspective of an ad hoc agent using OEATE 
    estimation method [1].

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    template_types : list
        a list of strings with the types to consider in the estimation
    parameters_minmax : list
        a list of tuples with the range (min,max) for each parameter to
        consider in the estimation.
    N : int, optional
        the size of the estimation bag in OEATE method(default is 100) 
    xi : int, optional
        the threshold for removing estimators (default is 2) 
    mr : float, optional
        the mutation rate for estimators (default is 0.2) 
    d : int, optional
        size of the set of numbers for parameter estimation (default is 100) 
    normalise : function, optional
        normalisation function for parameter estimation (default is `np.mean`)

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 
    
    Notes
    -----
    The OEATE estimation algorithm is the final and improved version of OEATA 
    algorithm [2].

    References
    ----------
    [1] do Carmo Alves, M.A., Shafipour Yourdshahi, E., Varma, A. et al. 
    On-line estimators for ad-hoc task execution: learning types and parameters 
    of teammates for effective teamwork. Auton Agent Multi-Agent Syst 36, 45 
    (2022). https://doi.org/10.1007/s10458-022-09571-9.

    [2] Elnaz Shafipour Yourdshahi, Matheus Aparecido do Carmo Alves, Leandro 
    Soriano Marcolino, and Plamen Angelov. 2020. On-line Estimators for Ad-hoc 
    Task Allocation. In Proceedings of the 19th International Conference on 
    Autonomous Agents and MultiAgent Systems (AAMAS '20). International 
    Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 
    1999–2001.

    Examples
    --------
    Considering the execution of an example of Level-based Foraging Environment
    using the OEATE as our estimation method

    >>> oeate_estimation(env, adhoc_agent,template_types, parameters_minmax,\\
    >>>     N=100, xi=2, mr=0.2, d=100, normalise=np.mean)
    (<src.envs.LevelForagingEnv.LevelForagingEnv object at 0x000001EFBC527EB0>, 
    ... <src.reasoning.oeate.OEATE object at 0x000001EFBC527FD0>
    """
    #####
    # OEATE INITIALISATION
    #####
    # Initialising the oeata method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from src.reasoning.estmethods.oeate import OEATE
        adhoc_agent.smart_parameters['estimation'] = \
            OEATE(env,template_types,parameters_minmax,N,xi,mr,d,normalise)
        
    #####    
    # OEATE PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.\
        smart_parameters['estimation'].run(env)

    #####
    # OEATE - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            if env.type_knowledge:
                selected_type = teammate.type
            else:
                selected_type = adhoc_agent.smart_parameters['estimation'].\
                    sample_type_for_agent(teammate)
                
            if env.parameter_knowledge:
                selected_parameter = teammate.get_parameters()
            else:
                selected_parameter = adhoc_agent.smart_parameters['estimation'].\
                    get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def oeate_a_estimation(env, adhoc_agent,template_types, parameters_minmax,\
     N=100, xi=2, mr=0.2, d=100, normalise=np.mean, **kwargs):
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork environment from the perspective of an ad hoc agent using OEATE 
    estimation method [1].

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    template_types : list
        a list of strings with the types to consider in the estimation
    parameters_minmax : list
        a list of tuples with the range (min,max) for each parameter to
        consider in the estimation.
    N : int, optional
        the size of the estimation bag in OEATE method(default is 100) 
    xi : int, optional
        the threshold for removing estimators (default is 2) 
    mr : float, optional
        the mutation rate for estimators (default is 0.2) 
    d : int, optional
        size of the set of numbers for parameter estimation (default is 100) 
    normalise : function, optional
        normalisation function for parameter estimation (default is `np.mean`)

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 
    
    Notes
    -----
    The OEATE estimation algorithm is the final and improved version of OEATA 
    algorithm [2].

    References
    ----------
    [1] do Carmo Alves, M.A., Shafipour Yourdshahi, E., Varma, A. et al. 
    On-line estimators for ad-hoc task execution: learning types and parameters 
    of teammates for effective teamwork. Auton Agent Multi-Agent Syst 36, 45 
    (2022). https://doi.org/10.1007/s10458-022-09571-9.

    [2] Elnaz Shafipour Yourdshahi, Matheus Aparecido do Carmo Alves, Leandro 
    Soriano Marcolino, and Plamen Angelov. 2020. On-line Estimators for Ad-hoc 
    Task Allocation. In Proceedings of the 19th International Conference on 
    Autonomous Agents and MultiAgent Systems (AAMAS '20). International 
    Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 
    1999–2001.

    Examples
    --------
    Considering the execution of an example of Level-based Foraging Environment
    using the OEATE as our estimation method

    >>> oeate_estimation(env, adhoc_agent,template_types, parameters_minmax,\\
    >>>     N=100, xi=2, mr=0.2, d=100, normalise=np.mean)
    (<src.envs.LevelForagingEnv.LevelForagingEnv object at 0x000001EFBC527EB0>, 
    ... <src.reasoning.oeate.OEATE object at 0x000001EFBC527FD0>
    """
    #####
    # OEATE INITIALISATION
    #####
    # Initialising the oeata method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from src.reasoning.estmethods.oeate_a import OEATE_A
        adhoc_agent.smart_parameters['estimation'] = \
            OEATE_A(env,template_types,parameters_minmax,N,xi,mr,d,normalise)
        
    #####    
    # OEATE PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.\
        smart_parameters['estimation'].run(env)

    #####
    # OEATE - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            if env.type_knowledge:
                selected_type = teammate.type
            else:
                selected_type = adhoc_agent.smart_parameters['estimation'].\
                    sample_type_for_agent(teammate)
                
            if env.parameter_knowledge:
                selected_parameter = teammate.get_parameters()
            else:
                selected_parameter = adhoc_agent.smart_parameters['estimation'].\
                    get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def pomcp_estimation(env, adhoc_agent, \
 template_types, parameters_minmax, discount_factor=0.9, max_iter=100, \
 max_depth=10,min_particles=100, **kwargs):
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork environment from the perspective of an ad hoc agent using POMCE 
    estimation method [1].

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    template_types : list
        a list of strings with the types to consider in the estimation
    parameters_minmax : list
        a list of tuples with the range (min,max) for each parameter to
        consider in the estimation.
    discount_factor : float, optional
        the discount factor to be used while performing the estimation 
        (default is 0.9) 
    max_iter : 
        number of max iteration within POCMP algorithm search (default is 100)
    max_depth : 
        max depth of the POMCP tree while performing the estimation 
        (default is 10)
    min_particles :
        minimum number of particles to be generated in the black-box procedure 
        while performing the estimation ( default is 100)

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 

    References
    ----------
    [1] do Carmo Alves, M.A., Shafipour Yourdshahi, E., Varma, A. et al. 
    On-line estimators for ad-hoc task execution: learning types and parameters 
    of teammates for effective teamwork. Auton Agent Multi-Agent Syst 36, 45 
    (2022). https://doi.org/10.1007/s10458-022-09571-9.

    Examples
    --------
    Considering the execution of an example of Level-based Foraging Environment
    using the POMCPE as our estimation method

    >>> pomcp_estimation(env, adhoc_agent, template_types, parameters_minmax, \\
    >>>     discount_factor=0.9, max_iter=100, max_depth=10,min_particles=100)
    (<src.envs.LevelForagingEnv.LevelForagingEnv object at 0x000001EFBC527EB0>, 
    ... <src.reasoning.pomce.POMCPE object at 0x000001EFBC527FD0>
    """
    #####
    # POMCE INITIALISATION
    ##### discount_factor=0.9,max_iter=10,max_depth=10,min_particles=100
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from src.reasoning.estmethods.pomce import POMCE
        adhoc_agent.smart_parameters['estimation'] = \
            POMCE(env,template_types,parameters_minmax,\
                discount_factor,max_iter,max_depth,min_particles)
        
    #####    
    # POMCE PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = \
        adhoc_agent.smart_parameters['estimation'].update(env)

    #####
    # POMCP - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].\
                sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].\
                get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)
  
    return env, adhoc_agent.smart_parameters['estimation']

def bae_estimation(env, adhoc_agent, \
    template_types, parameters_minmax, **kwargs):
    
    """
    Performs and returns the type and parameter estimation in an ad hoc
    teamwork reasoning environment from the perspective of an ad hoc agent
    using BAE estimation method.

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 
    adhoc_agent : AdhocAgent object or extensions
        the ad hoc agent in the environment
    template_types : list
        a list of strings with the types to consider in the estimation
    parameters_minmax : list
        a list of tuples with the range (min,max) for each parameter to
        consider in the estimation.
    ...

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    estimation : estimation object
        the current estimation for the environment 
    
    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    #####
    # BAE INITIALISATION
    #####
    multi_tree = kwargs.get('multi_tree')
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from src.reasoning.estmethods.bae import BAE
        adhoc_agent.smart_parameters['estimation'] = BAE(env, template_types, parameters_minmax)
        
    #####    
    # BAE PROCESS
    #####
    adversary_actions_prob_distribution = kwargs.get('adversary_actions_prob_distribution')
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.\
        smart_parameters['estimation'].update(env,adversary_actions_prob_distribution,multi_tree)

    #####
    # BAE - SET ESTIMATION
    #####
    env = adhoc_agent.smart_parameters['estimation'].sample_state(env)
    return env, adhoc_agent.smart_parameters['estimation']

def uniform_estimation(env, **kwargs):
    """
    Performs and returns an uniform type and parameter estimation in an ad hoc
    teamwork environment. The uniform estimation algorithm depends on the 
    environment of application.

    Parameters
    ----------
    env : AdhocReasoningEnv object or extensions
        the problem's environment 

    Returns
    -------
    env : AdhocReasoningEnv object or extensions
        the modified environment considering the current estimation
    
    See Also
    --------
    level_foraging_uniform_estimation : 
        Performs and returns an uniform type and parameter estimation for a
        Level-based Foraging environment.
    capture_uniform_estimation : 
        Performs and returns an uniform type and parameter estimation for a
        Capture-the-Prey environment.
    smartfirebrigade_uniform_estimation :
        Performs and returns an uniform type and parameter estimation for a
        Smart Fire-brigade environment.
    truco_uniform_estimation :
        Performs and returns an uniform type and parameter estimation for a
        Truco environment.
    """
    if isinstance(env,LevelForagingEnv):
        return level_foraging_uniform_estimation(env)
    elif isinstance(env,CaptureEnv):
        return capture_uniform_estimation(env)
    elif isinstance(env,SmartFireBrigadeEnv):
        return smartfirebrigade_uniform_estimation(env)
    else:
        raise NotImplemented

def level_foraging_uniform_estimation(env, template_types=['l1','l2','l3']):
    """
    Performs and returns an uniform type and parameter estimation for a
    Level-based Foraging environment.

    Parameters
    ----------
    env : LevelForagingEnv object
        target environment object for estimation
    template_types : list
        a list of strings with the types to consider in the estimation

    Returns
    -------
    env : LevelForagingEnv object
        the modified environment considering the current estimation
    """
    adhoc_agent = env.get_adhoc_agent()
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            teammate.type = rd.sample(template_types,1)[0]
            teammate.set_parameters(np.random.uniform(0.5,1,3))
    return env

def capture_uniform_estimation(env, template_types=['c1','c2','c3']):
    """
    Performs and returns an uniform type and parameter estimation for a
    Capture-the-Prey environment.

    Parameters
    ----------
    env : CaptureEnv object
        target environment object for estimation
    template_types : list
        a list of strings with the types to consider in the estimation

    Returns
    -------
    env : CaptureEnv object
        the modified environment considering the current estimation
    """
    adhoc_agent = env.get_adhoc_agent()
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            teammate.type = rd.sample(template_types,1)[0]
            teammate.set_parameters(np.random.uniform(0.5,1,2))
    return env

def smartfirebrigade_uniform_estimation(env):
    # TODO: Implement an uniform estimation for the SFB environment
    return env    

def truco_uniform_estimation(env):
    """
    Performs and returns an uniform type and parameter estimation for a
    Truco environment.

    Parameters
    ----------
    env : TrucoEnv object
        target environment object for estimation

    Returns
    -------
    env : TrucoEnv object
        the modified environment considering the current estimation
    """
    if env.visibility == 'partial':
        for player in env.components['player']:
            if player != env.components['player'][env.current_player]:
                player.hand = []
                while len(player.hand) < 3:
                    player.hand.append(env.components['cards in game'].pop(0))
                player.type = rd.sample(['t1', 't2', 't3'], 1)[0]

    env.visibility = 'full'
    return env
