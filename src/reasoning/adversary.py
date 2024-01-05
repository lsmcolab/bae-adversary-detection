from ..log import TrainedModel
import numpy as np
import random as rd

def adversary_planning(env, agent):
    if env.name is not None:
        if env.name == 'LevelForagingEnv5' or env.name == 'LevelForagingEnv6' or \
         env.name == 'LevelForagingEnv7' or env.name == 'LevelForagingEnv8':
            # uploading the model
            model = TrainedModel('AdversaryModel_'+env.name)
            
            # picking actions probabilities for the current state
            state_key = env.get_state_str_representation()
            if state_key in model.qtable:
                qtable = model.qtable[state_key]
            else:
                a = rd.choice(env.actions)
                return a, None
            
            norm = sum([qtable[str(a)]['qvalue'] for a in env.actions])
            actions_prob = [1/len(env.actions) for a in env.actions] \
                if norm == 0.0 else [qtable[str(a)]['qvalue']/norm for a in env.actions]
            # sampling an action based on the model actions probability
            unif = rd.uniform(0,1)
            cum_action_prob = np.cumsum(actions_prob)
            for a in range(len(cum_action_prob)):
                if unif < cum_action_prob[a]:
                    return a, None
            return None, None
        else:
            raise FileNotFoundError('No Adversarial Model found.')
    else:
        raise NotImplemented