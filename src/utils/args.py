from argparse import ArgumentParser, ArgumentTypeError

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_args():
    # 1. Reading the environment settings
    parser = ArgumentParser()
    # default args
    parser.add_argument('--atype', dest='atype', default='pomcp', type=str)
    parser.add_argument('--exp_num', dest='exp_num', default=0, type=int)
    parser.add_argument('--id', dest='id', default=0, type=int)
    parser.add_argument('--mode', dest='mode', default='default', type=str)
    args = parser.parse_args()
    return args

def get_estimation_args():
    # 1. Reading the environment settings
    parser = ArgumentParser()
    # default args
    parser.add_argument('--atype', dest='atype', default='mcts', type=str)
    parser.add_argument('--exp_num', dest='exp_num', default=0, type=int)
    parser.add_argument('--id', dest='id', default=6, type=int)
    parser.add_argument('--mode', dest='mode', default='aga', type=str)
    args = parser.parse_args()
    return args
