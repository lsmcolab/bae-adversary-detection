from chess import Board, WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN
import numpy as np

from gymnasium import spaces
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, StateSet

"""
    Load Scenario method
"""


"""
    Support classes
"""


"""
    Customising the Chess Env
"""
def end_condition(state):
    return state.state.is_game_over()

def chess_trasition(action, real_env):
    real_env.state.push(action)
    real_env.last_action = action
    if real_env.current_player == 'white':
        real_env.current_player = 'black'
        adhoc_agent = real_env.get_adhoc_agent()
        adhoc_agent.smart_parameters['adversary_last_action'] = action
        adhoc_agent.smart_parameters['adversary_last_observation'] = real_env.get_observation()
    else:
        real_env.current_player = 'white'
        adhoc_agent = real_env.get_adhoc_agent()
        adhoc_agent.smart_parameters['adversary_last_action'] = action
        adhoc_agent.smart_parameters['adversary_last_observation'] = real_env.get_observation()
    return real_env, {}

def reward(state, next_state):
    move = next_state.state.peek()
    piece = state.piece_at(move.to_square)
    
    if piece is not None:
        if piece.color == BLACK: # if white get black
            if piece.piece_type == PAWN:
                return 1.0
            elif piece.piece_type == KNIGHT:
                return 3.0
            elif piece.piece_type == BISHOP:
                return 3.0
            elif piece.piece_type == ROOK:
                return 5.0
            elif piece.piece_type == QUEEN:
                return 9.0
            else:
                raise IOError
        elif piece.color == WHITE:
            if piece.piece_type == PAWN:
                return -1.0
            elif piece.piece_type == KNIGHT:
                return -3.0
            elif piece.piece_type == BISHOP:
                return -3.0
            elif piece.piece_type == ROOK:
                return -5.0
            elif piece.piece_type == QUEEN:
                return -9.0
            else:
                raise IOError
        else:
            raise IOError
    return 0.0

def environment_transformation(copied_env):
    return copied_env


"""
    Chess Environments 
"""
class ChessEnv(AdhocReasoningEnv):

    def __init__(self, components, display=False):
        ###
        # Env Settings
        ###
        self.visibility = 'full'
        self.board = Board()
        self.last_action = None

        state_set = StateSet(spaces.Box(\
            low=0,high=np.inf,shape=(16,16), dtype=int), end_condition)
        transition_function = chess_trasition
        action_space = spaces.Discrete(self.board.legal_moves.count())
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(ChessEnv, self).__init__(state_set, \
            transition_function, action_space, reward_function, \
            observation_space, components)

        # Setting the inital state
        self.current_player = 'white'
        self.state = self.board.copy()
        self.state_set.initial_state = self.state.copy()
        self.state_set.initial_components = self.copy_components(self.components)

        ###
        # Setting graphical interface
        ###
        self.screen_size = (800,800) if display else (0,0)
        self.window = None
        self.display = display
        self.render_mode = "human"
        self.render_sleep = 0.5
        self.clock = None

    def show_state(self):
        for y in reversed(range(self.state.shape[1])):
            for x in range(self.state.shape[0]):
                print(self.state[x,y],end=' ')
            print()

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.levelbased.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = ChessEnv(components,self.display)
        copied_env.simulation = self.simulation
        copied_env.window = self.window
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state = self.state.copy()
        copied_env.episode = self.episode
        copied_env.state_set.initial_state = self.state_set.initial_state.copy()
        copied_env.current_player = self.current_player
        return copied_env

    def get_actions_list(self):
        return [m for m in self.state.legal_moves]

    def get_adhoc_agent(self):
        return self.components[self.current_player]

    def get_observation(self):
        state_fen = self.state.fen()
        return state_fen

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]
        
    def state_is_equal(self, state):
        return True

    def observation_is_equal(self, obs):
        state_fen = self.state.fen()
        return obs == state_fen

    def sample_state(self, agent):
        u_env = self.copy()
        return u_env

    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states
        
    def render(self, mode="human"): 
        
        if not self.display or self.simulation:
            return

        if not self.window: 
            try:
                from chess import svg
                from PyQt5.QtSvg import QSvgWidget
                from PyQt5.QtWidgets import QApplication, QWidget
                from gymnasium.error import DependencyNotInstalled
            except ImportError:
                raise DependencyNotInstalled(
                "PyQt5 is not installed, run `pip install PyQt5`"
            )
            class MainWindow(QWidget):
                def __init__(self,my_chess):
                    super().__init__()

                    self.setGeometry(100, 100, 650, 650)

                    self.chessboard = my_chess.board
                    self.widgetSvg = QSvgWidget(parent=self)
                    self.widgetSvg.setGeometry(10, 10, 600, 600)
                    self.chessboardSvg = svg.board(self.chessboard).encode("UTF-8")
                    self.widgetSvg.load(self.chessboardSvg)

                def paintEvent(self, event):
                    self.chessboardSvg = svg.board(self.chessboard).encode("UTF-8")
                    self.widgetSvg.load(self.chessboardSvg) 

                def update_board(self,my_chess):
                    self.chessboard = my_chess.state
                    self.widgetSvg = QSvgWidget(parent=self)
                    self.widgetSvg.setGeometry(10, 10, 600, 600)
                    self.chessboardSvg = svg.board(self.chessboard).encode("UTF-8")
                    self.widgetSvg.load(self.chessboardSvg)
                    
            self.app = QApplication([])
            self.window = MainWindow(self)

        self.window.update_board(self)
        self.window.show()
        self.app.exec()