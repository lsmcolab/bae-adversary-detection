import numpy as np
import torch
from collections import deque
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as  nn
import random

class QDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class MLP(torch.nn.Module):
    def __init__(self,in_shape):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_shape[0]+1,100)
        self.fc2 = nn.Linear(100,1)

    def forward(self,x):
        
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out


class ConvNet(torch.nn.Module):
    def __init__(self,in_shape):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2)
        (h,w) = (in_shape[0]+1,in_shape[1])
        new_shape = (32,(h-3)//2 + 1, (w-3)//2+1)
        self.fc1 = nn.Linear(new_shape[0]*new_shape[1]*new_shape[2],100)
        self.fc2 = nn.Linear(100,1)
    
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(out)
        out = torch.flatten(out,start_dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out



class DQN():
    def __init__(self,feature_shape,n_actions,lr=0.2,batch_size=16,buffer_size = 1000,discount_factor=0.95):
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialise_model(feature_shape)
        self.gamma = discount_factor
        self.epsilon = 1
        self.epsilon_decay = 1/500
        self.loss = []
        self.nactions = n_actions
        self.train_interval = 20
        
    def initialise_model(self,feature_shape):
        print("Creating Model ")
        if len(feature_shape) == 1:
            self.model = MLP(feature_shape)
        else:
            self.model = ConvNet(feature_shape)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        

    def feature(self,state,action):
        temp = state
        if len(temp.shape) == 1:
            ft = np.append(temp,np.array(action))
        else:
            ft = np.vstack((temp,np.array([action]*temp.shape[1])))       
            ft = np.expand_dims(ft,0) 
        return ft 

    def best_action(self,env):
       # TODO : Change Here
        max_reward =-100
        act = None
        list_act = [i for i in range(self.nactions)]
        for act_tar in list_act:
            feature = self.feature(env.get_feature(),act_tar)
            feature = np.expand_dims(feature,axis=0)
            feature_tensor = torch.from_numpy(feature)
            val = self.model(feature_tensor.float()).detach().numpy()[0].item()
            if np.isnan(val):
                val = -1000

            if (val > max_reward):
                max_reward=val
                act=act_tar
        if act is None:
            act = np.random.choice(env.get_actions_list(),1)[0]
        return act,max_reward


    
    def select_action(self,env):
        if self.train_interval == 0:
            self.replay_memory()
            self.train_interval = 20
        else:
            self.train_interval -= 1
        if np.random.random() > self.epsilon:
            return self.best_action(env)[0]
        else:
            return np.random.choice(env.get_actions_list())

    def add_memory(self,state,action,next_state,reward,done):
        self.memory.append((state,action,next_state,reward,done))

    def replay_memory(self):
        if len(self.memory) >= self.batch_size:
            mini_batch = random.sample(self.memory, self.batch_size)
            x = []
            y = []
            for (state,action, next_state, reward, done) in mini_batch:
                if (done):
                    target = reward
                else:
                    _, max_rew = self.best_action(next_state)
                    target = reward + self.gamma * max_rew
                x.append(self.feature(state.get_feature(),action))
                y.append(target)
            x = np.asarray(x)
            y = np.asarray(y)
            
            data = torch.utils.data.DataLoader(QDataset(x, y), batch_size=y.size, shuffle=True)
            self.loss.append(self.train(data))
            self.epsilon -= self.epsilon*self.epsilon_decay
    
    def train(self,train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            target = target.view(-1, 1, 1)

            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
def dqn_planning(env,agent,**kwargs):
    if 'dqn_model' not in agent.smart_parameters.keys():
        agent.smart_parameters['dqn_model'] = DQN(feature_shape = env.get_feature().shape,n_actions=len(env.get_actions_list()))
    
    action = agent.smart_parameters['dqn_model'].select_action(env)
    agent.smart_parameters['count'] = {}
    agent.smart_parameters['count']['nrollouts'] = 0
    agent.smart_parameters['count']['nsimulations'] = 0
    #action = 2
    return action, None
