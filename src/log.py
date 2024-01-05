import inspect
import os
import datetime
import warnings
import sys

######
# EXCEPTION METHODS
######
# returns the current line number in our program.
def lineno():
    return inspect.currentframe().f_back.f_lineno

######
# WRITE METHODS
######
class LogFile:

    def __init__(self,env,scenario_id,method,exp_num,*args):
        # creating the path
        if(not os.path.isdir("results")):
            os.mkdir("results")
        self.path = "./results/"

        self.env = env
        self.header = args

        # defining filename
        self.start_time = datetime.datetime.now()
        self.filename = str(method)+'_'+str(env)+str(scenario_id)+'_'+str(exp_num)+'.csv'

        # creating the result file
        self.write_header()

    def write_header(self):
        with open(self.path+self.filename, 'w') as logfile:
            for header in self.header[0]:
                logfile.write(str(header)+";")
            logfile.write('\n')

    def write(self,*args):
        with open(self.path+self.filename, 'a') as logfile:
            if(not len(args) ==len(self.header)):
                warnings.warn("Initialisation and writing have different sizes .")

            for key in args[0]:
                logfile.write(str(args[0][key])+";")

            logfile.write('\n')

class EstimationLogFile:

    def __init__(self,env,scenario_id,method,estimation_method, exp_num,\
     template_types, parameters_minmax):
        # creating the path
        if(not os.path.isdir("results")):
            os.mkdir("results")
        self.path = "./results/"

        self.env = env
        self.header =  ['Iteration', 'Reward', 'Time to reason', 'N Rollouts',\
         'N Simulations', 'TypeEstimation', 'ParametersEstimation', 'Memory Usage']

        # defining filename
        self.start_time = datetime.datetime.now()
        self.filename = str(method)+'_'+estimation_method+'_'+str(env)+str(scenario_id)+'_'+str(exp_num)+'.csv'

        # creating the result file
        self.write_header()

    def write_header(self):
        with open(self.path+self.filename, 'w') as logfile:
            for h in self.header:
                logfile.write(str(h)+";")
            logfile.write('\n')

    def write(self,*args):
        with open(self.path+self.filename, 'a') as logfile:
            if(not len(args) ==len(self.header)):
                warnings.warn("Initialisation and writing have different sizes .")

            for key in args[0]:
                logfile.write(str(args[0][key])+";")

            logfile.write('\n')

class BashLogFile:

    def __init__(self,file_name=""):
        # creating the path
        if(not os.path.isdir("./bashlog")):
            os.mkdir("./bashlog")
        self.path = "./bashlog/"

        # defining filename
        self.start_time = datetime.datetime.now()
        if(file_name ==""):
            self.filename = self.start_time.strftime("%d-%m-%Y_%Hh%Mm%Ss")+ ".csv"
        else:
            self.filename = file_name
        
        # saving original stderr
        self.original_stderr = sys.stderr

        # creating the log files
        ofile = open(self.path+'OUTPUT_'+self.filename,'w')
        ofile.close()

        efile = open(self.path+'ERROR_'+self.filename,'w')
        efile.close()

    def redirect_stderr(self):
        file = open(self.path+'ERROR_'+self.filename,'a')
        sys.stderr = file
    
    def reset_stderr(self):
        sys.stderr = self.original_stderr

    def write(self,log):
        print(log)
        with open(self.path+'OUTPUT_'+self.filename, 'a') as blfile:
            blfile.write(log+'\n')

######
# TRANING LOG METHODS
######
import pickle
class TrainedModel:

    def __init__(self,model_name):
        self.path = './src/reasoning/trainedmodels/'+model_name+'.pickle'
        if os.path.exists(self.path):
            with open(self.path,'rb') as model_file:
                self.qtable = pickle.load(model_file)
        else:
            self.qtable = None
        
    def update(self,position,actions,qtable):
        hash_key = self.get_hash_key(position)

        # updating model value
        if self.qtable is not None:
            if  hash_key not in self.qtable:
                self.qtable[hash_key] = {}

            for a in actions:
                if str(a) not in self.qtable[hash_key]:
                    self.qtable[hash_key][str(a)] = {}
                    self.qtable[hash_key][str(a)]['qvalue'] = 0.0
                    self.qtable[hash_key][str(a)]['updates'] = 1
                else:
                    balance = self.qtable[hash_key][str(a)]['updates']/ \
                                (self.qtable[hash_key][str(a)]['updates']+1)
                    
                    self.qtable[hash_key][str(a)]['qvalue'] = \
                        (balance)*(self.qtable[hash_key][str(a)]['qvalue']) + \
                            (1-balance)*(qtable[str(a)]['qvalue'])
                    
                    self.qtable[hash_key][str(a)]['updates'] = \
                        int(self.qtable[hash_key][str(a)]['updates']) + 1
        else:
            self.qtable = {hash_key:{}}
            for a in actions:
                self.qtable[hash_key][str(a)] = {}
                
                self.qtable[hash_key][str(a)]['qvalue'] = qtable[str(a)]['qvalue']
                
                self.qtable[hash_key][str(a)]['updates'] = 1
        
        # saving
        with open(self.path,'wb') as model_file:
            pickle.dump(self.qtable, model_file)

    def get_hash_key(self,position):
        return '('+str(position[0])+','+str(position[1])+')'