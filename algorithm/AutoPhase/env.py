import datetime, glob, shutil, gym, math, pickle, sys, subprocess, os, time
import numpy as np
from gym.spaces import Discrete, Box, Tuple
from IPython import embed
import utils
def execute_terminal_command(command):
    """
    Execute the compiler and run command
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("命令执行成功！")
            if result.stdout:
                print("命令输出：")
                print(result.stdout)
        else:
            print("命令执行失败。")
            if result.stderr:
                print("错误输出：")
                print(result.stderr)
    except Exception as e:
        print("执行命令时出现错误：", str(e))

def get_bc_files(folder_path):
    """
    Obtain the .bc files
    """
    bc_files = [file for file in os.listdir(folder_path) if file.endswith(".bc")]
    return bc_files

def get_optbc_files(folder_path):
    """
    Obtain the .opt.bc files
    """
    bc_files = [file for file in os.listdir(folder_path) if file.endswith(".opt.bc")]
    return bc_files

class ENV(gym.Env):
    """
    environment
    """
    def __init__(self, env_config):
        self.pass_len = 274 # pass_len (int): number of passes for the program 
        # self.feat_len = 56

        self.shrink = env_config.get('shrink', False) 
        if self.shrink:
            self.eff_pass_indices = [1,7,11,12,14,15,23,24,26,28,30,31,32,33,38,43]
            self.pass_len = len(self.eff_pass_indices) 
            # self.eff_feat_indices = [5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 31, 32, 33, 34, 36, 37, 38, 40, 42, 46, 49, 52, 55] 
            # self.feat_len = len(self.eff_feat_indices) 
        
        # control features' observations
        self.binary_obs = env_config.get('binary_obs', False)
        self.norm_obs = env_config.get('normalize', False) 
        self.orig_norm_obs = env_config.get('orig_and_normalize', False)

        # use act_hist as feature
        self.feature_type = env_config.get('feature_type', 'act_hist') 
        # init act_hist
        self.act_hist = [0] * self.pass_len
        # ohter feature
        self.bandit = self.feature_type == 'bandit'
        self.action_pgm = self.feature_type == 'act_pgm'
        self.action_meaning = [-1,0,1]
        self.reset_actions = [int(self.pass_len // 2)] * self.pass_len

        self.max_episode_steps=274

        if self.action_pgm:
             self.action_space=Tuple([Discrete(len(self.action_meaning))]*self.pass_len)
        elif self.bandit:
            self.action_space = Tuple([Discrete(self.pass_len)]*12)
        else:
            self.action_space = Discrete(self.pass_len)

        
        #init boundary
        if self.feature_type == 'act_hist' or self.feature_type == "act_hist_sparse":
            self.observation_space = Box(0.0,274,shape=(self.pass_len,),dtype = np.int32)

        else:
            raise

        #result
        self.prev_time = 200 
        self.O0_time = 200
        self.prev_obs = None
        self.min_time = 200
        self.verbose = env_config.get('verbose',False)
        self.log_obs_reward = env_config.get('log_obs_reward',False)

        pgm = env_config['pgm']
        pgm_dir = env_config.get('pgm_dir', None)
        pgm_files = env_config.get('pgm_files', None)
        run_dir = env_config.get('run_dir', None)
        self.delete_run_dir = env_config.get('delete_run_dir', True) # delete_run_dir (bool):  delete_run_dir is a Boolean that should be set to True if we want to close the log_file. It should be set to False otherwise.
        self.init_with_passes = env_config.get('init_with_passes', False) # init_with_passes (bool): init_with_passes is a Boolean that should be set to True when we want to reset the episodes but want to start with specific passes. Basically, it reinitialize the Rl training with specific predetermined passes. It should be set to False otherwise.
        self.log_results = env_config.get('log_results', False)
        
        if run_dir: # run_dir (str): run_diris the path of the running directory
            self.run_dir = run_dir+'_p'+str(os.getpid())
        else:
            currentDT = datetime.datetime.now()
            self.run_dir ="run-"+currentDT.strftime("%Y-%m-%d-%H-%M-%S-%f")+'_p'+str(os.getpid())

        if self.log_results:
            self.log_file = open(self.run_dir+".log","w")
        
        cwd = os.getcwd()
        self.run_dir = os.path.join(cwd, self.run_dir)

        if os.path.isdir(self.run_dir):
            shutil.rmtree(self.run_dir, ignore_errors=True)
        if pgm_dir:
            shutil.copytree(pgm_dir, self.run_dir)
        if pgm_files:
            os.makedirs(self.run_dir)
        for f in pgm_files:
          shutil.copy(f, self.run_dir)
        
        #passes reinitialize the training
        self.pre_passes_str= "-prune-eh -functionattrs -ipsccp -globalopt -mem2reg -deadargelim -sroa -early-cse -lazy-block-freq -instcombine -loop-simplify"
        self.pre_passes = utils.passes2indice(self.pre_passes_str) 

        self.passes = []
        self.best_passes = [] 

        self.pgm = pgm # pgm_name (str): pgm_name is the file name of the program we are optimizing (which is written in C programming language)
        self.pgm_name = pgm.replace('.c','')
        self.bc = self.pgm_name + '.prelto.2.bc'
        self.original_obs = []
    
    def __del__(self):
        if self.delete_run_dir:
            if self.log_results:
                self.log_file.close()
        if os.path.isdir(self.run_dir):
            shutil.rmtree(self.run_dir)

    def get_O3_rewards(self, diff=True, sim=False):
        """
        Returns:
        - O3 time
        """
        begin = time.time()
        folder_path = "/home/zmx/"
        cod1 = '/home/zmx/llvm9.0.0/bin/clang-9  -O0 -emit-llvm -c +' + self.run_dir + '/*.c'
        execute_terminal_command(cod1)
        bc_files = get_bc_files(folder_path)
        for bc_file in bc_files:
            cod2 = '/home/zmx/llvm9.0.0/bin/opt -S -O3 ' + bc_file + ' -o ' + bc_file.split('.')[0]+'.opt.bc'
            execute_terminal_command(cod2)
        opt_bc_files = get_optbc_files(folder_path)
        for bc_file in opt_bc_files:  
            cod3 = '/home/zmx/llvm9.0.0/bin/llc -O0 -filetype=obj ' + bc_file
            execute_terminal_command(cod3)
        cmd3 = '/home/zmx/llvm9.0.0/bin/clang-9 -o a.out -O0 -lm *.o'
        execute_terminal_command(cmd3)
        command3 = './a.out '
        execute_terminal_command(command3)
        cmd4 = 'rm -f *.o *.I *.s out a.out *.a *.s *.i *.bc *.opt.bc'
        execute_terminal_command(cmd4)

        return -(time.time() - begin)
    
    def get_time(self, passes, sim=False):
        """
        Args:
        passes selected
        Returns:
        - passes time
        """
        if self.shrink:
            actual_passes = [self.eff_pass_indices[index] for index in passes]
        else:
            actual_passes =  passes
        res,_ = utils.get_time(self.run_dir,actual_passes,sim=sim)
        return res
        
    def get_opt_rewards(self, diff=True, sim=False):
        """
        Returns:
        - reward
        """
        if self.shrink:
            actual_passes = [self.eff_pass_indices[index] for index in self.passes]
        else:
            actual_passes =  self.passes

        res, done = utils.get_time(self.run_dir,actual_passes,sim=sim)
        if res < self.min_time:
            self.min_time = res
            self.best_passes = actual_passes
        if (diff):
            rew = self.prev_time - res
            self.prev_time = res
        else:
            rew = - res
        return rew, True
    

    def print_info(self,message, end = '\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    def reset(self, init=None, get_obs=True, get_rew=False, ret=True, sim=False):
        """
        # reset() resets passes to []
        # reset(init=[1,2,3]) resets passes to [1,2,3]
        get_obs (bool, optional): get_obs is a Boolean that is set to True when we decide to get the list of observation features after we reset. 
        It should be set to False otherwise. Defaults to True.
        get_rew (bool, optional): get_rew is a Boolean that is set to True when we decide to get the reward after we reset. It should be set to False otherwise. 
        Defaults to False.
        ret (bool, optional): ret is a Boolean that is set to True when we decide to get the reward or the list of observation features after we reset. 
        It should be set to False otherwise. Defaults to True.
        sim (bool, optional): sim should be True if you want the arguments used to launch the process to be “make clean p v -s”, or sim should be False if you want 
        the argument used to launch the process to be "make clean accelerationCycle -s". Defaults to False. Defaults to False.
        """
        self.passes = []
        if self.feature_type == 'act_pgm':
            self.passes = self.reset_actions
        if self.init_with_passes:
            self.passes.extend(self.pre_passes)
        if init:
            self.passes.extend(init)  
        self.prev_time = self.get_time(self.passes)  
        self.O0_time = self.prev_time
        if(self.verbose):
            self.print_info("program: {} -- ".format(self.pgm_name)+" reset time: {}".format(self.prev_time))
        if ret:
            if get_rew:
                reward, _ = self.get_opt_rewards(sim=sim)
            obs = []
            if get_obs:
                if self.feature_type == 'act_hist' or self.feature_type == "act_hist_sparse":
                    self.act_hist = [0] * self.pass_len
                    obs = self.act_hist
                elif self.bandit:
                    obs = [1] * 12
                else:
                    raise
                obs = np.array(obs)
                if self.log_results:
                    self.prev_obs = obs
            if get_rew and not get_obs:
                return reward
            if get_obs and not get_rew:
                return obs
            if get_obs and get_rew:
                return (obs, reward)
        else:
            return 0

    def step(self, action, get_obs=True):
        """
        action (list): action is a list of the passes that the RL decide to apply as the next move after having analyzed its input values (observation features list and reward from the cycle count).
        get_obs (bool, optional): get_obs is a Boolean that is set to True when we decide to get the list of observation features during each step. It should be set to False otherwise. Defaults to True.
        Returns:
        Returns a tuple of observation features list, reward from time, the Boolean done from get_reward, and info (the dictionary initialized at the beginning of the function).
        """
        info = {}
        if self.bandit:
            self.passes = action
        else:
            self.passes.append(action)
        if self.feature_type == "act_hist_sparse" and len(self.passes) <  self.max_episode_steps:
            reward = 0
            done = False
        else:
            reward, done = self.get_opt_rewards()
        obs = []
        if(self.verbose):
            self.print_info("program: {} --".format(self.pgm_name) + "passes: {}".format(self.passes))
            self.print_info("reward: {} -- done: {}".format(reward, done))
            self.print_info("min_time: {} -- best_passes: {}".format(self.min_time, self.best_passes))
            self.print_info("act_hist: {}".format(self.act_hist))

        if get_obs:
            if self.feature_type == 'act_hist' or self.feature_type == "act_hist_sparse":
                self.act_hist[action] += 1
                obs = self.act_hist

            elif self.bandit:
                obs = self.passes
        obs = np.array(obs)
        if self.log_results:
            if self.feature_type == "act_hist_sparse" and (len(self.passes) == self.max_episode_steps):
                print("{}|{}|{}|{}|{}|{}|{}\n".format(self.prev_obs, action, reward, self.prev_time, self.min_time, self.passes, self.best_passes))
                self.log_file.write("{}|{}|{}|{}|{}|{}|{}\n".format(self.prev_obs, action, reward, self.prev_time, self.min_time, self.passes, self.best_passes))
            else:
                self.log_file.write("{}|{}|{}|{}|{}|{}|{}\n".format(self.prev_obs, action, reward, self.prev_time, self.min_time, self.passes, self.best_passes))
            self.log_file.flush()

        self.prev_obs = obs
        return (obs, reward, done, info)
    

    def render(self):
        print("pass: {}".format(self.passes))
        print("prev_time: {}".format(self.prev_time))

