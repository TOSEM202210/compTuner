import subprocess
import random
import time
import os

LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'ga_recordc1.log'


def write_log(ss, file):
    log = open(file, 'a')
    log.write(ss + '\n')
    log.flush()
    log.close()

all_flags = ['-falign-labels', '-fauto-inc-dec', '-fbranch-count-reg', '-fcaller-saves', '-fcode-hoisting', '-fcombine-stack-adjustments', '-fcompare-elim',
             '-fcprop-registers', '-fcrossjumping', '-fcse-follow-jumps', '-fdefer-pop', '-fdevirtualize', '-fdevirtualize-speculatively', '-fearly-inlining', 
             '-fexpensive-optimizations', '-fforward-propagate', '-ffunction-cse', '-fgcse', '-fguess-branch-probability', '-fhoist-adjacent-loads', '-fif-conversion', 
             '-fif-conversion2', '-findirect-inlining', '-finline-atomics', '-finline-functions-called-once', '-finline-small-functions', '-fipa-bit-cp', '-fipa-cp', 
             '-fipa-icf', '-fipa-icf-functions', '-fipa-icf-variables', '-fipa-profile', '-fipa-pure-const', '-fipa-ra', '-fipa-reference', '-fipa-sra', '-fipa-vrp', 
             '-fira-share-save-slots', '-fisolate-erroneous-paths-dereference', '-fjump-tables', '-flra-remat', '-fmove-loop-invariants', '-fomit-frame-pointer', 
             '-foptimize-sibling-calls', '-foptimize-strlen', '-fpartial-inlining', '-fpeephole2', '-fprefetch-loop-arrays', '-freg-struct-return', '-freorder-blocks', 
             '-freorder-blocks-and-partition', '-freorder-functions', '-frerun-cse-after-loop', '-fsched-critical-path-heuristic', '-fsched-group-heuristic', '-fsched-last-insn-heuristic',
             '-fsched-spec', '-fsched-stalled-insns-dep', '-fschedule-insns2', '-fshrink-wrap', '-fsigned-zeros', '-fsplit-wide-types', '-fssa-phiopt', '-fstore-merging', 
             '-fstrict-aliasing', '-fthread-jumps', '-ftrapping-math', '-ftree-bit-ccp', '-ftree-builtin-call-dce', '-ftree-ccp', '-ftree-ch', '-ftree-coalesce-vars', 
             '-ftree-copy-prop', '-ftree-dce', '-ftree-dominator-opts', '-ftree-dse', '-ftree-fre', '-ftree-loop-im', '-ftree-loop-optimize', '-ftree-pre', '-ftree-pta', 
             '-ftree-scev-cprop', '-ftree-sink', '-ftree-slsr', '-ftree-sra', '-ftree-switch-conversion', '-ftree-tail-merge', '-ftree-ter', '-ftree-vrp', '-fvar-tracking', '-fweb']


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

def get_objective_score(independent, k_iter):
    """
    Obtain the speedup
    """
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt = opt + negated_flag_name + ' '

    print(opt) 
    time_start = time.time() 
    command = "gcc -O2 " + opt + " -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
    execute_terminal_command(command)
    command2 = "gcc -o a.out -O2 " + opt + " -lm *.o"
    execute_terminal_command(command2)
    command3 = "./a.out 1125000"
    execute_terminal_command(command3)
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)

    time_end = time.time()  
    time_c = time_end - time_start   #time_opt

    time_o3 = time.time()
    command = "gcc -O3 -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
    execute_terminal_command(command)
    command2 = "gcc -o a.out -O3 -lm *.o"
    execute_terminal_command(command2)
    command3 = "./a.out 1125000"
    execute_terminal_command(command3)
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)

    time_o3_end = time.time()  
    time_o3_c = time_o3_end - time_o3   #time_o3
    print(time_o3_c /time_c)

    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)


def generate_random_conf(x):
    """
    Generate 0-1 mapping for disable-enable options
    """

    comb = bin(x).replace('0b', '')
    comb = '0' * (len(all_flags) - len(comb)) + comb
    conf = []
    for k, s in enumerate(comb):
        if s == '1':
            conf.append(1)
        else:
            conf.append(0)
    return conf


class GA:
    def __init__(self, options, get_objective_score):
        self.options  = options
        self.get_objective_score = get_objective_score
        self.begin = time.time()
        geneinfo = []
        for i in range(4):
            x = random.randint(0, 2 ** len(all_flags))
            geneinfo.append(generate_random_conf(x))
        fitness = []
        self.dep = []

        #initial combinations
        for x in geneinfo:
            tmp = self.get_objective_score(x,100086)
            fitness.append(-1.0 / tmp)
        
        #sort by speedup
        self.pop = [(x, fitness[i]) for i, x in enumerate(geneinfo)]
        self.pop = sorted(self.pop, key=lambda x:x[1])

        self.best = self.selectBest(self.pop)
        self.dep.append(1.0/self.best[1])
        self.end = time.time() - self.begin

    def selectBest(self, pop):
        """
        select best flag combinations
        """
        return pop[0]
        
    def selection(self, inds, k):
        """
        select preserved flag combinations
        """
        s_inds = sorted(inds, key=lambda x:x[1])
        return s_inds[:int(k)]

    def crossoperate(self, offspring):
        """
        cross flag combinations
        """
        dim = len(self.options)
        geninfo1 = offspring[0][0]
        geninfo2 = offspring[1][0]
        pos = random.randrange(1, dim)

        newoff = []
        for i in range(dim):
            if i>=pos:
                newoff.append(geninfo2[i])
            else:
                newoff.append(geninfo1[i])
        return newoff

    def mutation(self, crossoff):
        """
        mutate flag combinations
        """
        dim = len(self.options)
        pos = random.randrange(1, dim)
        crossoff[pos] = 1 - crossoff[pos]
        return crossoff

    def GA_main(self):
        ts = []
        time_end = 6000
        ts.append(self.end)
        time_zero = time.time()
        while ts[-1] < time_end:
            selectpop = self.selection(self.pop, 0.5 * 2)
            nextoff = []
            while len(nextoff) != 2:
                offspring = [random.choice(selectpop) for i in range(2)]
                crossoff = self.crossoperate(offspring)
                muteoff = self.mutation(crossoff)
                fit_muteoff = self.get_objective_score(muteoff,k_iter=100086)
                nextoff.append((muteoff, -1.0 / fit_muteoff))
            self.pop = nextoff       
            self.pop = sorted(self.pop, key=lambda x:x[1])
            self.best = self.selectBest(self.pop)
            ts.append(time.time() - time_zero + self.end)
            self.dep.append(1.0/self.best[1])
            ss = '{}: best-per {}, best-seq {}'.format(str(round(ts[-1])), str(self.best[1]), str(self.best[0]))
            write_log(str(ss),LOG_FILE)
        return self.dep
    
if __name__ == "__main__":
    ga = GA(all_flags,get_objective_score=get_objective_score)
    ga.GA_main()
