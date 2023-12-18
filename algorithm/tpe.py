import os, sys
import time
import copy
import random
import numpy as np
import subprocess

from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

random.seed(123)
iters = 300


LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'tpe_recordc1.log'

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

process = []
ts = []
b = 0
ts.append(0)

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

beginzero = time.time()

def get_objective_score(independent):
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
    l = time.time() - beginzero
    op_str = "iteration:{} speedup:{}".format(str(l), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    # process.append(-(time_o3_c /time_c))
    return -(time_o3_c /time_c)

    
    

if __name__ == '__main__':
    space = {}
    stats = []
    times = []
    time_end = 6000
    for option in all_flags:
        space[option] = hp.choice(option, [0, 1])
    begin = time.time()
    algo = partial(tpe.suggest, n_startup_jobs=1)
    while ts[-1] < time_end:
        best = fmin(get_objective_score, space, algo=algo, max_evals=iters)
        # stats.append(process)
        print(best)
        res = get_objective_score(best)
        b = time.time()
        print(res)
        ts.append(b-begin)
        ss = '{}: best-per {}, best-seq {}'.format(str(round(ts[-1])), str(res), str(best))
        write_log(str(ss),LOG_FILE)

   