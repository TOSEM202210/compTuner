import subprocess
import random
import time
import os


LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'ran_recordc1.log'


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

def generate_random_conf(x):
    """
    Generation 0-1 mapping for disable-enable options
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

if __name__ == "__main__":
    ts = []
    res = []  # speedup for different flag combinations
    seqs = [] # different flag combinations
    ts.append(0)
    time_end = 6000
    time_zero = time.time()
    while ts[-1] < time_end:
        """
        random generate
        """
        x = random.randint(0, 2 ** len(all_flags))
        seq = generate_random_conf(x)
        opt = ''
        for i in range(len(seq)):
            if seq[i]:
                opt = opt + all_flags[i] + ' '
            else:
                negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
                opt = opt + negated_flag_name + ' '
        seq.append(opt)

        """
        calculate the speedup time_o3 / time_opt
        """
        time_start = time.time()
        command = "/home/zmx/gcc8/bin/gcc -O2 " + opt + " -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
        execute_terminal_command(command)
        command2 = "/home/zmx/gcc8/bin/gcc -o a.out -O2 " + opt + " -lm *.o"
        execute_terminal_command(command2)
        command3 = "./a.out 1125000"
        execute_terminal_command(command3)
        cmd4 = 'rm -rf *.o *.I *.s a.out'
        execute_terminal_command(cmd4)

        time_end = time.time()  
        time_c = time_end - time_start   #time_opt

        time_o3 = time.time()
        command = "/home/zmx/gcc8/bin/gcc -O3 -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
        execute_terminal_command(command)
        command2 = "/home/zmx/gcc8/bin/gcc -o a.out -O3 -lm *.o"
        execute_terminal_command(command2)
        command3 = "./a.out 1125000"
        execute_terminal_command(command3)
        cmd4 = 'rm -rf *.o *.I *.s a.out'
        execute_terminal_command(cmd4)

        time_o3_end = time.time()  
        time_o3_c = time_o3_end - time_o3   #time_o3
        res.append(time_o3_c /time_c)
        ts.append(time.time()-time_zero)
        seqs.append(seq)

        best_per = max(res)
        best_seq = seqs[res.index(max(res))]
        ss = '{}: best-per {}, best-seq {}'.format(str(round(ts[-1])), str(best_per), str(best_seq))
        write_log(str(ss),LOG_FILE)