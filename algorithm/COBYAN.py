import os, time, subprocess, random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
import pandas as pd

all_flags = ['-O2', '-finline-functions', '-funswitch-loops', '-fpredictive-commoning', '-fgcse-after-reload', '-ftree-loop-vectorize', '-ftree-loop-distribution' , '-ftree-loop-distribute-patterns', 
             '-floop-interchange', '-floop-unroll-and-jam', '-fsplit-paths', '-ftree-slp-vectorize', '-fvect-cost-model', '-ftree-partial-pre', '-fpeel-loops', '-fipa-cp-clone']


LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'cobyan_recordc1.log'
ERROR_FILE = LOG_DIR + 'err.log'

def write_log(ss, file):
    log = open(file, 'a')
    log.write(ss + '\n')
    log.flush()
    log.close()

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
        if i == 0:
            if independent[i]:
                opt = opt + all_flags[i] + ' '
            else:
                opt = opt + ' '
        else:
            if independent[i]:
                opt = opt + all_flags[i] + ' '
            else:
                negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
                opt = opt + negated_flag_name + ' '

    time_start = time.time()
    command = "gcc " + opt + " -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
    execute_terminal_command(command)
    command2 = "gcc -o a.out " + opt + " -lm *.o"
    execute_terminal_command(command2)
    command3 = "./a.out 1125000"
    execute_terminal_command(command3)
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_end = time.time()  
    time_c = time_end - time_start   #time opt

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
    time_o3_c = time_o3_end - time_o3   #time o3
    print(time_o3_c /time_c)

    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)


def dataProcess():
    df = pd.read_csv('data.csv', header=0)
    df = df.drop(df.columns[0], axis=1)
    data = df.values.tolist()
    return(np.array(data))


if __name__ == "__main__":
    seqs = []
    speedups = []
    ts = []
    time_end = 6000
    begin = time.time()
    feature = [129.0, 37.0, 31.0, 0.0, 61.0, 26.0, 2.0, 26.0, 22.0, 11.0, 9.0, 0.0, 85.0, 4.0, 0.0, 139.0, 21.0, 479.0, 193.737, 20.0, 1.0, 31.0, 0.0, 407.0, 11.0, 136.0, 0.0, 102.0, 19.0, 0.0, 7.0, 100.0, 0.0, 16.0, 5.0, 13.0, 58.0, 36.0, 90.0, 59.0, 40.0, 991.0, 325.0, 9.0, 49.0, 2.0, 8.90872, 21.6932, 26.0, 5.0, 58.0, 3.0, 28.0]
    data = dataProcess()
    X = data[:, :-2]  # feature and flag
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kernel = Matern(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200, normalize_y=True)
    gp.fit(X_train, y_train)
    ts.append(time.time() - begin)
    while ts[-1] < time_end:
        x = random.randint(0, 2 ** len(all_flags))
        seq = generate_random_conf(x)
        seqs.append(seq)
        wait_for_predict = np.array(feature + seq)
        speedups.append(gp.predict(wait_for_predict.reshape(1, -1)))
        ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(max(speedups)), str(seqs[speedups.index(max(max(speedups)))]))
        write_log(ss, LOG_FILE)
        ts.append(time.time() - begin)
    independent = seqs[speedups.index(max(max(speedups)))]
    final = get_objective_score(independent, 100086)
    ts.append(time.time() - begin)
    ss = '{}: final-best {}, final-best-seq {}'.format(str(round(ts[-1])), str(final), str(independent))
    write_log(ss, LOG_FILE)