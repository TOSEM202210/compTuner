import os,subprocess,time
opt_passes_str = '-tti -tbaa -scoped-noalias -assumption-cache-tracker -targetlibinfo -verify -ee-instrument -simplifycfg -domtree -sroa -early-cse -lower-expect -targetlibinfo -tti -tbaa -scoped-noalias -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -domtree -callsite-splitting -ipsccp -called-value-propagation -attributor -globalopt -domtree -mem2reg -deadargelim -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -inline -functionattrs -argpromotion -domtree -sroa -basicaa -aa -memoryssa -early-cse-memssa -speculative-execution -basicaa -aa -lazy-value-info -jump-threading -correlated-propagation -simplifycfg -domtree -aggressive-instcombine -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -libcalls-shrinkwrap -loops -branch-prob -block-freq -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -pgo-memop-opt -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -tailcallelim -simplifycfg -reassociate -domtree -loops -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -loop-rotate -licm -loop-unswitch -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -indvars -loop-idiom -loop-deletion -loop-unroll -mldst-motion -phi-values -basicaa -aa -memdep -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -gvn -phi-values -basicaa -aa -memdep -memcpyopt -sccp -demanded-bits -bdce -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -lazy-value-info -jump-threading -correlated-propagation -basicaa -aa -phi-values -memdep -dse -loops -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -licm -postdomtree -adce -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -barrier -elim-avail-extern -basiccg -rpo-functionattrs -globalopt -globaldce -basiccg -globals-aa -float2int -domtree -loops -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -loop-rotate -loop-accesses -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-distribute -branch-prob -block-freq -scalar-evolution -basicaa -aa -loop-accesses -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-vectorize -loop-simplify -scalar-evolution -aa -loop-accesses -lazy-branch-prob -lazy-block-freq -loop-load-elim -basicaa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -domtree -loops -scalar-evolution -basicaa -aa -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -slp-vectorizer -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-unroll -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -licm -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -transform-warning -alignment-from-assumptions -strip-dead-prototypes -globaldce -constmerge -domtree -loops -branch-prob -block-freq -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -block-freq -loop-sink -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instsimplify -div-rem-pairs -simplifycfg -verify -domtree -targetlibinfo -domtree -loops -branch-prob -block-freq -targetlibinfo -domtree -loops -branch-prob -block-freq'
def execute_terminal_command(command):
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
    bc_files = [file for file in os.listdir(folder_path) if file.endswith(".bc")]
    return bc_files

def get_optbc_files(folder_path):
    bc_files = [file for file in os.listdir(folder_path) if file.endswith(".opt.bc")]
    return bc_files

def qw(s):
    return tuple(s.split())

def countPasses():
    count=len(qw(opt_passes_str))
    return count

opt_passes = qw(opt_passes_str)

def getPasses(opt_indice):
    return map((lambda x: opt_passes[x]), opt_indice)

def passes2indice(passes):
    indices = []
    passes = qw(passes)
    for passs in passes:
        for i in range(len(opt_passes)):
            if passs == opt_passes[i]:
                indices.append(i)
                break
    return indices

def get_time(run_dir, opt_indice, path=".", sim=False):
    ga_seq = getPasses(opt_indice)
    begin = time.time()
    folder_path = "/home/zmx/"
    cod1 = '/home/zmx/llvm9.0.0/bin/clang-9 -O0 -emit-llvm -c +' + run_dir + '/*.c'
    execute_terminal_command(cod1)
    bc_files = get_bc_files(folder_path)
    for bc_file in bc_files:
        cod2 = '/home/zmx/llvm9.0.0/bin/opt -S ' + ga_seq + ' ' + bc_file + ' -o ' + bc_file.split('.')[0]+'.opt.bc'
        execute_terminal_command(cod2)
    opt_bc_files = get_optbc_files(folder_path)
    for bc_file in opt_bc_files:  
        cod3 = '/home/zmx/llvm9.0.0/bin/llc -O0 -filetype=obj ' + bc_file
        execute_terminal_command(cod3)
    cmd3 = '/home/zmx/llvm9.0.0/bin/clang-9 -o a.out -O0 -lm *.o'
    execute_terminal_command(cmd3)
    command3 = './a.out'    
    execute_terminal_command(command3)
    cmd4 = 'rm -f *.o *.I *.s out a.out *.a *.s *.i *.bc *.opt.bc'
    execute_terminal_command(cmd4)

    res = time.time() - begin
    return res, True