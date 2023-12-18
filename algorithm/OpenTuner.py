import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

def execmd(cmd):
    print(cmd)
    from subprocess import Popen, PIPE
    pipe = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = pipe.communicate()
    reval = stdout.decode()
    return reval

GCC_FLAGS = ['-falign-labels', '-fauto-inc-dec', '-fbranch-count-reg', '-fcaller-saves', '-fcode-hoisting', '-fcombine-stack-adjustments', '-fcompare-elim',
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



class GccFlagsTuner(MeasurementInterface):

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            IntegerParameter('opt_level', 3, 3))
        for flag in GCC_FLAGS:
            manipulator.add_parameter(
                EnumParameter(flag,
                              ['on', 'off', 'default']))
        
        return manipulator

    def compile(self, cfg, id):
        """
        Compile a given configuration in parallel
        """
        gcc_cmd = 'gcc -c -I apps/utilities apps/utilities/polybench.c apps/correlation/*.c'
        _ = execmd(gcc_cmd)
        gcc_cmd = 'gcc -o ./tmp{0}.bin -lm *.o'.format(id)
        gcc_cmd += ' -O{0}'.format(cfg['opt_level'])
        for flag in GCC_FLAGS:
            if cfg[flag] == 'on':
                gcc_cmd += ' -f{0}'.format(flag)
            elif cfg[flag] == 'off':
                gcc_cmd += ' -fno-{0}'.format(flag)
        
        return self.call_program(gcc_cmd)

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        """
        Run a compile_result from compile() sequentially and return performance
        """
        assert compile_result['returncode'] == 0

        try:
            run_result = self.call_program('./tmp{0}.bin'.format(id))
            assert run_result['returncode'] == 0
        finally:
            self.call_program('rm ./tmp{0}.bin'.format(id))

        return Result(time=run_result['time'])

    def compile_and_run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        compile_result = self.compile(cfg, 0)
        return self.run_precompiled(desired_result, input, limit, compile_result, 0)


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    GccFlagsTuner.main(argparser.parse_args())
