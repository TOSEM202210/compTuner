The folder `Result for overview effectives`  contains the results for our Table **Results of Compiler Autotuning Techniques** 

-  `overall.txt` presents seven techniques' results for 20 programs (10 from PolyBench and 10 from cBench) on GCC and LLVM. For each program, we presents the seven techniques' speedup and time consumption for each speedup.

-  `best.txt` presents the best flag combinations of 10 selected flags (GCC1, GCC2, LLVM1, LLVM2) on P1, P2, P3, C1, C2, C3.

The folder `Result for ablation study`  contains the results for our **Ablation Study**

-  `results_for_speedup.txt` presents $CompTuner's, CompTuner_{high}'s, CompTuner_{impact}'s$ speedup performance on P1, P2, P3, C1, C2, C3 of three experiments. 

-  `results_for_prediction_error.txt` presents $CompTuner's, CompTuner_{high}'s$ prediction error on P1, P2, P3, C1, C2, C3. We use the mean value of the 50 test samples' prediction error as the result of Figure 2 and Figure 3 in our paper.

The folder `Result for Discussion`  contains the results for our Section **Discussion** 

-  `results_of_2000s.txt` presents seven techniques' results for 6 programs (3 from PolyBench and 3 from cBench) on GCC and LLVM in 2,000 seconds. For each program, we presents the seven techniques' speedup and time consumption for each speedup.

-  `results_of_10000s.txt` presents seven techniques' results for 6 programs (3 from PolyBench and 3 from cBench) on GCC and LLVM in 10,000 seconds. For each program, we presents the seven techniques' speedup and time consumption for each speedup.

-  `results_of_version.txt` presents CompTuner and BOCA results for 6 programs (3 from PolyBench and 3 from cBench) on GCC and LLVM on GCC 12.2.0 and LLVM 15.0.0. For each program, we presents the two techniques' speedup and time consumption for each speedup.

-  `results_of_flags_number.txt` presents CompTuner results for 6 programs (3 from PolyBench and 3 from cBench) on GCC and LLVM on GCC 8.3.0, GCC 12- and LLVM 9- (use the same number of flags). For each program, we presents the technique's speedup and time consumption for each speedup.

-  `results_of_model.txt` presents $CompTuner and CompTuner_{one-time}$(build the prediction model in a single phase) results for 6 programs (3 from PolyBench and 3 from cBench) on GCC 8.3.0. For each program, we presents the technique's speedup and prediction error for each speedup.

-  `results_of_AutoPhase.txt` presents CompTuner and AutoPhase results for 20 programs (10 from PolyBench and 10 from cBench) on LLVM 9.0.0. For each program, we presents the technique's speedup and time consumption for each speedup.