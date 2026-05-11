# Benchmark: forward + backward pass timing for the digit NN
# Architecture: 1024 -> 128 (ReLU) -> 64 (ReLU) -> 11 (Softmax)

~seq a,b b

# Init
@t0 (timeMs 0)
@net (initNetwork 0)
@t1 (timeMs 0)

# Single forward pass
@input (vecRandom 1024 1.0)
@t2 (timeMs 0)
@acts (forward net input)
@t3 (timeMs 0)

# Single backward pass
@t4 (timeMs 0)
@grads (backward net acts 3)
@t5 (timeMs 0)

# 10 forward+backward passes to get a stable average
~runN n,net (match (eq n 0)
    | true -> net
    | false -> runN (n - 1) (trainOne net input 3 0.01))

@t6 (timeMs 0)
@net2 (runN 10 net)
@t7 (timeMs 0)

@initMs   (t1 - t0)
@fwdMs    (t3 - t2)
@bwdMs    (t5 - t4)
@per10Ms  (t7 - t6)
@perStepMs (per10Ms / 10)
@estTotalMs (perStepMs * 1500)

seq (print (str ["init:          ", initMs, " ms"]))
seq (print (str ["1 forward:     ", fwdMs, " ms"]))
seq (print (str ["1 backward:    ", bwdMs, " ms"]))
seq (print (str ["1 train step:  ", perStepMs, " ms (avg over 10)"]))
seq (print (str ["est. full run: ", estTotalMs, " ms  (~", (estTotalMs / 1000), " s)  [150 samples x 10 epochs]"]))
0
