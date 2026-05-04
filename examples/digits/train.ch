# Sequence helper
~seq a,b b

# Load one sample from a PGM file
~loadSample path (
    @parts (split path "/")
    @filename (nth parts ((len parts) - 1))
    @label (toInt (nth (split filename "_") 0))
    @img (readPgm path)
    @pixels (get img "pixels")
    {input: pixels, label: label}
)

# Try to load a sample
~tryLoadSample path,acc (match (fileExists path) | true -> cons (loadSample path) acc | false -> acc)

# Load samples for one digit
~loadN digit,i,count,dir,acc (match (gte i count)
    | true -> acc
    | false -> loadN digit (i + 1) count dir (tryLoadSample (str [dir, "/", digit, "_", i, ".pgm"]) acc))

# Load all digits
~loadDigit d,numPerDigit,dir,acc (match (gt d 10)
    | true -> acc
    | false -> seq (print (str ["Loading digit ", d, "..."])) (loadDigit (d + 1) numPerDigit dir (append acc (loadN d 0 numPerDigit dir []))))

~loadAllSamples numPerDigit,dir (loadDigit 0 numPerDigit dir [])

# Compute loss on a single sample
~sampleLoss net,sample (
    @output (get (forward net (get sample "input")) "output")
    @label (get sample "label")
    @p (arrayGet output label)
    @clipped (match (lt p 0.0000001) | true -> 0.0000001 | false -> p)
    0.0 - (log clipped)
)

# Compute average loss
~avgLoss net,samples ((foldl (|>acc. |>s. acc + (sampleLoss net s)) 0.0 samples) / (toFloat (len samples)))

# Train one sample and print progress
~trainOneSample net,s,lr,idx,total (seq (print (str ["  sample ", idx, "/", total, " label=", (get s "label")])) (trainOne net (get s "input") (get s "label") lr))

# Train one epoch
~trainEpochV net,samples,lr,idx,total (matchList samples (|>_. net) (|>s. |>rest. trainEpochV (trainOneSample net s lr idx total) rest lr (idx + 1) total))

# Compute accuracy
~countCorrect net,samples,acc (matchList samples (|>_. acc) (|>s. |>rest. match (eq (predict net (get s "input")) (get s "label")) | true -> countCorrect net rest (acc + 1) | false -> countCorrect net rest acc))

~computeAccuracy net,samples ((toFloat (countCorrect net samples 0)) / (toFloat (len samples)))

# Run one epoch
~runEpoch net,samples,lr (
    @net2 (trainEpochV net samples lr 1 (len samples))
    @acc (computeAccuracy net2 samples)
    @loss (avgLoss net2 samples)
    seq (print (str ["  accuracy: ", acc, "  loss: ", loss])) net2
)

# Training loop
~trainLoop net,samples,epoch,maxEpochs,lr (match (gt epoch maxEpochs)
    | true -> net
    | false -> seq (print (str ["=== Epoch ", epoch, "/", maxEpochs, " ==="])) (trainLoop (runEpoch net samples lr) samples (epoch + 1) maxEpochs lr))

# === MAIN ===

seq (print "=== Churing Neural Network Training ===") 0
seq (print "Architecture: 1024 -> 128 (ReLU) -> 64 (ReLU) -> 11 (Softmax)") 0
seq (print "Using native FloatArray for fast matrix operations") 0

# Load all 165 training samples (150 digits + 15 other)
@samples (loadAllSamples 15 "examples/digits/data/train")
seq (print (str ["Loaded ", (len samples), " training samples"])) 0

# Initialize network
seq (print "Initializing network (140,747 parameters)...") 0
@net (initNetwork 0)
seq (print "Network initialized with Xavier weights") 0

# Train for 10 epochs with learning rate 0.01
@trained (trainLoop net samples 1 10 0.01)

# Final accuracy
@finalAcc (computeAccuracy trained samples)
seq (print (str ["=== Final training accuracy: ", finalAcc, " ==="])) 0

# Save weights
seq (print "Saving weights to examples/digits/weights.json...") 0
writeFile "examples/digits/weights.json" (toJson trained)
seq (print "Done!") 0
