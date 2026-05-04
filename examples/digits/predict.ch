# Sequence helper
~seq a,b b

# Load trained weights
seq (print "Loading weights...") 0
@weightsJson (readFile "examples/digits/weights.json")
@net (fromJson weightsJson)
seq (print "Weights loaded.") 0

# Read input image
@imgPath "examples/digits/data/test/0_0.pgm"
seq (print (str ["Image: ", imgPath])) 0
@img (readPgm imgPath)
@pixels (get img "pixels")

# Run prediction
@acts (forward net pixels)
@output (get acts "output")
@prediction (arrayArgmax output)
@label (match (eq prediction 10) | true -> "not a digit" | false -> str ["digit ", prediction])

print (str ["Predicted: ", label])
print (str ["Confidence: ", (arrayToList output)])
