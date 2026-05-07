# PGM P2 (ASCII grayscale) image reader/writer

# Read a PGM P2 file and return pixels as a [Float] list [0.0, 1.0]
# Returns a dict: {width: int, height: int, pixels: [Float]}
~readPgm path (
    @lines (filter (|>line. and (not (eq (trim line) "")) (not (startsWith line "#"))) (readLines path))
    @dims (split (trim (nth lines 1)) " ")
    @w (toInt (nth dims 0))
    @h (toInt (nth dims 1))
    @maxval (toFloat (trim (nth lines 2)))
    @pixelLines (drop 3 lines)
    @pixelStr (join " " pixelLines)
    @tokens (filter (|>t. not (eq t "")) (split (trim pixelStr) " "))
    @pixels (map (|>t. (toFloat t) / maxval) tokens)
    {width: w, height: h, pixels: pixels}
)

# Write a PGM P2 file from a [Float] list of pixels
~writePgm path,w,h,pixels (
    @dimLine (str [w, " ", h])
    @pixelVals (map (|>p. toString (round (p * 255.0))) pixels)
    @pixelLine (join " " pixelVals)
    @lines ["P2", dimLine, "255", pixelLine]
    writeLines path lines
)
