# PGM P2 (ASCII grayscale) image reader/writer

# Read a PGM P2 file and return pixels as a FloatArray [0.0, 1.0]
# Returns a dict: {width: int, height: int, pixels: FloatArray}
~readPgm path (
    @lines (filter (|>line. and (not (eq (trim line) "")) (not (startsWith line "#"))) (readLines path))
    @dims (split (trim (nth lines 1)) " ")
    @w (toInt (nth dims 0))
    @h (toInt (nth dims 1))
    @maxval (toFloat (trim (nth lines 2)))
    # Remaining lines are pixel data — join and split by spaces
    @pixelLines (drop 3 lines)
    @pixelStr (join " " pixelLines)
    @tokens (filter (|>t. not (eq t "")) (split (trim pixelStr) " "))
    @pixels (arrayFromList (map (|>t. (toFloat t) / maxval) tokens))
    {width: w, height: h, pixels: pixels}
)

# Write a PGM P2 file from pixels (FloatArray or list)
~writePgm path,w,h,pixels (
    @dimLine (str [w, " ", h])
    @pixelList (try (arrayToList pixels) (|>_. pixels))
    @pixelVals (map (|>p. toString (round (p * 255.0))) pixelList)
    @pixelLine (join " " pixelVals)
    @lines ["P2", dimLine, "255", pixelLine]
    writeLines path lines
)
