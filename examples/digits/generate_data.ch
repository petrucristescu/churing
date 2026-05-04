
# Check if point (x,y) is inside rectangle [x1, y1, x2, y2]
~inRect x,y,r (and (gte x (nth r 0)) (and (lte x (nth r 2)) (and (gte y (nth r 1)) (lte y (nth r 3)))))

# Check if point is inside any rectangle
~inAnyRect x,y,rects (matchList rects (|>_. false) (|>r. |>rest. matchBool (inRect x y r) true (inAnyRect x y rest)))

# Generate a 32x32 image from rectangles
~pixelAt i,rects (
    @y (floor (i / 32))
    @x (i - y * 32)
    matchBool (inAnyRect x y rects) 1.0 0.0
)
~makeImage rects (map (|>i. pixelAt i rects) (range 0 1024))

# Clamp a value to [0.0, 1.0]
~clamp01 v (matchBool (lt v 0.0) 0.0 (matchBool (gt v 1.0) 1.0 v))

# Add random noise to pixels
~addNoise pixels,amount (map (|>p. clamp01 (p + (random 0) * amount * 2.0 - amount)) pixels)

# Shift image by dx, dy (fill edges with 0)
~shiftPixelAt i,pixels,dx,dy (
    @y (floor (i / 32))
    @x (i - y * 32)
    @sx (x - dx)
    @sy (y - dy)
    @outOfBounds (or (lt sx 0) (or (gte sx 32) (or (lt sy 0) (gte sy 32))))
    match outOfBounds
    | true -> 0.0
    | false -> nth pixels (sy * 32 + sx)
)
~shiftImage pixels,dx,dy (map (|>i. shiftPixelAt i pixels dx dy) (range 0 1024))

# Digit templates as rectangle lists
~digitRects d (match d
    | 0 -> [[8, 4, 24, 7], [8, 25, 24, 28], [8, 4, 11, 28], [21, 4, 24, 28]]
    | 1 -> [[14, 4, 17, 28], [10, 4, 17, 7]]
    | 2 -> [[8, 4, 24, 7], [21, 4, 24, 16], [8, 14, 24, 17], [8, 14, 11, 28], [8, 25, 24, 28]]
    | 3 -> [[8, 4, 24, 7], [8, 14, 24, 17], [8, 25, 24, 28], [21, 4, 24, 28]]
    | 4 -> [[8, 4, 11, 17], [8, 14, 24, 17], [21, 4, 24, 28]]
    | 5 -> [[8, 4, 24, 7], [8, 4, 11, 16], [8, 14, 24, 17], [21, 14, 24, 28], [8, 25, 24, 28]]
    | 6 -> [[8, 4, 24, 7], [8, 4, 11, 28], [8, 14, 24, 17], [21, 14, 24, 28], [8, 25, 24, 28]]
    | 7 -> [[8, 4, 24, 7], [21, 4, 24, 28]]
    | 8 -> [[8, 4, 24, 7], [8, 14, 24, 17], [8, 25, 24, 28], [8, 4, 11, 28], [21, 4, 24, 28]]
    | 9 -> [[8, 4, 24, 7], [8, 4, 11, 17], [8, 14, 24, 17], [21, 4, 24, 28], [8, 25, 24, 28]]
    | _ -> [])

# Generate and save one variation
~generateOne digit,idx,dir (
    @base (makeImage (digitRects digit))
    @dx (floor ((random 0) * 5.0) - 2)
    @dy (floor ((random 0) * 5.0) - 2)
    @shifted (shiftImage base dx dy)
    @noisy (addNoise shifted 0.1)
    @path (str [dir, "/", digit, "_", idx, ".pgm"])
    writePgm path 32 32 noisy
)

# Sequence two expressions, return second
~seq a,b b

# Generate n variations for one digit
~generateN digit,n,startIdx,dir (match n
    | 0 -> 0
    | _ -> seq (generateOne digit startIdx dir) (generateN digit (n - 1) (startIdx + 1) dir))

# Generate all digits
~generateDigit d,trainDir,testDir (match (gt d 9)
    | true -> 0
    | false -> seq (generateN d 15 0 trainDir) (seq (generateN d 5 0 testDir) (generateDigit (d + 1) trainDir testDir)))

# Generate a pure random noise image
~makeNoise dummy (map (|>_. random 0) (range 0 1024))

# Generate a blank image with very slight noise
~makeBlank dummy (map (|>_. clamp01 ((random 0) * 0.15)) (range 0 1024))

# Generate one "other" image (random noise or blank variant, label=10)
~generateOtherOne idx,dir (
    @choice (floor ((random 0) * 3.0))
    @pixels (match choice
        | 0 -> makeNoise 0
        | 1 -> makeBlank 0
        | _ -> addNoise (makeBlank 0) 0.3)
    @path (str [dir, "/10_", idx, ".pgm"])
    writePgm path 32 32 pixels
)

~generateOtherN n,startIdx,dir (match n
    | 0 -> 0
    | _ -> seq (generateOtherOne startIdx dir) (generateOtherN (n - 1) (startIdx + 1) dir))

generateDigit 0 "examples/digits/data/train" "examples/digits/data/test"
generateOtherN 15 0 "examples/digits/data/train"
generateOtherN 5 0 "examples/digits/data/test"
