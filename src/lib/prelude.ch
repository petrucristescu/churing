# Prelude — pure Churing list helpers for the native compile pipeline.
# Interpreter ignores this file (OCaml VPrim versions take precedence).

~len xs (match xs | [] -> 0.0 | _ :: t -> 1.0 + len t)
~sum xs (match xs | [] -> 0.0 | h :: t -> h + sum t)
~product xs (match xs | [] -> 1.0 | h :: t -> h * product t)
~append a,b (match a | [] -> b | h :: t -> cons h (append t b))
~reverse_acc xs,acc (match xs | [] -> acc | h :: t -> reverse_acc t (cons h acc))
~reverse xs (reverse_acc xs [])
~take n,xs (match xs | [] -> [] | h :: t -> match (eq n 0.0) | true -> [] | _ -> cons h (take (n - 1.0) t))
~drop n,xs (match xs | [] -> [] | _ :: t -> match (eq n 0.0) | true -> xs | _ -> drop (n - 1.0) t)
~nth n,xs (match xs | h :: t -> match (eq n 0.0) | true -> h | _ -> nth (n - 1.0) t)
~range lo,hi (match (gte lo hi) | true -> [] | _ -> cons lo (range (lo + 1.0) hi))

~map f,xs (match xs | [] -> [] | h :: t -> cons (f h) (map f t))
~filter pred,xs (match xs | [] -> [] | h :: t -> match (pred h) | true -> cons h (filter pred t) | _ -> filter pred t)
~foldl f,acc,xs (match xs | [] -> acc | h :: t -> foldl f (f acc h) t)
~foldr f,acc,xs (match xs | [] -> acc | h :: t -> f h (foldr f acc t))
~any pred,xs (match xs | [] -> false | h :: t -> match (pred h) | true -> true | _ -> any pred t)
~all pred,xs (match xs | [] -> true | h :: t -> match (pred h) | true -> all pred t | _ -> false)
