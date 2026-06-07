open Ast

module StringMap = Map.Make(String)

let () = Random.self_init ()

type value =
  | VInt of int
  | VLong of int64
  | VFloat of float
  | VString of string
  | VBool of bool
  | VList of value list
  | VCons of value * value
  | VNil
  | VDict of (string * value) list
  | VFun of string list * expr * env
  | VRecFun of string * string list * expr * env  (* name * args * body * env — binds itself at call time *)
  | VPrim of (value list -> value)
  | VArray of value array  (* Generic array — any element type, O(1) access *)
  | VTensor of float array * int * int  (* data (row-major) * rows * cols *)
  | VTailCall of env * expr  (* Trampoline marker for tail call optimization *)
and env = value StringMap.t

exception RuntimeError of string
exception AssertionFailure of string

(* Directory of the source file being executed — set by churing.ml *)
let source_dir = ref ""

(* Load and parse a library file *)
let load_library library_name =
  (* Define multiple possible library paths to try *)
  let possible_paths = [
    (* Path relative to source file's directory (lib/ subfolder) *)
    Filename.concat !source_dir "lib";
    (* Path relative to source file's parent directory (lib/ subfolder) *)
    Filename.concat (Filename.dirname !source_dir) "lib";
    (* Path relative to source file's directory *)
    !source_dir;
    (* Path relative to executable in _build *)
    Filename.concat (Filename.dirname Sys.argv.(0)) "../lib";
    (* Path in source directory *)
    Filename.concat (Filename.dirname Sys.argv.(0)) "../../src/lib";
    (* Absolute path to src/lib - use this as fallback *)
    Sys.getcwd() ^ "/src/lib"
  ] in

  (* Try to find the library file in any of the possible paths *)
  let rec find_library paths =
    match paths with
    | [] -> raise (RuntimeError ("Library not found: " ^ library_name))
    | path :: rest ->
        let lib_file = Filename.concat path (library_name ^ ".ch") in
        if Sys.file_exists lib_file then
          lib_file
        else
          find_library rest
  in

  let lib_file = find_library possible_paths in
  try
    let ic = open_in lib_file in
    let len = in_channel_length ic in
    let src = really_input_string ic len in
    close_in ic;
    let exprs = Parser.parse src in
    exprs
  with
  | Sys_error msg -> raise (RuntimeError ("Library error: " ^ msg))
  | Parser.ParseError (msg, line, col) ->
      raise (RuntimeError (Printf.sprintf "Parse error in library %s at line %d, col %d: %s"
                           library_name line col msg))

(* Import cache to avoid loading the same library multiple times *)
let imported_libraries = ref StringMap.empty

(* String representation of a value (without newline) *)
let rec string_of_value = function
  | VInt n -> string_of_int n
  | VLong n -> Int64.to_string n
  | VFloat f ->
      if Float.equal (Float.round f) f then string_of_int (int_of_float f)
      else string_of_float f
  | VString s -> s
  | VBool true -> "true"
  | VBool false -> "false"
  | VList vs -> "[" ^ String.concat ", " (List.map string_of_value vs) ^ "]"
  | VCons _ as c -> string_of_cons c
  | VNil -> "nil"
  | VDict entries ->
      "{" ^ String.concat ", " (List.map (fun (k, v) -> k ^ ": " ^ string_of_value v) entries) ^ "}"
  | VArray arr ->
      "Array[" ^ String.concat ", " (Array.to_list (Array.map string_of_value arr)) ^ "]"
  | VTensor (_, rows, cols) ->
      Printf.sprintf "Tensor(%dx%d)" rows cols
  | VTailCall _ -> "<tailcall>"
  | VRecFun _ | _ -> "<fun>"

and string_of_cons v =
  let rec collect acc = function
    | VCons (h, t) -> collect (h :: acc) t
    | VNil -> "[" ^ String.concat ", " (List.rev_map string_of_value acc) ^ "]"
    | tail -> "(" ^ String.concat ", " (List.rev_map string_of_value acc) ^ " . " ^ string_of_value tail ^ ")"
  in
  collect [] v

(* Print function for values *)
let print_value v = print_endline (string_of_value v)

(* Boolean primitives using native VBool *)
let create_church_booleans env =
  let expect_bool name = function
    | VBool b -> b
    | _ -> raise (RuntimeError (name ^ ": expected boolean"))
  in
  let not_fn = VPrim (fun args -> VBool (not (expect_bool "not" (List.hd args)))) in
  let and_fn = VPrim (fun args ->
    let p = expect_bool "and" (List.hd args) in
    VPrim (fun args -> VBool (p && expect_bool "and" (List.hd args)))) in
  let or_fn = VPrim (fun args ->
    let p = expect_bool "or" (List.hd args) in
    VPrim (fun args -> VBool (p || expect_bool "or" (List.hd args)))) in
  let if_fn = VPrim (fun args ->
    let c = expect_bool "if" (List.hd args) in
    VPrim (fun args ->
      let t = List.hd args in
      VPrim (fun args ->
        let f = List.hd args in
        if c then t else f))) in
  env
  |> StringMap.add "true" (VBool true)
  |> StringMap.add "false" (VBool false)
  |> StringMap.add "not" not_fn
  |> StringMap.add "and" and_fn
  |> StringMap.add "or" or_fn
  |> StringMap.add "if" if_fn
  (* Print: side-effect output, returns the value printed *)
  |> StringMap.add "print" (VPrim (fun args ->
      let v = List.hd args in
      print_endline (string_of_value v);
      v))
  (* Environment variable access *)
  |> StringMap.add "env" (VPrim (fun args ->
      match List.hd args with
      | VString name ->
          (match Sys.getenv_opt name with
           | Some v -> VString v
           | None -> raise (RuntimeError ("env: variable not set: " ^ name)))
      | _ -> raise (RuntimeError "env: expected string")))
  |> StringMap.add "envOr" (VPrim (fun args ->
      match List.hd args with
      | VString name ->
          VPrim (fun args ->
            let default = List.hd args in
            match Sys.getenv_opt name with
            | Some v -> VString v
            | None -> default)
      | _ -> raise (RuntimeError "envOr: expected string")))
  (* Random number generation *)
  |> StringMap.add "random" (VPrim (fun _args -> VFloat (Random.float 1.0)))
  (* Comparison operators — work on numeric and string values *)
  |> StringMap.add "gt" (VPrim (fun args ->
      let a = List.hd args in
      VPrim (fun args ->
        let b = List.hd args in
        VBool (match a, b with
          | VInt x, VInt y -> x > y
          | VFloat x, VFloat y -> x > y
          | VInt x, VFloat y -> float_of_int x > y
          | VFloat x, VInt y -> x > float_of_int y
          | VLong x, VLong y -> Int64.compare x y > 0
          | VInt x, VLong y -> Int64.compare (Int64.of_int x) y > 0
          | VLong x, VInt y -> Int64.compare x (Int64.of_int y) > 0
          | VString x, VString y -> String.compare x y > 0
          | _ -> raise (RuntimeError "gt: incompatible types")))))
  |> StringMap.add "lt" (VPrim (fun args ->
      let a = List.hd args in
      VPrim (fun args ->
        let b = List.hd args in
        VBool (match a, b with
          | VInt x, VInt y -> x < y
          | VFloat x, VFloat y -> x < y
          | VInt x, VFloat y -> float_of_int x < y
          | VFloat x, VInt y -> x < float_of_int y
          | VLong x, VLong y -> Int64.compare x y < 0
          | VInt x, VLong y -> Int64.compare (Int64.of_int x) y < 0
          | VLong x, VInt y -> Int64.compare x (Int64.of_int y) < 0
          | VString x, VString y -> String.compare x y < 0
          | _ -> raise (RuntimeError "lt: incompatible types")))))
  |> StringMap.add "gte" (VPrim (fun args ->
      let a = List.hd args in
      VPrim (fun args ->
        let b = List.hd args in
        VBool (match a, b with
          | VInt x, VInt y -> x >= y
          | VFloat x, VFloat y -> x >= y
          | VInt x, VFloat y -> float_of_int x >= y
          | VFloat x, VInt y -> x >= float_of_int y
          | VLong x, VLong y -> Int64.compare x y >= 0
          | VInt x, VLong y -> Int64.compare (Int64.of_int x) y >= 0
          | VLong x, VInt y -> Int64.compare x (Int64.of_int y) >= 0
          | VString x, VString y -> String.compare x y >= 0
          | _ -> raise (RuntimeError "gte: incompatible types")))))
  |> StringMap.add "lte" (VPrim (fun args ->
      let a = List.hd args in
      VPrim (fun args ->
        let b = List.hd args in
        VBool (match a, b with
          | VInt x, VInt y -> x <= y
          | VFloat x, VFloat y -> x <= y
          | VInt x, VFloat y -> float_of_int x <= y
          | VFloat x, VInt y -> x <= float_of_int y
          | VLong x, VLong y -> Int64.compare x y <= 0
          | VInt x, VLong y -> Int64.compare (Int64.of_int x) y <= 0
          | VLong x, VInt y -> Int64.compare x (Int64.of_int y) <= 0
          | VString x, VString y -> String.compare x y <= 0
          | _ -> raise (RuntimeError "lte: incompatible types")))))

(* Math primitives *)
let create_math_functions env =
  let expect_numeric name = function
    | VInt n -> float_of_int n
    | VLong n -> Int64.to_float n
    | VFloat f -> f
    | _ -> raise (RuntimeError (name ^ ": expected numeric"))
  in
  let math1 name f = VPrim (fun args -> VFloat (f (expect_numeric name (List.hd args)))) in
  let math2 name f = VPrim (fun args ->
    let a = expect_numeric name (List.hd args) in
    VPrim (fun args -> VFloat (f a (expect_numeric name (List.hd args))))) in
  env
  |> StringMap.add "sqrt" (math1 "sqrt" Float.sqrt)
  |> StringMap.add "sin" (math1 "sin" Float.sin)
  |> StringMap.add "cos" (math1 "cos" Float.cos)
  |> StringMap.add "tan" (math1 "tan" Float.tan)
  |> StringMap.add "asin" (math1 "asin" Float.asin)
  |> StringMap.add "acos" (math1 "acos" Float.acos)
  |> StringMap.add "atan" (math1 "atan" Float.atan)
  |> StringMap.add "floor" (VPrim (fun args ->
      VInt (int_of_float (Float.floor (expect_numeric "floor" (List.hd args))))))
  |> StringMap.add "ceil" (VPrim (fun args ->
      VInt (int_of_float (Float.ceil (expect_numeric "ceil" (List.hd args))))))
  |> StringMap.add "round" (VPrim (fun args ->
      VInt (int_of_float (Float.round (expect_numeric "round" (List.hd args))))))
  |> StringMap.add "abs" (VPrim (fun args ->
      match List.hd args with
      | VInt n -> VInt (abs n)
      | VLong n -> VLong (Int64.abs n)
      | VFloat f -> VFloat (Float.abs f)
      | _ -> raise (RuntimeError "abs: expected numeric")))
  |> StringMap.add "pow" (math2 "pow" Float.pow)
  |> StringMap.add "exp" (math1 "exp" Float.exp)
  |> StringMap.add "log" (math1 "log" Float.log)
  |> StringMap.add "tanh" (math1 "tanh" Float.tanh)
  |> StringMap.add "min" (VPrim (fun args ->
      let a = expect_numeric "min" (List.hd args) in
      VPrim (fun args -> VFloat (Float.min a (expect_numeric "min" (List.hd args))))))
  |> StringMap.add "max" (VPrim (fun args ->
      let a = expect_numeric "max" (List.hd args) in
      VPrim (fun args -> VFloat (Float.max a (expect_numeric "max" (List.hd args))))))

(* Tensor primitives — contiguous float array, row-major layout *)
let create_tensor_functions env =
  let expect_tensor name = function
    | VTensor (data, rows, cols) -> (data, rows, cols)
    | _ -> raise (RuntimeError (name ^ ": expected Tensor"))
  in
  let expect_int name = function
    | VInt n -> n
    | _ -> raise (RuntimeError (name ^ ": expected int"))
  in
  let to_float name = function
    | VFloat f -> f
    | VInt n -> float_of_int n
    | VLong n -> Int64.to_float n
    | _ -> raise (RuntimeError (name ^ ": expected numeric"))
  in
  let list_of name = function
    | VList vs -> vs
    | VNil -> []
    | _ -> raise (RuntimeError (name ^ ": expected list"))
  in
  env
  (* tensorCreate rows cols val — filled tensor *)
  |> StringMap.add "tensorCreate" (VPrim (fun args ->
      let rows = expect_int "tensorCreate" (List.hd args) in
      VPrim (fun args ->
        let cols = expect_int "tensorCreate" (List.hd args) in
        VPrim (fun args ->
          let v = to_float "tensorCreate" (List.hd args) in
          VTensor (Array.make (rows * cols) v, rows, cols)))))
  (* tensorGet t row col *)
  |> StringMap.add "tensorGet" (VPrim (fun args ->
      let (data, rows, cols) = expect_tensor "tensorGet" (List.hd args) in
      VPrim (fun args ->
        let row = expect_int "tensorGet" (List.hd args) in
        VPrim (fun args ->
          let col = expect_int "tensorGet" (List.hd args) in
          if row < 0 || row >= rows || col < 0 || col >= cols then
            raise (RuntimeError "tensorGet: index out of bounds")
          else
            VFloat data.(row * cols + col)))))
  (* tensorSet t row col val — returns new tensor (immutable) *)
  |> StringMap.add "tensorSet" (VPrim (fun args ->
      let (data, rows, cols) = expect_tensor "tensorSet" (List.hd args) in
      VPrim (fun args ->
        let row = expect_int "tensorSet" (List.hd args) in
        VPrim (fun args ->
          let col = expect_int "tensorSet" (List.hd args) in
          VPrim (fun args ->
            let v = to_float "tensorSet" (List.hd args) in
            let d = Array.copy data in
            d.(row * cols + col) <- v;
            VTensor (d, rows, cols))))))
  (* tensorRows / tensorCols / tensorShape *)
  |> StringMap.add "tensorRows" (VPrim (fun args ->
      let (_, rows, _) = expect_tensor "tensorRows" (List.hd args) in
      VInt rows))
  |> StringMap.add "tensorCols" (VPrim (fun args ->
      let (_, _, cols) = expect_tensor "tensorCols" (List.hd args) in
      VInt cols))
  |> StringMap.add "tensorShape" (VPrim (fun args ->
      let (_, rows, cols) = expect_tensor "tensorShape" (List.hd args) in
      VList [VInt rows; VInt cols]))
  (* tensorVec [Float] — 1D tensor (rows=n, cols=1) *)
  |> StringMap.add "tensorVec" (VPrim (fun args ->
      let lst = list_of "tensorVec" (List.hd args) in
      let n = List.length lst in
      let data = Array.init n (fun i -> to_float "tensorVec" (List.nth lst i)) in
      VTensor (data, n, 1)))
  (* tensorMat [[Float]] — 2D tensor *)
  |> StringMap.add "tensorMat" (VPrim (fun args ->
      let rows_list = list_of "tensorMat" (List.hd args) in
      let nrows = List.length rows_list in
      if nrows = 0 then VTensor ([||], 0, 0)
      else begin
        let get_row v = List.map (to_float "tensorMat") (list_of "tensorMat" v) in
        let float_rows = List.map get_row rows_list in
        let ncols = List.length (List.hd float_rows) in
        let data = Array.make (nrows * ncols) 0.0 in
        List.iteri (fun i row ->
          List.iteri (fun j v -> data.(i * ncols + j) <- v) row) float_rows;
        VTensor (data, nrows, ncols)
      end))
  (* tensorToVec t — tensor → [Float] *)
  |> StringMap.add "tensorToVec" (VPrim (fun args ->
      let (data, _, _) = expect_tensor "tensorToVec" (List.hd args) in
      VList (Array.to_list (Array.map (fun f -> VFloat f) data))))
  (* tensorToMat t — tensor → [[Float]] *)
  |> StringMap.add "tensorToMat" (VPrim (fun args ->
      let (data, rows, cols) = expect_tensor "tensorToMat" (List.hd args) in
      VList (List.init rows (fun i ->
        VList (List.init cols (fun j -> VFloat data.(i * cols + j)))))))
  (* tensorRandom rows cols scale — uniform random in [-scale, scale] *)
  |> StringMap.add "tensorRandom" (VPrim (fun args ->
      let rows = expect_int "tensorRandom" (List.hd args) in
      VPrim (fun args ->
        let cols = expect_int "tensorRandom" (List.hd args) in
        VPrim (fun args ->
          let scale = to_float "tensorRandom" (List.hd args) in
          let data = Array.init (rows * cols)
            (fun _ -> Random.float (2.0 *. scale) -. scale) in
          VTensor (data, rows, cols)))))

(* Array primitives — generic O(1) access, type tracked by HM inference *)
let create_array_functions env =
  let expect_array name = function
    | VArray a -> a
    | _ -> raise (RuntimeError (name ^ ": expected Array"))
  in
  let expect_int name = function
    | VInt n -> n
    | _ -> raise (RuntimeError (name ^ ": expected int"))
  in
  env
  (* arrayCreate n val — create array of n elements, all set to val *)
  |> StringMap.add "arrayCreate" (VPrim (fun args ->
      let n = expect_int "arrayCreate" (List.hd args) in
      VPrim (fun args ->
        VArray (Array.make n (List.hd args)))))
  (* arrayGet arr i — O(1) index access *)
  |> StringMap.add "arrayGet" (VPrim (fun args ->
      let arr = expect_array "arrayGet" (List.hd args) in
      VPrim (fun args ->
        let i = expect_int "arrayGet" (List.hd args) in
        if i < 0 || i >= Array.length arr then
          raise (RuntimeError "arrayGet: index out of bounds")
        else arr.(i))))
  (* arraySet arr i val — returns NEW array with element changed *)
  |> StringMap.add "arraySet" (VPrim (fun args ->
      let arr = expect_array "arraySet" (List.hd args) in
      VPrim (fun args ->
        let i = expect_int "arraySet" (List.hd args) in
        VPrim (fun args ->
          let a = Array.copy arr in
          a.(i) <- List.hd args;
          VArray a))))
  (* arrayLength arr *)
  |> StringMap.add "arrayLength" (VPrim (fun args ->
      VInt (Array.length (expect_array "arrayLength" (List.hd args)))))
  (* arrayFromList list — convert list to Array *)
  |> StringMap.add "arrayFromList" (VPrim (fun args ->
      match List.hd args with
      | VList vs -> VArray (Array.of_list vs)
      | _ -> raise (RuntimeError "arrayFromList: expected list")))
  (* arrayToList arr — convert Array to list *)
  |> StringMap.add "arrayToList" (VPrim (fun args ->
      VList (Array.to_list (expect_array "arrayToList" (List.hd args)))))

(* String primitives *)
let create_string_functions env =
  let expect_string name = function
    | VString s -> s
    | _ -> raise (RuntimeError (name ^ ": expected string"))
  in
  let expect_int name = function
    | VInt n -> n
    | _ -> raise (RuntimeError (name ^ ": expected int"))
  in
  env
  |> StringMap.add "length" (VPrim (fun args ->
      VInt (String.length (expect_string "length" (List.hd args)))))
  |> StringMap.add "concat" (VPrim (fun args ->
      let s1 = expect_string "concat" (List.hd args) in
      VPrim (fun args -> VString (s1 ^ expect_string "concat" (List.hd args)))))
  |> StringMap.add "substring" (VPrim (fun args ->
      let s = expect_string "substring" (List.hd args) in
      VPrim (fun args ->
        let start = expect_int "substring" (List.hd args) in
        VPrim (fun args ->
          let len = expect_int "substring" (List.hd args) in
          if start < 0 || start + len > String.length s then
            raise (RuntimeError "substring: index out of bounds")
          else VString (String.sub s start len)))))
  |> StringMap.add "uppercase" (VPrim (fun args ->
      VString (String.uppercase_ascii (expect_string "uppercase" (List.hd args)))))
  |> StringMap.add "lowercase" (VPrim (fun args ->
      VString (String.lowercase_ascii (expect_string "lowercase" (List.hd args)))))
  |> StringMap.add "trim" (VPrim (fun args ->
      VString (String.trim (expect_string "trim" (List.hd args)))))
  |> StringMap.add "charAt" (VPrim (fun args ->
      let s = expect_string "charAt" (List.hd args) in
      VPrim (fun args ->
        let i = expect_int "charAt" (List.hd args) in
        if i < 0 || i >= String.length s then
          raise (RuntimeError "charAt: index out of bounds")
        else VString (String.make 1 s.[i]))))
  |> StringMap.add "indexOf" (VPrim (fun args ->
      let haystack = expect_string "indexOf" (List.hd args) in
      VPrim (fun args ->
        let needle = expect_string "indexOf" (List.hd args) in
        let len_h = String.length haystack in
        let len_n = String.length needle in
        let rec search i =
          if i > len_h - len_n then VInt (-1)
          else if String.sub haystack i len_n = needle then VInt i
          else search (i + 1)
        in
        if len_n = 0 then VInt 0
        else if len_n > len_h then VInt (-1)
        else search 0)))
  |> StringMap.add "startsWith" (VPrim (fun args ->
      let s = expect_string "startsWith" (List.hd args) in
      VPrim (fun args ->
        let prefix = expect_string "startsWith" (List.hd args) in
        let len = String.length prefix in
        VBool (len <= String.length s && String.sub s 0 len = prefix))))
  |> StringMap.add "endsWith" (VPrim (fun args ->
      let s = expect_string "endsWith" (List.hd args) in
      VPrim (fun args ->
        let suffix = expect_string "endsWith" (List.hd args) in
        let slen = String.length s in
        let suflen = String.length suffix in
        VBool (suflen <= slen && String.sub s (slen - suflen) suflen = suffix))))
  |> StringMap.add "replace" (VPrim (fun args ->
      let s = expect_string "replace" (List.hd args) in
      VPrim (fun args ->
        let old_s = expect_string "replace" (List.hd args) in
        VPrim (fun args ->
          let new_s = expect_string "replace" (List.hd args) in
          let old_len = String.length old_s in
          if old_len = 0 then VString s
          else
            let buf = Buffer.create (String.length s) in
            let rec aux i =
              if i >= String.length s then ()
              else if i + old_len <= String.length s && String.sub s i old_len = old_s then
                (Buffer.add_string buf new_s; aux (i + old_len))
              else (Buffer.add_char buf s.[i]; aux (i + 1))
            in
            aux 0; VString (Buffer.contents buf)))))
  |> StringMap.add "toString" (VPrim (fun args ->
      match List.hd args with
      | VString s -> VString s
      | VInt n -> VString (string_of_int n)
      | VLong n -> VString (Int64.to_string n)
      | VFloat f ->
          if Float.equal (Float.round f) f then VString (string_of_int (int_of_float f))
          else VString (string_of_float f)
      | VBool true -> VString "true"
      | VBool false -> VString "false"
      | v -> VString (string_of_value v)))
  (* str: concatenate a list of values into a string (auto-converts each) *)
  |> StringMap.add "str" (VPrim (fun args ->
      match List.hd args with
      | VList vs ->
          VString (String.concat "" (List.map string_of_value vs))
      | v -> VString (string_of_value v)))
  (* toFloat / toInt: parse strings to numbers *)
  |> StringMap.add "toFloat" (VPrim (fun args ->
      match List.hd args with
      | VString s ->
          (try VFloat (float_of_string s)
           with Failure _ -> raise (RuntimeError ("toFloat: invalid number: " ^ s)))
      | VInt n -> VFloat (float_of_int n)
      | VLong n -> VFloat (Int64.to_float n)
      | VFloat f -> VFloat f
      | _ -> raise (RuntimeError "toFloat: expected string or number")))
  |> StringMap.add "toInt" (VPrim (fun args ->
      match List.hd args with
      | VString s ->
          (try VInt (int_of_string s)
           with Failure _ -> raise (RuntimeError ("toInt: invalid integer: " ^ s)))
      | VInt n -> VInt n
      | VLong n -> VInt (Int64.to_int n)
      | VFloat f -> VInt (int_of_float f)
      | _ -> raise (RuntimeError "toInt: expected string or number")))
  (* split: split a string by a delimiter *)
  |> StringMap.add "split" (VPrim (fun args ->
      let s = expect_string "split" (List.hd args) in
      VPrim (fun args ->
        let delim = expect_string "split" (List.hd args) in
        if String.length delim <> 1 then
          raise (RuntimeError "split: delimiter must be a single character")
        else
          let parts = String.split_on_char delim.[0] s in
          VList (List.map (fun p -> VString p) parts))))
  (* join: join a list of values with a separator *)
  |> StringMap.add "join" (VPrim (fun args ->
      let sep = (match List.hd args with
        | VString s -> s
        | _ -> raise (RuntimeError "join: separator must be a string")) in
      VPrim (fun args ->
        match List.hd args with
        | VList vs ->
            VString (String.concat sep (List.map string_of_value vs))
        | _ -> raise (RuntimeError "join: expected list"))))

(* Date/Time primitives *)
let create_time_functions env =
  let expect_int name = function
    | VInt n -> n
    | _ -> raise (RuntimeError (name ^ ": expected int"))
  in
  let tm_of_ts ts = Unix.localtime (float_of_int ts) in
  env
  |> StringMap.add "now" (VPrim (fun _ -> VInt (int_of_float (Unix.time ()))))
  |> StringMap.add "timeMs" (VPrim (fun _ -> VInt (int_of_float (Unix.gettimeofday () *. 1000.0))))
  |> StringMap.add "year" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "year" (List.hd args)) in
      VInt (tm.Unix.tm_year + 1900)))
  |> StringMap.add "month" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "month" (List.hd args)) in
      VInt (tm.Unix.tm_mon + 1)))
  |> StringMap.add "day" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "day" (List.hd args)) in
      VInt tm.Unix.tm_mday))
  |> StringMap.add "hour" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "hour" (List.hd args)) in
      VInt tm.Unix.tm_hour))
  |> StringMap.add "minute" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "minute" (List.hd args)) in
      VInt tm.Unix.tm_min))
  |> StringMap.add "second" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "second" (List.hd args)) in
      VInt tm.Unix.tm_sec))
  |> StringMap.add "diffTime" (VPrim (fun args ->
      let a = expect_int "diffTime" (List.hd args) in
      VPrim (fun args ->
        let b = expect_int "diffTime" (List.hd args) in
        VInt (a - b))))
  |> StringMap.add "dayOfWeek" (VPrim (fun args ->
      let tm = tm_of_ts (expect_int "dayOfWeek" (List.hd args)) in
      VInt tm.Unix.tm_wday))

(* File I/O primitives *)
(* Dict primitives *)
let create_dict_functions env =
  let expect_string name = function
    | VString s -> s
    | _ -> raise (RuntimeError (name ^ ": expected string"))
  in
  let expect_dict name = function
    | VDict entries -> entries
    | _ -> raise (RuntimeError (name ^ ": expected dict"))
  in
  env
  |> StringMap.add "get" (VPrim (fun args ->
      let d = expect_dict "get" (List.hd args) in
      VPrim (fun args ->
        let k = expect_string "get" (List.hd args) in
        match List.assoc_opt k d with
        | Some v -> v
        | None -> raise (RuntimeError ("get: key not found: " ^ k)))))
  |> StringMap.add "set" (VPrim (fun args ->
      let d = expect_dict "set" (List.hd args) in
      VPrim (fun args ->
        let k = expect_string "set" (List.hd args) in
        VPrim (fun args ->
          let v = List.hd args in
          let d' = List.filter (fun (k', _) -> k' <> k) d in
          VDict ((k, v) :: d')))))
  |> StringMap.add "has" (VPrim (fun args ->
      let d = expect_dict "has" (List.hd args) in
      VPrim (fun args ->
        let k = expect_string "has" (List.hd args) in
        VBool (List.assoc_opt k d <> None))))
  |> StringMap.add "keys" (VPrim (fun args ->
      let d = expect_dict "keys" (List.hd args) in
      VList (List.map (fun (k, _) -> VString k) d)))
  |> StringMap.add "values" (VPrim (fun args ->
      let d = expect_dict "values" (List.hd args) in
      VList (List.map (fun (_, v) -> v) d)))
  |> StringMap.add "merge" (VPrim (fun args ->
      let d1 = expect_dict "merge" (List.hd args) in
      VPrim (fun args ->
        let d2 = expect_dict "merge" (List.hd args) in
        let merged = List.fold_left (fun acc (k, v) ->
          if List.assoc_opt k acc <> None then acc
          else (k, v) :: acc) d2 d1 in
        VDict merged)))
  |> StringMap.add "remove" (VPrim (fun args ->
      let d = expect_dict "remove" (List.hd args) in
      VPrim (fun args ->
        let k = expect_string "remove" (List.hd args) in
        VDict (List.filter (fun (k', _) -> k' <> k) d))))
  |> StringMap.add "entries" (VPrim (fun args ->
      let d = expect_dict "entries" (List.hd args) in
      VList (List.map (fun (k, v) -> VList [VString k; v]) d)))
  |> StringMap.add "fromEntries" (VPrim (fun args ->
      match List.hd args with
      | VList pairs ->
          VDict (List.map (fun pair ->
            match pair with
            | VList [VString k; v] -> (k, v)
            | _ -> raise (RuntimeError "fromEntries: expected [key, value] pairs")
          ) pairs)
      | _ -> raise (RuntimeError "fromEntries: expected list")))

(* JSON primitives *)
let rec value_to_json = function
  | VInt n -> string_of_int n
  | VLong n -> Int64.to_string n
  | VFloat f ->
      if Float.equal (Float.round f) f then string_of_int (int_of_float f)
      else string_of_float f
  | VString s -> "\"" ^ String.concat "" (List.map (fun c ->
      match c with
      | '"' -> "\\\""
      | '\\' -> "\\\\"
      | '\n' -> "\\n"
      | '\t' -> "\\t"
      | c -> String.make 1 c
    ) (List.init (String.length s) (String.get s))) ^ "\""
  | VBool true -> "true"
  | VBool false -> "false"
  | VNil -> "null"
  | VList vs -> "[" ^ String.concat "," (List.map value_to_json vs) ^ "]"
  | VArray arr ->
      "{\"__array__\":[" ^ String.concat "," (Array.to_list (Array.map value_to_json arr)) ^ "]}"
  | VDict entries ->
      "{" ^ String.concat "," (List.map (fun (k, v) ->
        "\"" ^ k ^ "\":" ^ value_to_json v) entries) ^ "}"
  | _ -> "null"

(* Simple recursive-descent JSON parser *)
let json_to_value s =
  let len = String.length s in
  let rec skip_ws i = if i < len && (s.[i] = ' ' || s.[i] = '\t' || s.[i] = '\n' || s.[i] = '\r') then skip_ws (i+1) else i in
  let rec parse_value i =
    let i = skip_ws i in
    if i >= len then raise (RuntimeError "fromJson: unexpected end of input")
    else if s.[i] = '"' then parse_string (i+1) (Buffer.create 16)
    else if s.[i] = '[' then parse_array (i+1) []
    else if s.[i] = '{' then parse_object (i+1) []
    else if s.[i] = 't' && i+3 < len && String.sub s i 4 = "true" then (VBool true, i+4)
    else if s.[i] = 'f' && i+4 < len && String.sub s i 5 = "false" then (VBool false, i+5)
    else if s.[i] = 'n' && i+3 < len && String.sub s i 4 = "null" then (VNil, i+4)
    else if s.[i] = '-' || (s.[i] >= '0' && s.[i] <= '9') then parse_number i
    else raise (RuntimeError (Printf.sprintf "fromJson: unexpected char '%c' at %d" s.[i] i))
  and parse_string i buf =
    if i >= len then raise (RuntimeError "fromJson: unterminated string")
    else if s.[i] = '"' then (VString (Buffer.contents buf), i+1)
    else if s.[i] = '\\' && i+1 < len then (
      (match s.[i+1] with
       | '"' -> Buffer.add_char buf '"'
       | '\\' -> Buffer.add_char buf '\\'
       | 'n' -> Buffer.add_char buf '\n'
       | 't' -> Buffer.add_char buf '\t'
       | '/' -> Buffer.add_char buf '/'
       | c -> Buffer.add_char buf '\\'; Buffer.add_char buf c);
      parse_string (i+2) buf)
    else (Buffer.add_char buf s.[i]; parse_string (i+1) buf)
  and parse_number i =
    let j = ref i in
    let is_float = ref false in
    if !j < len && s.[!j] = '-' then incr j;
    while !j < len && s.[!j] >= '0' && s.[!j] <= '9' do incr j done;
    if !j < len && s.[!j] = '.' then (is_float := true; incr j;
      while !j < len && s.[!j] >= '0' && s.[!j] <= '9' do incr j done);
    if !j < len && (s.[!j] = 'e' || s.[!j] = 'E') then (is_float := true; incr j;
      if !j < len && (s.[!j] = '+' || s.[!j] = '-') then incr j;
      while !j < len && s.[!j] >= '0' && s.[!j] <= '9' do incr j done);
    let num_str = String.sub s i (!j - i) in
    if !is_float then (VFloat (float_of_string num_str), !j)
    else (VInt (int_of_string num_str), !j)
  and parse_array i acc =
    let i = skip_ws i in
    if i < len && s.[i] = ']' then (VList (List.rev acc), i+1)
    else
      let v, i = parse_value i in
      let i = skip_ws i in
      if i < len && s.[i] = ',' then parse_array (i+1) (v :: acc)
      else if i < len && s.[i] = ']' then (VList (List.rev (v :: acc)), i+1)
      else raise (RuntimeError "fromJson: expected ',' or ']' in array")
  and parse_object i acc =
    let i = skip_ws i in
    if i < len && s.[i] = '}' then (VDict (List.rev acc), i+1)
    else if i < len && s.[i] = '"' then
      let key_v, i = parse_string (i+1) (Buffer.create 16) in
      let key = match key_v with VString k -> k | _ -> failwith "impossible" in
      let i = skip_ws i in
      if i >= len || s.[i] <> ':' then raise (RuntimeError "fromJson: expected ':' after key");
      let v, i = parse_value (i+1) in
      let i = skip_ws i in
      if i < len && s.[i] = ',' then parse_object (i+1) ((key, v) :: acc)
      else if i < len && s.[i] = '}' then
        let entries = List.rev ((key, v) :: acc) in
        (match entries with
         | [("__array__", VList items)] -> (VArray (Array.of_list items), i+1)
         | _ -> (VDict entries, i+1))
      else raise (RuntimeError "fromJson: expected ',' or '}' in object")
    else raise (RuntimeError "fromJson: expected key string in object")
  in
  fst (parse_value (skip_ws 0))

let create_json_functions env =
  env
  |> StringMap.add "toJson" (VPrim (fun args -> VString (value_to_json (List.hd args))))
  |> StringMap.add "fromJson" (VPrim (fun args ->
      match List.hd args with
      | VString s -> json_to_value s
      | _ -> raise (RuntimeError "fromJson: expected string")))

(* MySQL driver — shells out to mysql CLI *)
let create_mysql_functions env =
  let expect_string name = function
    | VString s -> s
    | _ -> raise (RuntimeError (name ^ ": expected string"))
  in
  let expect_dict name = function
    | VDict d -> d
    | _ -> raise (RuntimeError (name ^ ": expected dict"))
  in
  let get_field d key default =
    match List.assoc_opt key d with
    | Some (VString s) -> s
    | Some (VInt n) -> string_of_int n
    | _ -> default
  in
  (* Build mysql CLI command from connection dict *)
  let mysql_args conn =
    let d = expect_dict "mysql" conn in
    let host = get_field d "host" "127.0.0.1" in
    let port = get_field d "port" "3306" in
    let user = get_field d "user" "root" in
    let pass = get_field d "password" "" in
    let db = get_field d "database" "" in
    let pass_arg = if pass = "" then "" else "-p" ^ pass in
    [| "mysql"; "-h"; host; "-P"; port; "-u"; user; pass_arg; db; "--batch"; "--raw" |]
  in
  (* Parse tab-separated mysql --batch output into list of dicts *)
  let parse_mysql_output output =
    let lines = String.split_on_char '\n' output in
    let lines = List.filter (fun s -> String.length (String.trim s) > 0) lines in
    match lines with
    | [] -> VList []
    | header :: rows ->
        let cols = String.split_on_char '\t' header in
        let ncols = List.length cols in
        VList (List.map (fun row ->
          let vals = String.split_on_char '\t' row in
          let nvals = List.length vals in
          let padded =
            if nvals < ncols then vals @ List.init (ncols - nvals) (fun _ -> "NULL")
            else if nvals > ncols then List.filteri (fun i _ -> i < ncols) vals
            else vals in
          let pairs = List.combine cols padded in
          VDict (List.map (fun (k, v) ->
            if v = "NULL" then (k, VNil)
            else
              (* Try to parse as int, then float, then keep as string *)
              try (k, VInt (int_of_string v))
              with _ ->
                try (k, VFloat (float_of_string v))
                with _ -> (k, VString v)
          ) pairs)
        ) rows)
  in
  let read_all ic =
    let buf = Buffer.create 256 in
    (try while true do Buffer.add_char buf (input_char ic) done
     with End_of_file -> ());
    Buffer.contents buf
  in
  let run_mysql conn sql =
    let args = mysql_args conn in
    let (stdout_ic, stdin_oc, stderr_ic) = Unix.open_process_full
      (String.concat " " (Array.to_list (Array.map Filename.quote args)))
      (Unix.environment ()) in
    output_string stdin_oc sql;
    output_char stdin_oc '\n';
    close_out stdin_oc;
    let output = read_all stdout_ic in
    let errors = read_all stderr_ic in
    let status = Unix.close_process_full (stdout_ic, stdin_oc, stderr_ic) in
    match status with
    | Unix.WEXITED 0 -> output
    | _ ->
        let msg = String.trim errors in
        let msg = if msg = "" then String.trim output else msg in
        let msg = if msg = "" then "command failed" else msg in
        raise (RuntimeError ("mysql: " ^ msg))
  in
  env
  |> StringMap.add "mysqlConnect" (VPrim (fun args ->
      (* Accept a config dict or build from env vars *)
      match List.hd args with
      | VDict _ as d -> d
      | VInt 0 | VNil ->
          (* Build from environment variables *)
          VDict [
            ("host", VString (match Sys.getenv_opt "MYSQL_HOST" with Some s -> s | None -> "127.0.0.1"));
            ("port", VString (match Sys.getenv_opt "MYSQL_PORT" with Some s -> s | None -> "3306"));
            ("user", VString (match Sys.getenv_opt "MYSQL_USER" with Some s -> s | None -> "root"));
            ("password", VString (match Sys.getenv_opt "MYSQL_PASSWORD" with Some s -> s | None -> ""));
            ("database", VString (match Sys.getenv_opt "MYSQL_DATABASE" with Some s -> s | None -> ""));
          ]
      | _ -> raise (RuntimeError "mysqlConnect: expected config dict or 0 for env vars")))
  |> StringMap.add "mysqlQuery" (VPrim (fun args ->
      let conn = List.hd args in
      VPrim (fun args ->
        let sql = expect_string "mysqlQuery" (List.hd args) in
        let output = run_mysql conn sql in
        parse_mysql_output output)))
  |> StringMap.add "mysqlExec" (VPrim (fun args ->
      let conn = List.hd args in
      VPrim (fun args ->
        let sql = expect_string "mysqlExec" (List.hd args) in
        let _ = run_mysql conn sql in
        VBool true)))
  |> StringMap.add "mysqlClose" (VPrim (fun _ -> VBool true))

let create_io_functions env =
  let expect_string name = function
    | VString s -> s
    | _ -> raise (RuntimeError (name ^ ": expected string"))
  in
  env
  |> StringMap.add "readFile" (VPrim (fun args ->
      let path = expect_string "readFile" (List.hd args) in
      try
        let ic = open_in path in
        let len = in_channel_length ic in
        let content = really_input_string ic len in
        close_in ic;
        VString content
      with Sys_error msg -> raise (RuntimeError ("readFile: " ^ msg))))
  |> StringMap.add "writeFile" (VPrim (fun args ->
      let path = expect_string "writeFile" (List.hd args) in
      VPrim (fun args ->
        let content = expect_string "writeFile" (List.hd args) in
        try
          let oc = open_out path in
          output_string oc content;
          close_out oc;
          VBool true
        with Sys_error msg -> raise (RuntimeError ("writeFile: " ^ msg)))))
  |> StringMap.add "appendFile" (VPrim (fun args ->
      let path = expect_string "appendFile" (List.hd args) in
      VPrim (fun args ->
        let content = expect_string "appendFile" (List.hd args) in
        try
          let oc = open_out_gen [Open_append; Open_creat] 0o644 path in
          output_string oc content;
          close_out oc;
          VBool true
        with Sys_error msg -> raise (RuntimeError ("appendFile: " ^ msg)))))
  |> StringMap.add "fileExists" (VPrim (fun args ->
      let path = expect_string "fileExists" (List.hd args) in
      VBool (Sys.file_exists path)))
  |> StringMap.add "deleteFile" (VPrim (fun args ->
      let path = expect_string "deleteFile" (List.hd args) in
      try Sys.remove path; VBool true
      with Sys_error msg -> raise (RuntimeError ("deleteFile: " ^ msg))))
  |> StringMap.add "readLines" (VPrim (fun args ->
      let path = expect_string "readLines" (List.hd args) in
      try
        let ic = open_in path in
        let rec read_all acc =
          try read_all (input_line ic :: acc)
          with End_of_file -> close_in ic; List.rev acc
        in
        VList (List.map (fun s -> VString s) (read_all []))
      with Sys_error msg -> raise (RuntimeError ("readLines: " ^ msg))))
  |> StringMap.add "writeLines" (VPrim (fun args ->
      let path = expect_string "writeLines" (List.hd args) in
      VPrim (fun args ->
        let lines = match List.hd args with
          | VList vs -> List.map (fun v -> match v with
              | VString s -> s
              | _ -> raise (RuntimeError "writeLines: expected list of strings")) vs
          | _ -> raise (RuntimeError "writeLines: expected list")
        in
        try
          let oc = open_out path in
          List.iter (fun s -> output_string oc s; output_char oc '\n') lines;
          close_out oc;
          VBool true
        with Sys_error msg -> raise (RuntimeError ("writeLines: " ^ msg)))))

(* List primitives *)
(* Forward reference for eval — set by eval_with_imports at startup *)
let eval_ref : (env -> expr -> env * value) ref = ref (fun _ _ -> failwith "eval not initialized")
let force_ref : (value -> value) ref = ref (fun v -> v)

let apply_fn f arg =
  match f with
  | VPrim fn -> fn [arg]
  | VFun (x::xs, body, closure) ->
      let closure' = StringMap.add x arg closure in
      if xs = [] then !force_ref (snd (!eval_ref closure' body))
      else VFun (xs, body, closure')
  | VRecFun (name, x::xs, body, closure) ->
      let self = VRecFun (name, x::xs, body, closure) in
      let closure' = StringMap.add name self (StringMap.add x arg closure) in
      if xs = [] then !force_ref (snd (!eval_ref closure' body))
      else VFun (xs, body, closure')
  | _ -> raise (RuntimeError "apply: expected function")

let create_list_functions env =
  let expect_list name = function
    | VList vs -> vs
    | VNil -> []
    | _ -> raise (RuntimeError (name ^ ": expected list"))
  in
  let expect_int name = function
    | VInt n -> n
    | _ -> raise (RuntimeError (name ^ ": expected int"))
  in
  env
  |> StringMap.add "nil" VNil
  |> StringMap.add "cons" (VPrim (fun args ->
      let h = List.hd args in
      VPrim (fun args ->
        match List.hd args with
        | VList vs -> VList (h :: vs)
        | VNil -> VList [h]
        | t -> VCons (h, t))))
  |> StringMap.add "head" (VPrim (fun args ->
      match expect_list "head" (List.hd args) with
      | x :: _ -> x
      | [] -> raise (RuntimeError "head: empty list")))
  |> StringMap.add "tail" (VPrim (fun args ->
      match expect_list "tail" (List.hd args) with
      | _ :: xs -> VList xs
      | [] -> raise (RuntimeError "tail: empty list")))
  |> StringMap.add "empty" (VPrim (fun args ->
      VBool (expect_list "empty" (List.hd args) = [])))
  |> StringMap.add "len" (VPrim (fun args ->
      VInt (List.length (expect_list "len" (List.hd args)))))
  |> StringMap.add "nth" (VPrim (fun args ->
      let l = expect_list "nth" (List.hd args) in
      VPrim (fun args ->
        let n = expect_int "nth" (List.hd args) in
        if n < 0 || n >= List.length l then raise (RuntimeError "nth: index out of bounds")
        else List.nth l n)))
  |> StringMap.add "reverse" (VPrim (fun args ->
      VList (List.rev (expect_list "reverse" (List.hd args)))))
  |> StringMap.add "range" (VPrim (fun args ->
      let lo = expect_int "range" (List.hd args) in
      VPrim (fun args ->
        let hi = expect_int "range" (List.hd args) in
        let rec build i acc = if i < lo then acc else build (i-1) (VInt i :: acc) in
        VList (build (hi - 1) []))))
  |> StringMap.add "map" (VPrim (fun args ->
      let f = List.hd args in
      VPrim (fun args ->
        let l = expect_list "map" (List.hd args) in
        VList (List.map (apply_fn f) l))))
  (* attempt: run a thunk, catching a runtime/assertion error into a Result
     (["ok", v] / ["err", msg]). Replaces try/catch — errors become first-class
     values, composed with matchResult. *)
  |> StringMap.add "attempt" (VPrim (fun args ->
      let thunk = List.hd args in
      try VList [VString "ok"; apply_fn thunk (VInt 0)]
      with RuntimeError msg | AssertionFailure msg -> VList [VString "err"; VString msg]))
  |> StringMap.add "filter" (VPrim (fun args ->
      let f = List.hd args in
      VPrim (fun args ->
        let l = expect_list "filter" (List.hd args) in
        VList (List.filter (fun v ->
          match apply_fn f v with
          | VBool b -> b
          | _ -> raise (RuntimeError "filter: predicate must return bool")
        ) l))))
  |> StringMap.add "foldl" (VPrim (fun args ->
      let f = List.hd args in
      VPrim (fun args ->
        let init = List.hd args in
        VPrim (fun args ->
          let l = expect_list "foldl" (List.hd args) in
          List.fold_left (fun acc v -> apply_fn (apply_fn f acc) v) init l))))
  |> StringMap.add "foldr" (VPrim (fun args ->
      let f = List.hd args in
      VPrim (fun args ->
        let init = List.hd args in
        VPrim (fun args ->
          let l = expect_list "foldr" (List.hd args) in
          List.fold_right (fun v acc -> apply_fn (apply_fn f v) acc) l init))))
  (* Church-style eliminators: matchList value on_empty on_cons *)
  |> StringMap.add "matchList" (VPrim (fun args ->
      let l = List.hd args in
      VPrim (fun args ->
        let on_empty = List.hd args in
        VPrim (fun args ->
          let on_cons = List.hd args in
          match l with
          | VList [] | VNil -> apply_fn on_empty VNil
          | VList (h :: t) -> apply_fn (apply_fn on_cons h) (VList t)
          | _ -> raise (RuntimeError "matchList: expected list")))))
  (* matchBool: matchBool cond on_true on_false — Scott-encoded boolean eliminator *)
  |> StringMap.add "matchBool" (VPrim (fun args ->
      match List.hd args with
      | VBool true -> VPrim (fun args -> let t = List.hd args in VPrim (fun _ -> t))
      | VBool false -> VPrim (fun _ -> VPrim (fun args -> List.hd args))
      | _ -> raise (RuntimeError "matchBool: expected bool")))

(* Helper for binary numeric operations with type coercion *)
let eval_numeric_binop va vb int_op long_op float_op name =
  match va, vb with
  | VInt x, VInt y -> VInt (int_op x y)
  | VLong x, VLong y -> VLong (long_op x y)
  | VInt x, VLong y -> VLong (long_op (Int64.of_int x) y)
  | VLong x, VInt y -> VLong (long_op x (Int64.of_int y))
  | VFloat x, VFloat y -> VFloat (float_op x y)
  | VInt x, VFloat y -> VFloat (float_op (float_of_int x) y)
  | VFloat x, VInt y -> VFloat (float_op x (float_of_int y))
  | _ -> raise (RuntimeError ("Type error in " ^ name))

let eval_div_values va vb =
  let check_zero_int y = if y = 0 then raise (RuntimeError "Division by zero") in
  let check_zero_long y = if y = 0L then raise (RuntimeError "Division by zero") in
  let check_zero_float y = if y = 0.0 then raise (RuntimeError "Division by zero") in
  match va, vb with
  | VInt _, VInt 0 | VInt _, VLong 0L -> raise (RuntimeError "Division by zero")
  | VLong _, VInt 0 | VLong _, VLong 0L -> raise (RuntimeError "Division by zero")
  | VInt x, VInt y -> check_zero_int y; VFloat (float_of_int x /. float_of_int y)
  | VLong x, VLong y -> check_zero_long y; VFloat (Int64.to_float x /. Int64.to_float y)
  | VInt x, VLong y -> check_zero_long y; VFloat (float_of_int x /. Int64.to_float y)
  | VLong x, VInt y -> check_zero_int y; VFloat (Int64.to_float x /. float_of_int y)
  | VFloat x, VFloat y -> check_zero_float y; VFloat (x /. y)
  | VInt x, VFloat y -> check_zero_float y; VFloat (float_of_int x /. y)
  | VFloat x, VInt y -> check_zero_int y; VFloat (x /. float_of_int y)
  | _ -> raise (RuntimeError "Type error in div")

(* Pattern matching — returns Some bindings or None *)
let rec match_pattern (pat : Ast.pattern) (v : value) : (string * value) list option =
  match pat, v with
  | Ast.PWild, _ -> Some []
  | Ast.PVar x, _ -> Some [(x, v)]
  | Ast.PInt n, VInt m when n = m -> Some []
  | Ast.PStr s, VString t when s = t -> Some []
  | Ast.PBool b, VBool c when b = c -> Some []
  | Ast.PList pats, VList vs ->
      if List.length pats <> List.length vs then None
      else
        let rec zip ps vs acc =
          match ps, vs with
          | [], [] -> Some acc
          | p :: ps', v :: vs' ->
              (match match_pattern p v with
               | Some binds -> zip ps' vs' (acc @ binds)
               | None -> None)
          | _ -> None
        in
        zip pats vs []
  | Ast.PList [], VNil -> Some []
  | Ast.PCons (hp, tp), VList (h :: t) ->
      (match match_pattern hp h with
       | Some hb ->
           (match match_pattern tp (VList t) with
            | Some tb -> Some (hb @ tb)
            | None -> None)
       | None -> None)
  | Ast.PCons (hp, tp), VCons (h, t) ->
      (match match_pattern hp h with
       | Some hb ->
           (match match_pattern tp t with
            | Some tb -> Some (hb @ tb)
            | None -> None)
       | None -> None)
  | _ -> None

(* Define an eval_with_imports function to handle environment properly *)
(* eval_with_imports and force are mutually recursive for tail call optimization *)
let rec eval_with_imports env expr =
  match expr with
  | Import lib_name ->
      if StringMap.mem lib_name !imported_libraries then
        (env, VInt 0) (* Return unchanged env and dummy value *)
      else
      (* Native primitives for known libraries *)
      let env_with_natives =
        match lib_name with
        | "operators" -> create_church_booleans env
        | "math" -> create_math_functions env
        | "array" -> create_array_functions env
        | "tensor" -> create_tensor_functions env
        | "string" -> create_string_functions env
        | "list" -> create_list_functions env
        | "time" -> create_time_functions env
        | "io" -> create_io_functions env
        | "dict" -> create_dict_functions env
        | "json" -> create_json_functions env
        | "mysql" -> create_mysql_functions env
        | _ -> env
      in
      imported_libraries := StringMap.add lib_name true !imported_libraries;
      (* Then load .ch file from src/lib/ if it exists *)
      let final_env =
        try
          let lib_exprs = load_library lib_name in
          List.fold_left
            (fun acc_env expr ->
              match expr with
              | FunDef (name, [], body) ->
                  (* Zero-arg: evaluate body and bind as a value *)
                  let (_, value) = eval_with_imports acc_env body in
                  StringMap.add name (force value) acc_env
              | FunDef (name, args, body) ->
                  let func_val = VRecFun (name, args, body, acc_env) in
                  StringMap.add name func_val acc_env
              | Let (name, value_expr) ->
                  let (_, value) = eval_with_imports acc_env value_expr in
                  StringMap.add name (force value) acc_env
              | Import _ ->
                  let (new_env, _) = eval_with_imports acc_env expr in
                  new_env
              | _ ->
                  let (_, v) = eval_with_imports acc_env expr in
                  ignore (force v);
                  acc_env
            ) env_with_natives lib_exprs
        with RuntimeError _ -> env_with_natives
      in
      (final_env, VInt 0)
  | Int n -> (env, VInt n)
  | Lng n -> (env, VLong n)
  | Float f -> (env, VFloat f)
  | Str s -> (env, VString s)
  | Bool b -> (env, VBool b)
  | List items ->
      let env', vs = List.fold_left (fun (e, acc) item ->
        let e', v = eval_with_imports e item in
        (e', acc @ [force v])
      ) (env, []) items in
      (env', VList vs)
  | Dict entries ->
      let env', pairs = List.fold_left (fun (e, acc) (key, expr) ->
        let e', v = eval_with_imports e expr in
        (e', acc @ [(key, force v)])
      ) (env, []) entries in
      (env', VDict pairs)
  | Add (a, b) -> eval_binop env a b ( + ) Int64.add ( +. ) "add"
  | Sub (a, b) -> eval_binop env a b ( - ) Int64.sub ( -. ) "sub"
  | Mul (a, b) -> eval_binop env a b ( * ) Int64.mul ( *. ) "mul"
  | Div (a, b) -> eval_div env a b
  | Eq (a, b) ->
      let (env', va) = eval_with_imports env a in
      let va = force va in
      let (env'', vb) = eval_with_imports env' b in
      let vb = force vb in
      let rec values_equal a b =
        match a, b with
        | VInt x, VInt y -> x = y
        | VFloat x, VFloat y -> x = y
        | VInt x, VFloat y -> float_of_int x = y
        | VFloat x, VInt y -> x = float_of_int y
        | VLong x, VLong y -> x = y
        | VInt x, VLong y -> Int64.of_int x = y
        | VLong x, VInt y -> x = Int64.of_int y
        | VLong x, VFloat y -> Int64.to_float x = y
        | VFloat x, VLong y -> x = Int64.to_float y
        | VString x, VString y -> x = y
        | VBool x, VBool y -> x = y
        | VList xs, VList ys ->
            List.length xs = List.length ys && List.for_all2 values_equal xs ys
        | VCons (h1, t1), VCons (h2, t2) ->
            values_equal h1 h2 && values_equal t1 t2
        | VNil, VNil -> true
        | VArray a, VArray b ->
            Array.length a = Array.length b &&
            (let ok = ref true in
             Array.iteri (fun i v ->
               if !ok && not (values_equal v b.(i)) then ok := false) a;
             !ok)
        | VDict a, VDict b ->
            List.length a = List.length b &&
            List.for_all (fun (k, v) ->
              match List.assoc_opt k b with
              | Some v2 -> values_equal v v2
              | None -> false) a
        | _ -> false
      in
      let result = values_equal va vb in
      (env'', VBool result)
  | Var x ->
      (try (env, StringMap.find x env)
       with Not_found -> raise (RuntimeError ("Unbound variable: " ^ x)))
  | Lam (x, body) -> (env, VFun ([x], body, env))
  | App (f, a) ->
      let (env', vf) = eval_with_imports env f in
      let vf = force vf in
      let (env'', va) = eval_with_imports env' a in
      let va = force va in
      (env'',
       match vf with
       | VFun ([], body, closure) ->
           (* Tail call: defer body evaluation *)
           VTailCall (closure, body)
       | VFun (x::xs, body, closure) ->
           let closure' = StringMap.add x va closure in
           if xs = [] then
             (* Tail call: defer body evaluation *)
             VTailCall (closure', body)
           else VFun (xs, body, closure')
       | VRecFun (name, x::xs, body, closure) ->
           (* Bind the function's own name in its closure for recursion *)
           let self = VRecFun (name, x::xs, body, closure) in
           let closure' = StringMap.add name self (StringMap.add x va closure) in
           if xs = [] then
             (* Tail call: defer body evaluation *)
             VTailCall (closure', body)
           else VFun (xs, body, closure')
       | VBool b ->
           (* Booleans are applicable: true t f = t, false t f = f *)
           let first_arg = va in
           VPrim (fun args -> if b then first_arg else List.hd args)
       | VPrim fn -> fn [va]
       | _ -> raise (RuntimeError "Attempt to call a non-function"))
  | Let (name, value) ->
      let (env', v) = eval_with_imports env value in
      let v = force v in
      (StringMap.add name v env', v)
  | FunDef (name, args, body) ->
      let f = if args = [] then VFun (args, body, env)
              else VRecFun (name, args, body, env) in
      let env' = StringMap.add name f env in
      (env', f)
  | Seq (a, b) ->
      let (env', va) = eval_with_imports env a in
      ignore (force va); (* Force first expression for side effects *)
      eval_with_imports env' b
  | Assert e ->
      let (env', v) = eval_with_imports env e in
      let v = force v in
      (match v with
       | VBool true -> (env', VBool true)
       | VBool false ->
           raise (AssertionFailure ("Assertion failed: " ^ Ast.string_of_expr e))
       | _ ->
           raise (AssertionFailure ("Assert requires a boolean, got: " ^ Ast.string_of_expr e)))
  | Match (scrutinee, arms) ->
      let (env', sv) = eval_with_imports env scrutinee in
      let sv = force sv in
      let rec try_arms = function
        | [] -> raise (RuntimeError ("No matching pattern for: " ^ string_of_value sv))
        | (pat, body) :: rest ->
            (match match_pattern pat sv with
             | Some bindings ->
                 let env'' = List.fold_left (fun e (k, v) -> StringMap.add k v e) env' bindings in
                 eval_with_imports env'' body
             | None -> try_arms rest)
      in
      try_arms arms

(* Trampoline: resolve VTailCall chains without growing the stack.
   force is tail-recursive, so OCaml optimizes it into a loop. *)
and eval_binop env a b int_op long_op float_op name =
  let (env', va) = eval_with_imports env a in
  let va = force va in
  let (env'', vb) = eval_with_imports env' b in
  let vb = force vb in
  (env'', eval_numeric_binop va vb int_op long_op float_op name)

and eval_div env a b =
  let (env', va) = eval_with_imports env a in
  let va = force va in
  let (env'', vb) = eval_with_imports env' b in
  let vb = force vb in
  (env'', eval_div_values va vb)

and force = function
  | VTailCall (tc_env, tc_expr) ->
      force (snd (eval_with_imports tc_env tc_expr))
  | v -> v

(* Initialize forward references for list operations *)
let () = eval_ref := eval_with_imports
let () = force_ref := force

(* Simple wrapper around eval_with_imports to keep original interface *)
let eval env expr =
  let (_, value) = eval_with_imports env expr in
  force value

(* Evaluate a top-level expression, returning updated env and optional last value *)
let rec eval_toplevel (env : env) (expr : expr) : env * value option =
  match expr with
  | Seq (a, b) ->
      let env', _ = eval_toplevel env a in
      eval_toplevel env' b
  | Let (name, value) ->
      let v = eval env value in
      (StringMap.add name v env, None)
  | FunDef ("main", [], body) ->
      (* ~() main block: evaluate body directly *)
      let v = eval env body in
      (env, Some v)
  | FunDef (name, args, body) ->
      let f = if args = [] then VFun (args, body, env)
              else VRecFun (name, args, body, env) in
      (StringMap.add name f env, None)
  | Import lib_name ->
      let (new_env, _) = eval_with_imports env (Import lib_name) in
      (new_env, None)
  | _ ->
      let v = eval env expr in
      (env, Some v)

let load_prelude () =
  let stdlib = ["operators"; "math"; "array"; "tensor"; "string"; "list"; "time"; "io"; "dict"; "json"; "mysql"; "church_list"; "result"; "vector"; "matrix"; "activations"; "loss"; "pgm"; "nn"] in
  List.fold_left (fun env lib ->
    fst (eval_with_imports env (Import lib))
  ) StringMap.empty stdlib

let rec last_expr = function
  | [] -> None
  | [x] -> Some x
  | _ :: rest -> last_expr rest

let should_print_result exprs =
  match last_expr exprs with
  | None -> false
  | Some (Assert _) -> false
  | Some (FunDef _) -> false
  | Some (Let _) -> false
  | Some (Import _) -> false
  | Some (Seq (_, b)) -> (match b with Assert _ | FunDef _ | Let _ | Import _ -> false | _ -> true)
  | Some _ -> true

let eval_program exprs =
  imported_libraries := StringMap.empty;
  try
    let _env, last_value = List.fold_left (fun (env, _last) expr ->
      eval_toplevel env expr
    ) (load_prelude (), None) exprs in
    (* Auto-print last value only if the last expression is a value expression *)
    if should_print_result exprs then
      (match last_value with
       | Some v -> print_value v
       | None -> ())
  with
  | Division_by_zero ->
      raise (RuntimeError "Division by zero")
