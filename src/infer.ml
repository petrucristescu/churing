open Types
open Ast

exception TypeError of string

module StringMap = Map.Make(String)

type subst = (string * typ) list

let rec apply subst t =
  match t with
  | TVar v ->
      (try List.assoc v subst with Not_found -> t)
  | TFun (a, b) -> TFun (apply subst a, apply subst b)
  | TList a -> TList (apply subst a)
  | TDict a -> TDict (apply subst a)
  | TArray a -> TArray (apply subst a)
  | _ -> t

let compose s1 s2 =
  List.map (fun (v, t) -> (v, apply s1 t)) s2 @ s1

let counter = ref 0
let reset_counter () = counter := 0
let fresh_var () =
  let v = Printf.sprintf "a%d" !counter in
  incr counter; TVar v

let fresh_var_name () =
  let v = Printf.sprintf "a%d" !counter in
  incr counter; v

let rec occurs v = function
  | TVar v' -> v = v'
  | TFun (a, b) -> occurs v a || occurs v b
  | TList a -> occurs v a
  | TDict a -> occurs v a
  | TArray a -> occurs v a
  | _ -> false

let rec unify t1 t2 =
  match t1, t2 with
  | TInt, TInt | TLong, TLong | TFloat, TFloat | TString, TString | TBool, TBool -> []
  | TVar v, t | t, TVar v ->
      if t = TVar v then [] else
      if occurs v t then raise (TypeError ("Occurs check: " ^ v))
      else [ (v, t) ]
  | TFun (a1, b1), TFun (a2, b2) ->
      let s1 = unify a1 a2 in
      let s2 = unify (apply s1 b1) (apply s1 b2) in
      compose s2 s1
  | TList a1, TList a2 -> unify a1 a2
  | TDict a1, TDict a2 -> unify a1 a2
  | TArray a1, TArray a2 -> unify a1 a2
  | TTensor, TTensor -> []
  | _ -> raise (TypeError ("Cannot unify types: " ^ Ast.string_of_typ t1 ^ " and " ^ Ast.string_of_typ t2))

(* Scheme helpers *)
let mono t = Forall ([], t)

let apply_scheme s (Forall (vars, t)) =
  let s' = List.filter (fun (v, _) -> not (List.mem v vars)) s in
  Forall (vars, apply s' t)

let apply_env s env = StringMap.map (apply_scheme s) env

let free_vars_typ t =
  let rec aux acc = function
    | TVar v -> if List.mem v acc then acc else v :: acc
    | TFun (a, b) -> aux (aux acc a) b
    | TList a -> aux acc a
    | TDict a -> aux acc a
    | TArray a -> aux acc a
    | _ -> acc
  in aux [] t

let free_vars_scheme (Forall (vars, t)) =
  List.filter (fun v -> not (List.mem v vars)) (free_vars_typ t)

let free_vars_env env =
  StringMap.fold (fun _ s acc ->
    List.fold_left (fun acc v -> if List.mem v acc then acc else v :: acc)
      acc (free_vars_scheme s)
  ) env []

let generalize env t =
  let env_vars = free_vars_env env in
  let t_vars = free_vars_typ t in
  let vars = List.filter (fun v -> not (List.mem v env_vars)) t_vars in
  Forall (vars, t)

let instantiate (Forall (vars, t)) =
  let subst = List.map (fun v -> (v, TVar (fresh_var_name ()))) vars in
  let rec aux = function
    | TVar v as tv -> (try List.assoc v subst with Not_found -> tv)
    | TFun (a, b) -> TFun (aux a, aux b)
    | TList a -> TList (aux a)
    | TDict a -> TDict (aux a)
    | TArray a -> TArray (aux a)
    | t -> t
  in aux t

let add_operators_to_env env =
  let t_bool = TBool in
  let env = StringMap.add "true" (mono t_bool) env in
  let env = StringMap.add "false" (mono t_bool) env in

  let t_if =
    let a = fresh_var () in
    TFun (t_bool, TFun (a, TFun (a, a)))
  in
  let env = StringMap.add "if" (mono t_if) env in

  let t_not = TFun (t_bool, t_bool) in
  let env = StringMap.add "not" (mono t_not) env in

  let t_and = TFun (t_bool, TFun (t_bool, t_bool)) in
  let env = StringMap.add "and" (mono t_and) env in

  let t_or = TFun (t_bool, TFun (t_bool, t_bool)) in
  let env = StringMap.add "or" (mono t_or) env in

  (* Environment variable access *)
  let print_a = fresh_var () in
  let env = StringMap.add "print" (mono (TFun (print_a, print_a))) env in
  let env = StringMap.add "env" (mono (TFun (TString, TString))) env in
  let env_a = fresh_var () in
  let env = StringMap.add "envOr" (mono (TFun (TString, TFun (env_a, env_a)))) env in

  let rand_a = fresh_var () in
  let env = StringMap.add "random" (mono (TFun (rand_a, TFloat))) env in

  (* Comparison operators — polymorphic: a -> a -> Bool *)
  let cmp_a = fresh_var () in
  let t_cmp = TFun (cmp_a, TFun (cmp_a, t_bool)) in
  let env = StringMap.add "gt" (mono t_cmp) env in
  let cmp_b = fresh_var () in
  let env = StringMap.add "lt" (mono (TFun (cmp_b, TFun (cmp_b, t_bool)))) env in
  let cmp_c = fresh_var () in
  let env = StringMap.add "gte" (mono (TFun (cmp_c, TFun (cmp_c, t_bool)))) env in
  let cmp_d = fresh_var () in
  let env = StringMap.add "lte" (mono (TFun (cmp_d, TFun (cmp_d, t_bool)))) env in

  (* Math functions *)
  let t_float_float = TFun (TFloat, TFloat) in
  let t_float_int = TFun (TFloat, TInt) in
  let t_float_float_float = TFun (TFloat, TFun (TFloat, TFloat)) in
  let env = StringMap.add "sqrt" (mono t_float_float) env in
  let env = StringMap.add "sin" (mono t_float_float) env in
  let env = StringMap.add "cos" (mono t_float_float) env in
  let env = StringMap.add "tan" (mono t_float_float) env in
  let env = StringMap.add "asin" (mono t_float_float) env in
  let env = StringMap.add "acos" (mono t_float_float) env in
  let env = StringMap.add "atan" (mono t_float_float) env in
  let env = StringMap.add "floor" (mono t_float_int) env in
  let env = StringMap.add "ceil" (mono t_float_int) env in
  let env = StringMap.add "round" (mono t_float_int) env in
  let env = StringMap.add "abs" (mono t_float_float) env in
  let env = StringMap.add "pow" (mono t_float_float_float) env in
  let env = StringMap.add "exp" (mono t_float_float) env in
  let env = StringMap.add "log" (mono t_float_float) env in
  let env = StringMap.add "tanh" (mono t_float_float) env in
  let env = StringMap.add "min" (mono t_float_float_float) env in
  let env = StringMap.add "max" (mono t_float_float_float) env in

  (* String functions *)
  let t_str = TString in
  let t_int = TInt in
  let env = StringMap.add "length" (mono (TFun (t_str, t_int))) env in
  let env = StringMap.add "concat" (mono (TFun (t_str, TFun (t_str, t_str)))) env in
  let env = StringMap.add "substring" (mono (TFun (t_str, TFun (t_int, TFun (t_int, t_str))))) env in
  let env = StringMap.add "uppercase" (mono (TFun (t_str, t_str))) env in
  let env = StringMap.add "lowercase" (mono (TFun (t_str, t_str))) env in
  let env = StringMap.add "trim" (mono (TFun (t_str, t_str))) env in
  let env = StringMap.add "charAt" (mono (TFun (t_str, TFun (t_int, t_str)))) env in
  let env = StringMap.add "indexOf" (mono (TFun (t_str, TFun (t_str, t_int)))) env in
  let env = StringMap.add "startsWith" (mono (TFun (t_str, TFun (t_str, t_bool)))) env in
  let env = StringMap.add "endsWith" (mono (TFun (t_str, TFun (t_str, t_bool)))) env in
  let env = StringMap.add "replace" (mono (TFun (t_str, TFun (t_str, TFun (t_str, t_str))))) env in
  let env = StringMap.add "split" (mono (TFun (t_str, TFun (t_str, TList t_str)))) env in
  let toFloat_a = fresh_var () in
  let env = StringMap.add "toFloat" (mono (TFun (toFloat_a, TFloat))) env in
  let toInt_a = fresh_var () in
  let env = StringMap.add "toInt" (mono (TFun (toInt_a, TInt))) env in
  let a = fresh_var () in
  let env = StringMap.add "toString" (mono (TFun (a, t_str))) env in
  let str_a = fresh_var () in
  let env = StringMap.add "str" (mono (TFun (TList str_a, t_str))) env in
  let join_a = fresh_var () in
  let env = StringMap.add "join" (mono (TFun (t_str, TFun (TList join_a, t_str)))) env in

  (* Date/Time functions *)
  let t_dummy = fresh_var () in
  let env = StringMap.add "now" (mono (TFun (t_dummy, TInt))) env in
  let t_dummy2 = fresh_var () in
  let env = StringMap.add "timeMs" (mono (TFun (t_dummy2, TInt))) env in
  let env = StringMap.add "year" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "month" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "day" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "hour" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "minute" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "second" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "dayOfWeek" (mono (TFun (TInt, TInt))) env in
  let env = StringMap.add "diffTime" (mono (TFun (TInt, TFun (TInt, TInt)))) env in

  (* File I/O functions *)
  let env = StringMap.add "readFile" (mono (TFun (TString, TString))) env in
  let env = StringMap.add "writeFile" (mono (TFun (TString, TFun (TString, TBool)))) env in
  let env = StringMap.add "appendFile" (mono (TFun (TString, TFun (TString, TBool)))) env in
  let env = StringMap.add "fileExists" (mono (TFun (TString, TBool))) env in
  let env = StringMap.add "deleteFile" (mono (TFun (TString, TBool))) env in
  let env = StringMap.add "readLines" (mono (TFun (TString, TList TString))) env in
  let env = StringMap.add "writeLines" (mono (TFun (TString, TFun (TList TString, TBool)))) env in

  (* Dict operations *)
  let d1 = fresh_var () in
  let env = StringMap.add "get" (mono (TFun (TDict d1, TFun (TString, d1)))) env in
  let d2 = fresh_var () in
  let env = StringMap.add "set" (mono (TFun (TDict d2, TFun (TString, TFun (d2, TDict d2))))) env in
  let d3 = fresh_var () in
  let env = StringMap.add "has" (mono (TFun (TDict d3, TFun (TString, TBool)))) env in
  let d4 = fresh_var () in
  let env = StringMap.add "keys" (mono (TFun (TDict d4, TList TString))) env in
  let d5 = fresh_var () in
  let env = StringMap.add "values" (mono (TFun (TDict d5, TList d5))) env in
  let d6 = fresh_var () in
  let env = StringMap.add "merge" (mono (TFun (TDict d6, TFun (TDict d6, TDict d6)))) env in
  let d7 = fresh_var () in
  let env = StringMap.add "remove" (mono (TFun (TDict d7, TFun (TString, TDict d7)))) env in
  let d8 = fresh_var () in
  let env = StringMap.add "entries" (mono (TFun (TDict d8, TList (TList d8)))) env in
  let d9 = fresh_var () in
  let env = StringMap.add "fromEntries" (mono (TFun (TList (TList d9), TDict d9))) env in

  (* JSON functions *)
  let json_a = fresh_var () in
  let env = StringMap.add "toJson" (mono (TFun (json_a, TString))) env in
  let json_b = fresh_var () in
  let env = StringMap.add "fromJson" (mono (TFun (TString, json_b))) env in

  (* MySQL functions *)
  let mysql_conn = fresh_var () in
  let env = StringMap.add "mysqlConnect" (mono (TFun (mysql_conn, TDict TString))) env in
  let mysql_v = fresh_var () in
  let env = StringMap.add "mysqlQuery" (mono (TFun (TDict TString, TFun (TString, TList (TDict mysql_v))))) env in
  let env = StringMap.add "mysqlExec" (mono (TFun (TDict TString, TFun (TString, TBool)))) env in
  let mysql_c = fresh_var () in
  let env = StringMap.add "mysqlClose" (mono (TFun (mysql_c, TBool))) env in

  (* List operations — polymorphic via fresh vars *)
  let a1 = fresh_var () in
  let env = StringMap.add "cons" (mono (TFun (a1, TFun (TList a1, TList a1)))) env in
  let a2 = fresh_var () in
  let env = StringMap.add "head" (mono (TFun (TList a2, a2))) env in
  let a3 = fresh_var () in
  let env = StringMap.add "tail" (mono (TFun (TList a3, TList a3))) env in
  let a4 = fresh_var () in
  let env = StringMap.add "empty" (mono (TFun (TList a4, t_bool))) env in
  let a5 = fresh_var () in
  let env = StringMap.add "len" (mono (TFun (TList a5, TInt))) env in

  (* Higher-order list operations *)
  let a6 = fresh_var () in
  let b6 = fresh_var () in
  let env = StringMap.add "map" (mono (TFun (TFun (a6, b6), TFun (TList a6, TList b6)))) env in
  let a7 = fresh_var () in
  let env = StringMap.add "filter" (mono (TFun (TFun (a7, t_bool), TFun (TList a7, TList a7)))) env in
  let a8 = fresh_var () in
  let b8 = fresh_var () in
  let env = StringMap.add "foldl" (mono (TFun (TFun (b8, TFun (a8, b8)), TFun (b8, TFun (TList a8, b8))))) env in
  let a9 = fresh_var () in
  let b9 = fresh_var () in
  let env = StringMap.add "foldr" (mono (TFun (TFun (a9, TFun (b9, b9)), TFun (b9, TFun (TList a9, b9))))) env in
  let a10 = fresh_var () in
  let env = StringMap.add "nth" (mono (TFun (TList a10, TFun (TInt, a10)))) env in
  let a11 = fresh_var () in
  let env = StringMap.add "reverse" (mono (TFun (TList a11, TList a11))) env in
  let a12 = fresh_var () in
  let env = StringMap.add "range" (mono (TFun (TInt, TFun (TInt, TList a12)))) env in

  (* Cons cell operations *)
  let a13 = fresh_var () in
  let b13 = fresh_var () in
  let env = StringMap.add "pair" (mono (TFun (a13, TFun (b13, TFun (TFun (a13, TFun (b13, a13)), a13))))) env in
  let env = StringMap.add "nil" (mono (TList (fresh_var ()))) env in

  (* Church-style eliminators *)
  let ml_a = fresh_var () in
  let ml_b = fresh_var () in
  (* matchList : [a] -> (unit -> b) -> (a -> [a] -> b) -> b *)
  let env = StringMap.add "matchList" (mono (TFun (TList ml_a,
    TFun (TFun (TList ml_a, ml_b),
    TFun (TFun (ml_a, TFun (TList ml_a, ml_b)), ml_b))))) env in
  let mb_a = fresh_var () in
  let mb_b = fresh_var () in
  (* matchBool : Bool -> a -> a -> a *)
  let env = StringMap.add "matchBool" (mono (TFun (TBool, TFun (mb_a, TFun (mb_b, mb_a))))) env in

  (* Generic array operations — polymorphic over element type *)
  let ar_a = fresh_var () in
  let env = StringMap.add "arrayCreate" (mono (TFun (TInt, TFun (ar_a, TArray ar_a)))) env in
  let ar_b = fresh_var () in
  let env = StringMap.add "arrayGet" (mono (TFun (TArray ar_b, TFun (TInt, ar_b)))) env in
  let ar_c = fresh_var () in
  let env = StringMap.add "arraySet" (mono (TFun (TArray ar_c, TFun (TInt, TFun (ar_c, TArray ar_c))))) env in
  let ar_d = fresh_var () in
  let env = StringMap.add "arrayLength" (mono (TFun (TArray ar_d, TInt))) env in
  let ar_e = fresh_var () in
  let env = StringMap.add "arrayFromList" (mono (TFun (TList ar_e, TArray ar_e))) env in
  let ar_f = fresh_var () in
  let env = StringMap.add "arrayToList" (mono (TFun (TArray ar_f, TList ar_f))) env in

  (* Tensor primitives *)
  let t = TTensor in
  let env = StringMap.add "tensorCreate" (mono (TFun (TInt, TFun (TInt, TFun (TFloat, t))))) env in
  let env = StringMap.add "tensorGet" (mono (TFun (t, TFun (TInt, TFun (TInt, TFloat))))) env in
  let env = StringMap.add "tensorSet" (mono (TFun (t, TFun (TInt, TFun (TInt, TFun (TFloat, t)))))) env in
  let env = StringMap.add "tensorRows" (mono (TFun (t, TInt))) env in
  let env = StringMap.add "tensorCols" (mono (TFun (t, TInt))) env in
  let env = StringMap.add "tensorShape" (mono (TFun (t, TList TInt))) env in
  let env = StringMap.add "tensorVec" (mono (TFun (TList TFloat, t))) env in
  let env = StringMap.add "tensorMat" (mono (TFun (TList (TList TFloat), t))) env in
  let env = StringMap.add "tensorToVec" (mono (TFun (t, TList TFloat))) env in
  let env = StringMap.add "tensorToMat" (mono (TFun (t, TList (TList TFloat)))) env in
  let env = StringMap.add "tensorRandom" (mono (TFun (TInt, TFun (TInt, TFun (TFloat, t))))) env in
  env

let rec infer env = function
  | Int _ -> ([], TInt)
  | Lng _ -> ([], TLong)
  | Float _ -> ([], TFloat)
  | Str _ -> ([], TString)
  | Bool _ -> ([], TBool)
  | List [] -> ([], TList (fresh_var ()))
  | List (x :: xs) ->
      let s0, t0 = infer env x in
      let s, t = List.fold_left (fun (s_acc, t_acc) elem ->
        let s1, t1 = infer (apply_env s_acc env) elem in
        let t_acc' = apply s1 t_acc in
        let s2 = unify t_acc' t1 in
        (compose s2 (compose s1 s_acc), apply s2 t1)
      ) (s0, t0) xs in
      (s, TList t)
  | Dict [] -> ([], TDict (fresh_var ()))
  | Dict ((_, v) :: rest) ->
      let s0, t0 = infer env v in
      let s, t = List.fold_left (fun (s_acc, t_acc) (_, v_expr) ->
        let s1, t1 = infer (apply_env s_acc env) v_expr in
        let t_acc' = apply s1 t_acc in
        let s2 = unify t_acc' t1 in
        (compose s2 (compose s1 s_acc), apply s2 t1)
      ) (s0, t0) rest in
      (s, TDict t)
  | Add (a, b) | Sub (a, b) | Mul(a, b) ->
      let s1, t1 = infer env a in
      let s2, t2 = infer (apply_env s1 env) b in
      let t1 = apply s2 t1 in
      let t2 = apply s2 t2 in
      let s3 = unify t1 t2 in
      (compose s3 (compose s2 s1), apply s3 t1)
  | Div (a, b) ->
      let s1, t1 = infer env a in
      let s2, t2 = infer (apply_env s1 env) b in
      let t1 = apply s2 t1 in
      let t2 = apply s2 t2 in
      let s3 = unify t1 t2 in
      if apply s3 t1 = TFloat || apply s3 t1 = TInt || apply s3 t1 = TLong then
        (compose s3 (compose s2 s1), TFloat)
      else
        (compose s3 (compose s2 s1), apply s3 t1)
  | Eq (a, b) ->
      let s1, t1 = infer env a in
      let s2, t2 = infer (apply_env s1 env) b in
      let s3 = unify (apply s2 t1) t2 in
      (compose s3 (compose s2 s1), TBool)
  | Var x ->
      (match StringMap.find_opt x env with
      | Some scheme -> ([], instantiate scheme)
      | None -> raise (TypeError ("Unbound variable: " ^ x)))
  | Lam (x, Lam (y, Var v)) when v = x ->
      let a = fresh_var () in
      let b = fresh_var () in
      ([], TFun (a, TFun (b, a)))
  | Lam (x, Lam (y, Var v)) when v = y ->
      let a = fresh_var () in
      let b = fresh_var () in
      ([], TFun (a, TFun (b, b)))
  | Lam (x, body) ->
      let tv = fresh_var () in
      let env' = StringMap.add x (mono tv) env in
      let s1, t1 = infer env' body in
      (s1, TFun (apply s1 tv, t1))
  | App (f, a) ->
      let s1, t1 = infer env f in
      let s2, t2 = infer (apply_env s1 env) a in
      let tv = fresh_var () in
      let t1' = apply s2 t1 in
      (* Allow Bool to be applied (Church-style) *)
      let s3 = match t1' with
        | TBool -> unify (TFun (t2, TFun (t2, t2))) (TFun (t2, TFun (t2, t2)))
        | _ -> unify t1' (TFun (t2, tv))
      in
      let result_t = match t1' with
        | TBool ->
            let a = fresh_var () in
            TFun (a, a)
        | _ -> apply s3 tv
      in
      (compose s3 (compose s2 s1), result_t)
  | Let (name, value) ->
      let s1, t1 = infer env value in
      (s1, apply s1 t1)
  | FunDef (name, args, body) ->
      let arg_types = List.map (fun _ -> fresh_var ()) args in
      let ret_type = fresh_var () in
      let fun_type = List.fold_right (fun a b -> TFun (a, b)) arg_types ret_type in
      let env' = List.fold_left2 (fun e a t -> StringMap.add a (mono t) e) env args arg_types in
      let env' = StringMap.add name (mono fun_type) env' in
      let s1, t1 = infer env' body in
      let s2 = unify (apply s1 ret_type) t1 in
      let s = compose s2 s1 in
      let t = List.fold_right (fun a b -> TFun (a, b)) (List.map (apply s) arg_types) (apply s t1) in
      (s, t)
  | Seq (a, b) ->
      let s1, _ = infer env a in
      let s2, t2 = infer (apply_env s1 env) b in
      (compose s2 s1, t2)
  | Assert e ->
      let s1, t1 = infer env e in
      let s2 = unify (apply s1 t1) TBool in
      (compose s2 s1, TBool)
  | Match (scrutinee, arms) ->
      let s0, t_scrut = infer env scrutinee in
      let result_tv = fresh_var () in
      let s, t_result = List.fold_left (fun (s_acc, t_res) (pat, body) ->
        let env' = apply_env s_acc env in
        let s_pat, pat_bindings = infer_pattern pat (apply s_acc t_scrut) in
        let env'' = List.fold_left (fun e (name, t) -> StringMap.add name (mono t) e) env' pat_bindings in
        let s1, t_body = infer env'' body in
        let s2 = unify (apply s1 t_res) t_body in
        (compose s2 (compose s1 (compose s_pat s_acc)), apply s2 t_body)
      ) (s0, result_tv) arms in
      (s, t_result)
  | Import _ ->
      ([], TInt)
  | Llvm _ ->
      ([], fresh_var ())

and infer_pattern pat t_scrut : subst * (string * typ) list =
  match pat with
  | Ast.PWild -> ([], [])
  | Ast.PVar x -> ([], [(x, t_scrut)])
  | Ast.PInt _ -> (unify t_scrut TInt, [])
  | Ast.PStr _ -> (unify t_scrut TString, [])
  | Ast.PBool _ -> (unify t_scrut TBool, [])
  | Ast.PList pats ->
      let elem_tv = fresh_var () in
      let s0 = unify t_scrut (TList elem_tv) in
      let s, bindings = List.fold_left (fun (s_acc, binds) p ->
        let s1, pb = infer_pattern p (apply s_acc elem_tv) in
        (compose s1 s_acc, binds @ pb)
      ) (s0, []) pats in
      (s, bindings)
  | Ast.PCons (hp, tp) ->
      let elem_tv = fresh_var () in
      let s0 = unify t_scrut (TList elem_tv) in
      let s1, hb = infer_pattern hp (apply s0 elem_tv) in
      let s2, tb = infer_pattern tp (apply (compose s1 s0) t_scrut) in
      (compose s2 (compose s1 s0), hb @ tb)
