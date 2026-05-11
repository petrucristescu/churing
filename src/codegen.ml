open Ast

module StringMap = Map.Make(String)
module StringSet = Set.Make(String)

external get_fn_addr : string -> Llvm_executionengine.llexecutionengine -> nativeint
  = "llvm_ee_get_function_address"

external call_f1 : nativeint -> float -> float = "caml_call_f1"
external call_f2 : nativeint -> float -> float -> float = "caml_call_f2"

let jit_initialized = ref false
let lam_counter = ref 0

let ensure_initialized () =
  if not !jit_initialized then begin
    jit_initialized := true;
    ignore (Llvm_executionengine.initialize ());
    Llvm_all_backends.initialize ()
  end

(* All closure function pointers are stored as opaque ptr *)
let ptr_ty ctx = Llvm.pointer_type ctx

(* Closure struct: { ptr fn_ptr, ptr env_ptr } *)
let closure_ty ctx =
  let p = ptr_ty ctx in
  Llvm.struct_type ctx [| p; p |]

let fresh_lam () =
  let n = !lam_counter in
  incr lam_counter;
  Printf.sprintf "_lam_%d" n

(* Compute free variables of expr, excluding names in 'bound' *)
let rec free_vars bound = function
  | Int _ | Float _ | Bool _ | Str _ | Lng _ -> StringSet.empty
  | Var x -> if StringSet.mem x bound then StringSet.empty else StringSet.singleton x
  | Lam (x, body) -> free_vars (StringSet.add x bound) body
  | App (f, a) -> StringSet.union (free_vars bound f) (free_vars bound a)
  | Add (a, b) | Sub (a, b) | Mul (a, b) | Div (a, b) | Eq (a, b) ->
      StringSet.union (free_vars bound a) (free_vars bound b)
  | Let (_, v) -> free_vars bound v
  | Seq (a, b) ->
      let fv_a = free_vars bound a in
      let bound' = match a with Let (n, _) -> StringSet.add n bound | _ -> bound in
      StringSet.union fv_a (free_vars bound' b)
  | Llvm (_, args) ->
      List.fold_left (fun s e -> StringSet.union s (free_vars bound e)) StringSet.empty args
  | _ -> StringSet.empty

(* Flatten App(App(Var f, a1), a2) → Some (f, [a1; a2]) *)
let rec flatten_app acc = function
  | App (inner, arg) -> flatten_app (arg :: acc) inner
  | Var name -> Some (name, acc)
  | _ -> None

(* Get or declare malloc(i64) -> ptr *)
let get_malloc ctx md =
  let p = ptr_ty ctx in
  let ft = Llvm.function_type p [| Llvm.i64_type ctx |] in
  match Llvm.lookup_function "malloc" md with
  | Some f -> f
  | None -> Llvm.declare_function "malloc" ft md

(* Build a closure struct value {fn_ptr, env_ptr} without heap allocation *)
let build_closure ctx builder fn env_ptr =
  let clos_t = closure_ty ctx in
  let p = ptr_ty ctx in
  let fn_p = Llvm.build_bitcast fn p "fn_p" builder in
  let c0 = Llvm.build_insertvalue (Llvm.undef clos_t) fn_p 0 "c0" builder in
  Llvm.build_insertvalue c0 env_ptr 1 "clos" builder

(* Call a closure value: extract fn_ptr and env_ptr, call fn_ptr(arg, env_ptr) *)
let call_closure ctx builder clos_val arg_val =
  let f64 = Llvm.double_type ctx in
  let p = ptr_ty ctx in
  let fn_ptr  = Llvm.build_extractvalue clos_val 0 "fn_ptr"  builder in
  let env_ptr = Llvm.build_extractvalue clos_val 1 "env_ptr" builder in
  let ft = Llvm.function_type f64 [| f64; p |] in
  Llvm.build_call ft fn_ptr [| arg_val; env_ptr |] "app" builder

(* Create a closure wrapping a 1-arg named function f(double) -> double.
   Generates a wrapper _wrap_<name>(double x, ptr _env) -> double if not already present. *)
let wrap_fn_as_closure ctx md builder fn name =
  let f64 = Llvm.double_type ctx in
  let p = ptr_ty ctx in
  let wname = "_wrap_" ^ name in
  let wfn = match Llvm.lookup_function wname md with
    | Some f -> f
    | None ->
        let wft = Llvm.function_type f64 [| f64; p |] in
        let wfn = Llvm.define_function wname wft md in
        let bb = Llvm.entry_block wfn in
        let wb = Llvm.builder_at_end ctx bb in
        let orig_ft = Llvm.function_type f64 [| f64 |] in
        let r = Llvm.build_call orig_ft fn [| (Llvm.params wfn).(0) |] "r" wb in
        ignore (Llvm.build_ret r wb);
        wfn
  in
  build_closure ctx builder wfn (Llvm.const_null p)

(* Compile an expression.
   known_fns maps name → (arity, llvalue) for top-level named functions.
   Returns f64 for numeric values, closure_ty for lambda/function values. *)
let rec compile_expr ctx md builder (known_fns : (string, int * Llvm.llvalue) Hashtbl.t) env = function
  | Int n   -> Llvm.const_float (Llvm.double_type ctx) (float_of_int n)
  | Float f -> Llvm.const_float (Llvm.double_type ctx) f
  | Bool b  -> Llvm.const_float (Llvm.double_type ctx) (if b then 1.0 else 0.0)

  | Var x ->
    (match StringMap.find_opt x env with
     | Some v -> v
     | None ->
         match Hashtbl.find_opt known_fns x with
         | Some (1, fn) -> wrap_fn_as_closure ctx md builder fn x
         | Some (n, _)  ->
             failwith (Printf.sprintf
               "codegen: cannot use %d-arg function '%s' as a value (partial application not yet supported)" n x)
         | None -> failwith ("codegen: unbound variable: " ^ x))

  | Lam (x, body) ->
      let f64 = Llvm.double_type ctx in
      let p = ptr_ty ctx in
      (* Free vars of body excluding x and all known top-level functions *)
      let top = Hashtbl.fold (fun n _ s -> StringSet.add n s) known_fns StringSet.empty in
      let fv_set = free_vars (StringSet.add x top) body in
      let fv = StringSet.elements fv_set in
      let lam_name = fresh_lam () in
      (* Lifted lambda function: double(double arg, ptr env) *)
      let lft = Llvm.function_type f64 [| f64; p |] in
      let lam_fn = Llvm.define_function lam_name lft md in
      let lbb = Llvm.entry_block lam_fn in
      let lb  = Llvm.builder_at_end ctx lbb in
      Llvm.set_value_name x (Llvm.params lam_fn).(0);
      (* Unpack captured vars from env pointer (env is double[n]) *)
      let env_param = (Llvm.params lam_fn).(1) in
      let lam_env =
        List.fold_left (fun acc (i, name) ->
          let idx = Llvm.const_int (Llvm.i64_type ctx) i in
          let ep  = Llvm.build_gep f64 env_param [| idx |] ("ep_" ^ name) lb in
          let v   = Llvm.build_load f64 ep name lb in
          StringMap.add name v acc)
        (StringMap.singleton x (Llvm.params lam_fn).(0))
        (List.mapi (fun i n -> (i, n)) fv)
      in
      let result = compile_expr ctx md lb known_fns lam_env body in
      (* Only f64 return supported for now; closures-returning-closures need #95 *)
      (if Llvm.classify_type (Llvm.type_of result) <> Llvm.TypeKind.Double then
        failwith "codegen: lambda body must return a numeric value (closures returning closures require #95)");
      ignore (Llvm.build_ret result lb);
      (* In caller: alloc env array, fill with captured values, build closure *)
      let env_ptr =
        if fv = [] then Llvm.const_null p
        else begin
          let n    = List.length fv in
          let size = Llvm.const_int (Llvm.i64_type ctx) (n * 8) in
          let mfn  = get_malloc ctx md in
          let mft  = Llvm.function_type p [| Llvm.i64_type ctx |] in
          let ep   = Llvm.build_call mft mfn [| size |] "env_p" builder in
          List.iteri (fun i name ->
            let idx  = Llvm.const_int (Llvm.i64_type ctx) i in
            let slot = Llvm.build_gep f64 ep [| idx |] ("es_" ^ name) builder in
            let cap  = StringMap.find name env in
            ignore (Llvm.build_store cap slot builder))
          fv;
          ep
        end
      in
      build_closure ctx builder lam_fn env_ptr

  | App (e_fn, e_arg) ->
      let f64 = Llvm.double_type ctx in
      (* Try to flatten to a direct known-function call *)
      (match flatten_app [e_arg] e_fn with
       | Some (name, args) ->
           (match Hashtbl.find_opt known_fns name with
            | Some (arity, fn) when List.length args = arity ->
                let vargs = List.map (compile_expr ctx md builder known_fns env) args in
                let ft = Llvm.function_type f64 (Array.make arity f64) in
                Llvm.build_call ft fn (Array.of_list vargs) "dcall" builder
            | _ ->
                let fv = compile_expr ctx md builder known_fns env e_fn in
                let av = compile_expr ctx md builder known_fns env e_arg in
                call_closure ctx builder fv av)
       | None ->
           let fv = compile_expr ctx md builder known_fns env e_fn in
           let av = compile_expr ctx md builder known_fns env e_arg in
           call_closure ctx builder fv av)

  | Seq (Let (name, ve), rest) ->
      let v = compile_expr ctx md builder known_fns env ve in
      compile_expr ctx md builder known_fns (StringMap.add name v env) rest
  | Seq (a, b) ->
      ignore (compile_expr ctx md builder known_fns env a);
      compile_expr ctx md builder known_fns env b
  | Let (_, ve) ->
      compile_expr ctx md builder known_fns env ve

  | Add (a, b) ->
      Llvm.build_fadd
        (compile_expr ctx md builder known_fns env a)
        (compile_expr ctx md builder known_fns env b)
        "add" builder
  | Sub (a, b) ->
      Llvm.build_fsub
        (compile_expr ctx md builder known_fns env a)
        (compile_expr ctx md builder known_fns env b)
        "sub" builder
  | Mul (a, b) ->
      Llvm.build_fmul
        (compile_expr ctx md builder known_fns env a)
        (compile_expr ctx md builder known_fns env b)
        "mul" builder
  | Div (a, b) ->
      Llvm.build_fdiv
        (compile_expr ctx md builder known_fns env a)
        (compile_expr ctx md builder known_fns env b)
        "div" builder
  | Eq (a, b) ->
      let cmp = Llvm.build_fcmp Llvm.Fcmp.Oeq
        (compile_expr ctx md builder known_fns env a)
        (compile_expr ctx md builder known_fns env b)
        "eq" builder in
      Llvm.build_uitofp cmp (Llvm.double_type ctx) "eqf" builder

  | Llvm (intrinsic, args) ->
      let f64 = Llvm.double_type ctx in
      let vargs = List.map (compile_expr ctx md builder known_fns env) args in
      let arity = List.length vargs in
      let ft = Llvm.function_type f64 (Array.make arity f64) in
      let fn = match Llvm.lookup_function intrinsic md with
        | Some f -> f
        | None   -> Llvm.declare_function intrinsic ft md
      in
      Llvm.build_call ft fn (Array.of_list vargs) "r" builder

  | e -> failwith ("codegen: unsupported expression: " ^ string_of_expr e)

(* Compile a named function.
   If already pre-declared (compile_module path), fills its entry block.
   If not found (JIT path), defines it from scratch. *)
let compile_fundef ctx md known_fns name args body =
  let f64 = Llvm.double_type ctx in
  let fn = match Llvm.lookup_function name md with
    | Some f -> f
    | None ->
        let ft = Llvm.function_type f64 (Array.make (List.length args) f64) in
        Llvm.define_function name ft md
  in
  let bb      = Llvm.entry_block fn in
  let builder = Llvm.builder_at_end ctx bb in
  let params  = Array.to_list (Llvm.params fn) in
  let env =
    List.fold_left2
      (fun m aname param ->
         Llvm.set_value_name aname param;
         StringMap.add aname param m)
      StringMap.empty args params
  in
  let result = compile_expr ctx md builder known_fns env body in
  ignore (Llvm.build_ret result builder);
  fn

(* Compile a list of top-level expressions into an LLVM module.
   Two-pass: pre-declare all functions, then compile bodies (enables mutual recursion). *)
let compile_module exprs =
  let ctx = Llvm.create_context () in
  let md  = Llvm.create_module ctx "churing" in
  let known_fns : (string, int * Llvm.llvalue) Hashtbl.t = Hashtbl.create 16 in
  let f64 = Llvm.double_type ctx in
  (* Pre-pass: create function stubs with empty entry blocks *)
  List.iter (function
    | FunDef (name, args, _) ->
        let arity = List.length args in
        let ft    = Llvm.function_type f64 (Array.make arity f64) in
        let fn    = Llvm.define_function name ft md in
        Hashtbl.add known_fns name (arity, fn)
    | _ -> ()
  ) exprs;
  (* Compilation pass: fill in function bodies *)
  List.iter (function
    | FunDef (name, args, body) ->
        ignore (compile_fundef ctx md known_fns name args body)
    | _ -> ()
  ) exprs;
  (ctx, md)

(* ── JIT infrastructure for interpreter fallback (llvm intrinsic calls) ─────── *)

type jit_fn =
  | F1 of nativeint
  | F2 of nativeint

let make_jit_fn fn_name arity ee =
  let addr = get_fn_addr fn_name ee in
  match arity with
  | 1 -> F1 addr
  | 2 -> F2 addr
  | n -> failwith (Printf.sprintf "codegen: LLVM intrinsic arity %d not supported" n)

let call_jit_fn jit_fn float_args =
  match jit_fn, float_args with
  | F1 addr, [x]    -> call_f1 addr x
  | F2 addr, [x; y] -> call_f2 addr x y
  | _ -> failwith "codegen: JIT arity mismatch"

let jit_cache
  : (string * int, Llvm.llcontext * jit_fn * Llvm_executionengine.llexecutionengine) Hashtbl.t
  = Hashtbl.create 16

let call_intrinsic intrinsic float_args =
  ensure_initialized ();
  let arity = List.length float_args in
  let key = (intrinsic, arity) in
  let _, jit_fn, _ =
    match Hashtbl.find_opt jit_cache key with
    | Some entry -> entry
    | None ->
      let ctx = Llvm.create_context () in
      let md  = Llvm.create_module ctx ("jit_" ^ intrinsic) in
      let known_fns : (string, int * Llvm.llvalue) Hashtbl.t = Hashtbl.create 4 in
      let arg_names = List.init arity (fun i -> Printf.sprintf "x%d" i) in
      let body = Ast.Llvm (intrinsic, List.map (fun n -> Ast.Var n) arg_names) in
      let fn_name = "jit_" ^ String.concat "_" (String.split_on_char '.' intrinsic) in
      ignore (compile_fundef ctx md known_fns fn_name arg_names body);
      let ee =
        try Llvm_executionengine.create md
        with Llvm_executionengine.Error msg ->
          failwith ("LLVM JIT create failed: " ^ msg)
      in
      let jit_fn = make_jit_fn fn_name arity ee in
      let entry = (ctx, jit_fn, ee) in
      Hashtbl.add jit_cache key entry;
      entry
  in
  call_jit_fn jit_fn float_args
