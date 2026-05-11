open Ast

module StringMap = Map.Make(String)

(* Bind directly to the internal C symbol already exported by llvm.executionengine.
   The OCaml signature of get_function_address_ is (string -> ee -> nativeint),
   so the C function takes (value Name, value EE) in that order. *)
external get_fn_addr : string -> Llvm_executionengine.llexecutionengine -> nativeint
  = "llvm_ee_get_function_address"

(* C stubs in codegen_stubs.c — call a JIT-compiled function via raw nativeint pointer. *)
external call_f1 : nativeint -> float -> float = "caml_call_f1"
external call_f2 : nativeint -> float -> float -> float = "caml_call_f2"

let jit_initialized = ref false

let ensure_initialized () =
  if not !jit_initialized then begin
    jit_initialized := true;
    ignore (Llvm_executionengine.initialize ());
    Llvm_all_backends.initialize ()
  end

(* Compile an AST expression to an LLVM value.
   All numeric types are represented as double in this initial slice. *)
let rec compile_expr ctx md builder env = function
  | Int n   -> Llvm.const_float (Llvm.double_type ctx) (float_of_int n)
  | Float f -> Llvm.const_float (Llvm.double_type ctx) f
  | Bool b  -> Llvm.const_float (Llvm.double_type ctx) (if b then 1.0 else 0.0)
  | Var x ->
    (match StringMap.find_opt x env with
     | Some v -> v
     | None   -> failwith ("codegen: unbound variable: " ^ x))
  | Add (a, b) ->
    Llvm.build_fadd
      (compile_expr ctx md builder env a)
      (compile_expr ctx md builder env b)
      "add" builder
  | Sub (a, b) ->
    Llvm.build_fsub
      (compile_expr ctx md builder env a)
      (compile_expr ctx md builder env b)
      "sub" builder
  | Mul (a, b) ->
    Llvm.build_fmul
      (compile_expr ctx md builder env a)
      (compile_expr ctx md builder env b)
      "mul" builder
  | Div (a, b) ->
    Llvm.build_fdiv
      (compile_expr ctx md builder env a)
      (compile_expr ctx md builder env b)
      "div" builder
  | Eq (a, b) ->
    let cmp = Llvm.build_fcmp Llvm.Fcmp.Oeq
      (compile_expr ctx md builder env a)
      (compile_expr ctx md builder env b)
      "eq" builder in
    (* Convert i1 → double so all functions stay in double land *)
    Llvm.build_uitofp cmp (Llvm.double_type ctx) "eqf" builder
  | Llvm (intrinsic, args) ->
    let f64 = Llvm.double_type ctx in
    let vargs = List.map (compile_expr ctx md builder env) args in
    let arity = List.length vargs in
    let ft = Llvm.function_type f64 (Array.make arity f64) in
    let fn = match Llvm.lookup_function intrinsic md with
      | Some f -> f
      | None   -> Llvm.declare_function intrinsic ft md
    in
    Llvm.build_call ft fn (Array.of_list vargs) "r" builder
  | e -> failwith ("codegen: unsupported expression: " ^ string_of_expr e)

(* Compile a named function (all args and return type are double). *)
let compile_fundef ctx md name args body =
  let f64 = Llvm.double_type ctx in
  let arity = List.length args in
  let ft = Llvm.function_type f64 (Array.make arity f64) in
  let fn = Llvm.define_function name ft md in
  let bb = Llvm.entry_block fn in
  let builder = Llvm.builder_at_end ctx bb in
  let params = Array.to_list (Llvm.params fn) in
  let env = List.fold_left2
    (fun m aname param ->
       Llvm.set_value_name aname param;
       StringMap.add aname param m)
    StringMap.empty args params
  in
  let result = compile_expr ctx md builder env body in
  ignore (Llvm.build_ret result builder);
  fn

(* Compile a list of top-level FunDef nodes into an LLVM module.
   Returns (context, module) — caller must keep context alive as long as the module is used.
   Covers #91 scope: literals, arithmetic, llvm intrinsic calls, named functions. *)
let compile_module exprs =
  let ctx = Llvm.create_context () in
  let md  = Llvm.create_module ctx "churing" in
  List.iter (function
    | FunDef (name, args, body) -> ignore (compile_fundef ctx md name args body)
    | _ -> ()
  ) exprs;
  (ctx, md)

(* Typed native function pointer obtained via JIT. *)
type jit_fn =
  | F1 of nativeint   (* double -> double *)
  | F2 of nativeint   (* double -> double -> double *)

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

(* JIT cache: (intrinsic_name, arity) → (context, jit_fn, execution_engine).
   Context is stored alongside the EE so the module memory stays valid. *)
let jit_cache
  : (string * int, Llvm.llcontext * jit_fn * Llvm_executionengine.llexecutionengine) Hashtbl.t
  = Hashtbl.create 16

(* JIT-compile a wrapper for the given LLVM intrinsic on first call, then invoke it.
   All arguments and the return value are doubles. *)
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
      let arg_names = List.init arity (fun i -> Printf.sprintf "x%d" i) in
      let body = Ast.Llvm (intrinsic, List.map (fun n -> Ast.Var n) arg_names) in
      let fn_name = "jit_" ^ String.concat "_" (String.split_on_char '.' intrinsic) in
      ignore (compile_fundef ctx md fn_name arg_names body);
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
