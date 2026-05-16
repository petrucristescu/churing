open Ast

module StringMap = Map.Make(String)
module StringSet = Set.Make(String)

external get_fn_addr : string -> Llvm_executionengine.llexecutionengine -> nativeint
  = "llvm_ee_get_function_address"

external call_f1 : nativeint -> float -> float = "caml_call_f1"
external call_f2 : nativeint -> float -> float -> float = "caml_call_f2"

let jit_initialized = ref false
let lam_counter = ref 0
let str_counter = ref 0
let source_dir = ref ""

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

(* Cons cell struct: { double head, ptr tail }; nil = null ptr *)
let cons_cell_ty ctx =
  Llvm.struct_type ctx [| Llvm.double_type ctx; ptr_ty ctx |]

let fresh_lam () =
  let n = !lam_counter in
  incr lam_counter;
  Printf.sprintf "_lam_%d" n

(* Primitive names handled inline in the App case — not in known_fns or env *)
let inline_primitives = StringSet.of_list [
  "if";
  "cons"; "head"; "tail"; "empty";
  "gt"; "lt"; "gte"; "lte"; "not"; "and"; "or";
  "print";
  "length"; "concat"; "substring"; "uppercase"; "lowercase"; "trim";
  "contains"; "startsWith"; "endsWith"; "replace"; "toFloat";
  "indexOf"; "charAt";
]

(* Variables bound by a pattern *)
let rec pattern_vars = function
  | PVar x -> StringSet.singleton x
  | PCons (h, t) -> StringSet.union (pattern_vars h) (pattern_vars t)
  | PList ps ->
      List.fold_left (fun s p -> StringSet.union s (pattern_vars p)) StringSet.empty ps
  | _ -> StringSet.empty

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
  | List items ->
      List.fold_left (fun s e -> StringSet.union s (free_vars bound e)) StringSet.empty items
  | Match (e, arms) ->
      let fv_e = free_vars bound e in
      List.fold_left (fun s (pat, body) ->
        let bound' = StringSet.union bound (pattern_vars pat) in
        StringSet.union s (free_vars bound' body))
      fv_e arms
  | _ -> StringSet.empty

(* Flatten App(App(Var f, a1), a2) → Some (f, [a1; a2]) *)
let rec flatten_app acc = function
  | App (inner, arg) -> flatten_app (arg :: acc) inner
  | Var name -> Some (name, acc)
  | _ -> None

(* Boehm GC allocator — replaces malloc so the GC can track heap objects *)
let get_gc_malloc ctx md =
  let p = ptr_ty ctx in
  let ft = Llvm.function_type p [| Llvm.i64_type ctx |] in
  match Llvm.lookup_function "GC_malloc" md with
  | Some f -> f
  | None -> Llvm.declare_function "GC_malloc" ft md

(* Allocate a cons cell and fill head / tail fields *)
let build_cons ctx md builder head_v tail_ptr =
  let p = ptr_ty ctx in
  let cell_ty = cons_cell_ty ctx in
  let gc_malloc = get_gc_malloc ctx md in
  let gc_malloc_ft = Llvm.function_type p [| Llvm.i64_type ctx |] in
  let size = Llvm.const_int (Llvm.i64_type ctx) 16 in
  let cell = Llvm.build_call gc_malloc_ft gc_malloc [| size |] "cell" builder in
  let i32 = Llvm.i32_type ctx in
  let z = Llvm.const_int i32 0 in
  let hgep = Llvm.build_gep cell_ty cell [| z; Llvm.const_int i32 0 |] "hgep" builder in
  ignore (Llvm.build_store head_v hgep builder);
  let tgep = Llvm.build_gep cell_ty cell [| z; Llvm.const_int i32 1 |] "tgep" builder in
  ignore (Llvm.build_store tail_ptr tgep builder);
  cell

let build_list_head ctx builder cell =
  let cell_ty = cons_cell_ty ctx in
  let i32 = Llvm.i32_type ctx in
  let z = Llvm.const_int i32 0 in
  let gep = Llvm.build_gep cell_ty cell [| z; Llvm.const_int i32 0 |] "hgep" builder in
  Llvm.build_load (Llvm.double_type ctx) gep "head" builder

let build_list_tail ctx builder cell =
  let cell_ty = cons_cell_ty ctx in
  let p = ptr_ty ctx in
  let i32 = Llvm.i32_type ctx in
  let z = Llvm.const_int i32 0 in
  let gep = Llvm.build_gep cell_ty cell [| z; Llvm.const_int i32 1 |] "tgep" builder in
  Llvm.build_load p gep "tail" builder

(* Heap-allocate a closure {fn_ptr, env_ptr} and return a ptr to it *)
let build_closure ctx md builder fn env_ptr =
  let p = ptr_ty ctx in
  let clos_t = closure_ty ctx in
  let gc_malloc = get_gc_malloc ctx md in
  let gc_malloc_ft = Llvm.function_type p [| Llvm.i64_type ctx |] in
  let size = Llvm.const_int (Llvm.i64_type ctx) 16 in
  let clos_ptr = Llvm.build_call gc_malloc_ft gc_malloc [| size |] "clos_p" builder in
  let i32 = Llvm.i32_type ctx in
  let z = Llvm.const_int i32 0 in
  let fn_p = Llvm.build_bitcast fn p "fn_p" builder in
  let fgep = Llvm.build_gep clos_t clos_ptr [| z; Llvm.const_int i32 0 |] "fgep" builder in
  ignore (Llvm.build_store fn_p fgep builder);
  let egep = Llvm.build_gep clos_t clos_ptr [| z; Llvm.const_int i32 1 |] "egep" builder in
  ignore (Llvm.build_store env_ptr egep builder);
  clos_ptr

(* Call a closure ptr: load {fn_ptr, env_ptr}, call fn_ptr(arg, env_ptr) with given ret_ty *)
let call_closure ctx builder clos_ptr arg_val ret_ty =
  let p = ptr_ty ctx in
  let clos_t = closure_ty ctx in
  let i32 = Llvm.i32_type ctx in
  let z = Llvm.const_int i32 0 in
  let fn_gep  = Llvm.build_gep clos_t clos_ptr [| z; Llvm.const_int i32 0 |] "fn_gep"  builder in
  let env_gep = Llvm.build_gep clos_t clos_ptr [| z; Llvm.const_int i32 1 |] "env_gep" builder in
  let fn_ptr  = Llvm.build_load p fn_gep  "fn_ptr"  builder in
  let env_ptr = Llvm.build_load p env_gep "env_ptr" builder in
  let arg_ty  = Llvm.type_of arg_val in
  let ft = Llvm.function_type ret_ty [| arg_ty; p |] in
  Llvm.build_call ft fn_ptr [| arg_val; env_ptr |] "app" builder

(* Wrap a 1-arg f64→f64 named function as a heap-allocated closure ptr *)
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
  build_closure ctx md builder wfn (Llvm.const_null p)

(* ── Static type analysis for pre-pass ─────────────────────────────────────── *)

(* Check if arg_name is directly matched with cons/nil patterns in body *)
let rec arg_is_list_in_body arg_name = function
  | Match (Var x, arms) when x = arg_name ->
      List.exists (fun (pat, _) ->
        match pat with PCons _ | PList _ -> true | _ -> false) arms
  | Match (e, arms) ->
      arg_is_list_in_body arg_name e ||
      List.exists (fun (_, b) -> arg_is_list_in_body arg_name b) arms
  | Seq (a, b) -> arg_is_list_in_body arg_name a || arg_is_list_in_body arg_name b
  | App (f, a) -> arg_is_list_in_body arg_name f || arg_is_list_in_body arg_name a
  | Lam (x, body) when x <> arg_name -> arg_is_list_in_body arg_name body
  | Let (_, v) -> arg_is_list_in_body arg_name v
  | _ -> false

(* Check if arg_name appears in direct function-call position, indicating a closure arg *)
let rec arg_is_closure_in_body arg_name = function
  | App (Var f, _) when f = arg_name -> true
  | App (f, a) ->
      arg_is_closure_in_body arg_name f || arg_is_closure_in_body arg_name a
  | Match (e, arms) ->
      arg_is_closure_in_body arg_name e ||
      List.exists (fun (_, b) -> arg_is_closure_in_body arg_name b) arms
  | Seq (a, b) -> arg_is_closure_in_body arg_name a || arg_is_closure_in_body arg_name b
  | Lam (x, body) when x <> arg_name -> arg_is_closure_in_body arg_name body
  | Let (_, v) -> arg_is_closure_in_body arg_name v
  | _ -> false

(* Vars introduced as list by a cons pattern's tail binding *)
let rec cons_tail_vars = function
  | PCons (_, PVar t) -> StringSet.singleton t
  | PCons (_, tp) -> cons_tail_vars tp
  | _ -> StringSet.empty

(* Check if body produces a list result.
   list_fns: known list-returning functions.
   list_vars: local vars known to be lists (from pattern bindings).
   is_list_arg: test if a var name is a list-typed argument. *)
let rec body_returns_list_v list_fns list_vars is_list_arg = function
  | List _ -> true
  | Lam _ -> true  (* lambdas are heap-allocated — return ptr *)
  | Var x -> StringSet.mem x list_vars || StringSet.mem x list_fns || is_list_arg x
  | Match (_, arms) ->
      List.exists (fun (pat, body) ->
        let extra = cons_tail_vars pat in
        body_returns_list_v list_fns (StringSet.union list_vars extra) is_list_arg body) arms
  | Seq (_, b) -> body_returns_list_v list_fns list_vars is_list_arg b
  | App (f, _) ->
      (match flatten_app [] f with
       | Some ("cons", _) -> true
       | Some (name, _) -> StringSet.mem name list_fns
       | None -> false)
  | _ -> false

let body_returns_list list_fns body =
  body_returns_list_v list_fns StringSet.empty (fun _ -> false) body

(* Get or declare an external C function in the module *)
let declare_ext md name ft =
  match Llvm.lookup_function name md with
  | Some f -> f
  | None -> Llvm.declare_function name ft md

(* ── Main expression compiler ───────────────────────────────────────────────── *)
(* known_fns: name → (function_type, function_value)
   Storing ft explicitly avoids relying on Llvm.type_of which may return ptr in LLVM 18.
   in_tail: true when this expression is in tail position — enables tail call marking.
   Returns f64 for numbers/bools, closure struct for lambdas, ptr for lists. *)
let rec compile_expr ctx md builder (known_fns : (string, Llvm.lltype * Llvm.llvalue) Hashtbl.t) env in_tail = function
  | Int n   -> Llvm.const_float (Llvm.double_type ctx) (float_of_int n)
  | Float f -> Llvm.const_float (Llvm.double_type ctx) f
  | Lng n   -> Llvm.const_float (Llvm.double_type ctx) (Int64.to_float n)
  | Bool b  -> Llvm.const_float (Llvm.double_type ctx) (if b then 1.0 else 0.0)

  | Str s ->
      let bytes = s ^ "\x00" in
      let const_val = Llvm.const_string ctx bytes in
      let n = !str_counter in
      incr str_counter;
      let gname = Printf.sprintf ".str%d" n in
      let g = Llvm.define_global gname const_val md in
      Llvm.set_linkage Llvm.Linkage.Private g;
      Llvm.set_global_constant true g;
      let z = Llvm.const_int (Llvm.i64_type ctx) 0 in
      Llvm.build_gep (Llvm.type_of const_val) g [| z; z |] "strp" builder

  | Var x ->
    (match StringMap.find_opt x env with
     | Some v -> v
     | None ->
         match Hashtbl.find_opt known_fns x with
         | Some (ft, fn) ->
             let param_types = Llvm.param_types ft in
             let ret_ty = Llvm.return_type ft in
             if Array.length param_types = 1 &&
                Llvm.classify_type param_types.(0) = Llvm.TypeKind.Double &&
                Llvm.classify_type ret_ty = Llvm.TypeKind.Double
             then wrap_fn_as_closure ctx md builder fn x
             else failwith (Printf.sprintf
               "codegen: cannot use '%s' as a value (only f64→f64 functions wrap as closures)" x)
         | None ->
             (match x with
              | "nil"   -> Llvm.const_null (ptr_ty ctx)
              | "true"  -> Llvm.const_float (Llvm.double_type ctx) 1.0
              | "false" -> Llvm.const_float (Llvm.double_type ctx) 0.0
              | _ -> failwith ("codegen: unbound variable: " ^ x)))

  | Lam (x, body) ->
      let f64 = Llvm.double_type ctx in
      let p = ptr_ty ctx in
      let i8 = Llvm.i8_type ctx in
      let top = Hashtbl.fold (fun n _ s -> StringSet.add n s) known_fns inline_primitives in
      let fv_set = free_vars (StringSet.add x top) body in
      let fv = StringSet.elements fv_set in
      (* Captured values with their types *)
      let fv_caps = List.map (fun name ->
        match StringMap.find_opt name env with
        | Some v -> (name, v)
        | None -> failwith (Printf.sprintf "codegen: lambda captures unbound variable '%s'" name)
      ) fv in
      let lam_name = fresh_lam () in
      (* Determine arg and return types from static analysis *)
      let arg_is_ptr = arg_is_list_in_body x body || arg_is_closure_in_body x body in
      let ret_is_ptr = body_returns_list StringSet.empty body in
      let arg_ty = if arg_is_ptr then p else f64 in
      let ret_ty = if ret_is_ptr then p else f64 in
      let lft = Llvm.function_type ret_ty [| arg_ty; p |] in
      let lam_fn = Llvm.define_function lam_name lft md in
      let lbb = Llvm.entry_block lam_fn in
      let lb  = Llvm.builder_at_end ctx lbb in
      Llvm.set_value_name x (Llvm.params lam_fn).(0);
      let env_param = (Llvm.params lam_fn).(1) in
      (* Unpack env: byte-level GEP so mixed f64/ptr slots are handled correctly *)
      let lam_env =
        List.fold_left (fun acc (i, (name, cap_val)) ->
          let slot_ty = Llvm.type_of cap_val in
          let byte_off = Llvm.const_int (Llvm.i64_type ctx) (i * 8) in
          let ep = Llvm.build_gep i8 env_param [| byte_off |] ("ep_" ^ name) lb in
          let v  = Llvm.build_load slot_ty ep name lb in
          StringMap.add name v acc)
        (StringMap.singleton x (Llvm.params lam_fn).(0))
        (List.mapi (fun i nc -> (i, nc)) fv_caps)
      in
      let result = compile_expr ctx md lb known_fns lam_env true body in
      ignore (Llvm.build_ret result lb);
      (* Pack env: byte-level GEP, store each captured value with its actual type *)
      let env_ptr =
        if fv = [] then Llvm.const_null p
        else begin
          let n    = List.length fv_caps in
          let size = Llvm.const_int (Llvm.i64_type ctx) (n * 8) in
          let mfn  = get_gc_malloc ctx md in
          let mft  = Llvm.function_type p [| Llvm.i64_type ctx |] in
          let ep   = Llvm.build_call mft mfn [| size |] "env_p" builder in
          List.iteri (fun i (name, _) ->
            let cap      = StringMap.find name env in
            let slot_ty  = Llvm.type_of cap in
            let byte_off = Llvm.const_int (Llvm.i64_type ctx) (i * 8) in
            let slot     = Llvm.build_gep i8 ep [| byte_off |] ("es_" ^ name) builder in
            ignore (Llvm.build_store cap slot builder);
            ignore slot_ty)  (* slot_ty used via cap *)
          fv_caps;
          ep
        end
      in
      build_closure ctx md builder lam_fn env_ptr

  | App (e_fn, e_arg) ->
      let f64 = Llvm.double_type ctx in
      let p = ptr_ty ctx in
      (match flatten_app [e_arg] e_fn with
       (* ── Built-in list operations ── *)
       | Some ("cons", [h_expr; t_expr]) ->
           let h = compile_expr ctx md builder known_fns env false h_expr in
           let t = compile_expr ctx md builder known_fns env false t_expr in
           build_cons ctx md builder h t
       | Some ("head", [lst_expr]) ->
           let lst = compile_expr ctx md builder known_fns env false lst_expr in
           build_list_head ctx builder lst
       | Some ("tail", [lst_expr]) ->
           let lst = compile_expr ctx md builder known_fns env false lst_expr in
           build_list_tail ctx builder lst
       | Some ("empty", [lst_expr]) ->
           let lst = compile_expr ctx md builder known_fns env false lst_expr in
           let null_ptr = Llvm.const_null p in
           let cond = Llvm.build_icmp Llvm.Icmp.Eq lst null_ptr "is_empty" builder in
           Llvm.build_uitofp cond f64 "emptyf" builder
       (* ── Comparison ops ── *)
       | Some ("gt",  [a; b]) ->
           let av = compile_expr ctx md builder known_fns env false a in
           let bv = compile_expr ctx md builder known_fns env false b in
           Llvm.build_uitofp (Llvm.build_fcmp Llvm.Fcmp.Ogt av bv "gt" builder) f64 "gtf" builder
       | Some ("lt",  [a; b]) ->
           let av = compile_expr ctx md builder known_fns env false a in
           let bv = compile_expr ctx md builder known_fns env false b in
           Llvm.build_uitofp (Llvm.build_fcmp Llvm.Fcmp.Olt av bv "lt" builder) f64 "ltf" builder
       | Some ("gte", [a; b]) ->
           let av = compile_expr ctx md builder known_fns env false a in
           let bv = compile_expr ctx md builder known_fns env false b in
           Llvm.build_uitofp (Llvm.build_fcmp Llvm.Fcmp.Oge av bv "gte" builder) f64 "gtef" builder
       | Some ("lte", [a; b]) ->
           let av = compile_expr ctx md builder known_fns env false a in
           let bv = compile_expr ctx md builder known_fns env false b in
           Llvm.build_uitofp (Llvm.build_fcmp Llvm.Fcmp.Ole av bv "lte" builder) f64 "ltef" builder
       | Some ("not", [x]) ->
           let xv = compile_expr ctx md builder known_fns env false x in
           Llvm.build_uitofp
             (Llvm.build_fcmp Llvm.Fcmp.Oeq xv (Llvm.const_float f64 0.0) "not" builder)
             f64 "notf" builder
       | Some ("and", [a; b]) ->
           let av = compile_expr ctx md builder known_fns env false a in
           let bv = compile_expr ctx md builder known_fns env false b in
           let ca = Llvm.build_fcmp Llvm.Fcmp.One av (Llvm.const_float f64 0.0) "ca" builder in
           let cb = Llvm.build_fcmp Llvm.Fcmp.One bv (Llvm.const_float f64 0.0) "cb" builder in
           Llvm.build_uitofp (Llvm.build_and ca cb "andb" builder) f64 "andf" builder
       | Some ("or", [a; b]) ->
           let av = compile_expr ctx md builder known_fns env false a in
           let bv = compile_expr ctx md builder known_fns env false b in
           let ca = Llvm.build_fcmp Llvm.Fcmp.One av (Llvm.const_float f64 0.0) "ca" builder in
           let cb = Llvm.build_fcmp Llvm.Fcmp.One bv (Llvm.const_float f64 0.0) "cb" builder in
           Llvm.build_uitofp (Llvm.build_or ca cb "orb" builder) f64 "orf" builder
       (* ── print via printf ── *)
       | Some ("print", [val_expr]) ->
           let v = compile_expr ctx md builder known_fns env false val_expr in
           let i8p = Llvm.pointer_type ctx in
           let printf_ft = Llvm.var_arg_function_type (Llvm.i32_type ctx) [| i8p |] in
           let printf_fn = match Llvm.lookup_function "printf" md with
             | Some f -> f
             | None -> Llvm.declare_function "printf" printf_ft md in
           let fmt_str = "%g\n\x00" in
           let fmt_const = Llvm.const_string ctx fmt_str in
           let fmt_global = match Llvm.lookup_global ".fmt_g" md with
             | Some g -> g
             | None ->
                 let g = Llvm.define_global ".fmt_g" fmt_const md in
                 Llvm.set_linkage Llvm.Linkage.Private g;
                 Llvm.set_global_constant true g;
                 g in
           let fmt_ptr = Llvm.build_gep (Llvm.type_of fmt_const)
             fmt_global [| Llvm.const_int (Llvm.i64_type ctx) 0;
                            Llvm.const_int (Llvm.i64_type ctx) 0 |]
             "fmt" builder in
           ignore (Llvm.build_call printf_ft printf_fn [| fmt_ptr; v |] "" builder);
           v
       (* ── String operations ── *)
       | Some ("length", [s_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type f64 [| p |] in
           Llvm.build_call ft (declare_ext md "churing_str_length" ft) [| sv |] "slen" builder
       | Some ("concat", [a_expr; b_expr]) ->
           let av = compile_expr ctx md builder known_fns env false a_expr in
           let bv = compile_expr ctx md builder known_fns env false b_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p; p |] in
           Llvm.build_call ft (declare_ext md "churing_concat" ft) [| av; bv |] "concat" builder
       | Some ("substring", [s_expr; st_expr; ln_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let stv = compile_expr ctx md builder known_fns env false st_expr in
           let lnv = compile_expr ctx md builder known_fns env false ln_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p; f64; f64 |] in
           Llvm.build_call ft (declare_ext md "churing_substring" ft) [| sv; stv; lnv |] "substr" builder
       | Some ("uppercase", [s_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p |] in
           Llvm.build_call ft (declare_ext md "churing_uppercase" ft) [| sv |] "ucase" builder
       | Some ("lowercase", [s_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p |] in
           Llvm.build_call ft (declare_ext md "churing_lowercase" ft) [| sv |] "lcase" builder
       | Some ("trim", [s_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p |] in
           Llvm.build_call ft (declare_ext md "churing_trim" ft) [| sv |] "trim" builder
       | Some ("contains", [s_expr; sub_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let subv = compile_expr ctx md builder known_fns env false sub_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type f64 [| p; p |] in
           Llvm.build_call ft (declare_ext md "churing_contains" ft) [| sv; subv |] "contains" builder
       | Some ("startsWith", [s_expr; pre_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let prev = compile_expr ctx md builder known_fns env false pre_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type f64 [| p; p |] in
           Llvm.build_call ft (declare_ext md "churing_starts_with" ft) [| sv; prev |] "sw" builder
       | Some ("endsWith", [s_expr; suf_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let sufv = compile_expr ctx md builder known_fns env false suf_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type f64 [| p; p |] in
           Llvm.build_call ft (declare_ext md "churing_ends_with" ft) [| sv; sufv |] "ew" builder
       | Some ("replace", [s_expr; from_expr; to_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let fromv = compile_expr ctx md builder known_fns env false from_expr in
           let tov = compile_expr ctx md builder known_fns env false to_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p; p; p |] in
           Llvm.build_call ft (declare_ext md "churing_replace" ft) [| sv; fromv; tov |] "replace" builder
       | Some (("toString" | "str"), [n_expr]) ->
           let nv = compile_expr ctx md builder known_fns env false n_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| f64 |] in
           Llvm.build_call ft (declare_ext md "churing_to_string" ft) [| nv |] "tostr" builder
       | Some ("toFloat", [s_expr]) ->
           let sv = compile_expr ctx md builder known_fns env false s_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type f64 [| p |] in
           Llvm.build_call ft (declare_ext md "churing_to_float" ft) [| sv |] "tofloat" builder
       | Some ("indexOf", [s_expr; sub_expr]) ->
           let sv  = compile_expr ctx md builder known_fns env false s_expr in
           let sub = compile_expr ctx md builder known_fns env false sub_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type f64 [| p; p |] in
           Llvm.build_call ft (declare_ext md "churing_index_of" ft) [| sv; sub |] "indexof" builder
       | Some ("charAt", [s_expr; idx_expr]) ->
           let sv  = compile_expr ctx md builder known_fns env false s_expr in
           let iv  = compile_expr ctx md builder known_fns env false idx_expr in
           let p = ptr_ty ctx in
           let ft = Llvm.function_type p [| p; f64 |] in
           Llvm.build_call ft (declare_ext md "churing_char_at" ft) [| sv; iv |] "charat" builder
       (* ── if cond then_val else_val ── *)
       | Some ("if", [cond_e; then_e; else_e]) ->
           let cond_v = compile_expr ctx md builder known_fns env false cond_e in
           let zero = Llvm.const_float f64 0.0 in
           let cmp = Llvm.build_fcmp Llvm.Fcmp.One cond_v zero "if_cond" builder in
           let fn = Llvm.block_parent (Llvm.insertion_block builder) in
           let then_bb  = Llvm.append_block ctx "if_then"  fn in
           let else_bb  = Llvm.append_block ctx "if_else"  fn in
           let merge_bb = Llvm.append_block ctx "if_merge" fn in
           ignore (Llvm.build_cond_br cmp then_bb else_bb builder);
           Llvm.position_at_end then_bb builder;
           let then_v = compile_expr ctx md builder known_fns env in_tail then_e in
           let then_end = Llvm.insertion_block builder in
           ignore (Llvm.build_br merge_bb builder);
           Llvm.position_at_end else_bb builder;
           let else_v = compile_expr ctx md builder known_fns env in_tail else_e in
           let else_end = Llvm.insertion_block builder in
           ignore (Llvm.build_br merge_bb builder);
           Llvm.position_at_end merge_bb builder;
           Llvm.build_phi [(then_v, then_end); (else_v, else_end)] "if_r" builder
       (* ── Known function direct call ── *)
       | Some (name, args) ->
           (match Hashtbl.find_opt known_fns name with
            | Some (ft, fn) ->
                let arity = Array.length (Llvm.param_types ft) in
                if List.length args = arity then begin
                  let vargs = List.map (compile_expr ctx md builder known_fns env false) args in
                  let call = Llvm.build_call ft fn (Array.of_list vargs) "dcall" builder in
                  if in_tail then Llvm.set_tail_call true call;
                  call
                end else begin
                  let fv = compile_expr ctx md builder known_fns env false e_fn in
                  let av = compile_expr ctx md builder known_fns env false e_arg in
                  call_closure ctx builder fv av f64
                end
            | None ->
                (* Unresolved name: simulate curried closure application over all args.
                   Non-final calls return ptr (intermediate closure); final returns f64. *)
                let fv = compile_expr ctx md builder known_fns env false (Var name) in
                let vargs = List.map (compile_expr ctx md builder known_fns env false) args in
                let n = List.length vargs in
                snd (List.fold_left (fun (i, clos) av ->
                  let ret_ty = if i = n - 1 then f64 else p in
                  (i + 1, call_closure ctx builder clos av ret_ty)
                ) (0, fv) vargs))
       | None ->
           (* Non-variable function position: determine return type from e_fn shape *)
           let fv = compile_expr ctx md builder known_fns env false e_fn in
           let av = compile_expr ctx md builder known_fns env false e_arg in
           let ret_ty = match e_fn with
             | Lam (_, lbody) ->
                 if body_returns_list StringSet.empty lbody then p else f64
             | App _ -> p  (* intermediate curried call returns a closure *)
             | _ -> f64
           in
           call_closure ctx builder fv av ret_ty)

  | List items ->
      let p = ptr_ty ctx in
      let vs = List.map (compile_expr ctx md builder known_fns env false) items in
      List.fold_right (build_cons ctx md builder) vs (Llvm.const_null p)

  | Match (scrutinee_expr, arms) ->
      let scrutinee = compile_expr ctx md builder known_fns env false scrutinee_expr in
      let fn = Llvm.block_parent (Llvm.insertion_block builder) in
      let f64 = Llvm.double_type ctx in
      let p = ptr_ty ctx in
      let end_bb = Llvm.append_block ctx "match_end" fn in
      let arm_results : (Llvm.llvalue * Llvm.llbasicblock) list ref = ref [] in
      let add_result v =
        let pred_bb = Llvm.insertion_block builder in
        ignore (Llvm.build_br end_bb builder);
        arm_results := (v, pred_bb) :: !arm_results
      in
      let rec compile_arms = function
        | [] -> ignore (Llvm.build_unreachable builder)
        | (pat, body) :: rest ->
            let compile_body env' =
              let result = compile_expr ctx md builder known_fns env' in_tail body in
              add_result result
            in
            (match pat with
             | PWild -> compile_body env
             | PVar x -> compile_body (StringMap.add x scrutinee env)
             | PInt n ->
                 let body_bb = Llvm.append_block ctx "arm_body" fn in
                 let next_bb = Llvm.append_block ctx "arm_next" fn in
                 let cmp = Llvm.build_fcmp Llvm.Fcmp.Oeq scrutinee
                   (Llvm.const_float f64 (float_of_int n)) "peq" builder in
                 ignore (Llvm.build_cond_br cmp body_bb next_bb builder);
                 Llvm.position_at_end body_bb builder;
                 compile_body env;
                 Llvm.position_at_end next_bb builder;
                 compile_arms rest
             | PBool b ->
                 let body_bb = Llvm.append_block ctx "arm_body" fn in
                 let next_bb = Llvm.append_block ctx "arm_next" fn in
                 let bv = Llvm.const_float f64 (if b then 1.0 else 0.0) in
                 let cmp = Llvm.build_fcmp Llvm.Fcmp.Oeq scrutinee bv "peq" builder in
                 ignore (Llvm.build_cond_br cmp body_bb next_bb builder);
                 Llvm.position_at_end body_bb builder;
                 compile_body env;
                 Llvm.position_at_end next_bb builder;
                 compile_arms rest
             | PList [] ->
                 let body_bb = Llvm.append_block ctx "arm_nil" fn in
                 let next_bb = Llvm.append_block ctx "arm_next" fn in
                 let null_ptr = Llvm.const_null p in
                 let cmp = Llvm.build_icmp Llvm.Icmp.Eq scrutinee null_ptr "is_nil" builder in
                 ignore (Llvm.build_cond_br cmp body_bb next_bb builder);
                 Llvm.position_at_end body_bb builder;
                 compile_body env;
                 Llvm.position_at_end next_bb builder;
                 compile_arms rest
             | PCons (hp, tp) ->
                 let body_bb = Llvm.append_block ctx "arm_cons" fn in
                 let next_bb = Llvm.append_block ctx "arm_next" fn in
                 let null_ptr = Llvm.const_null p in
                 let cmp = Llvm.build_icmp Llvm.Icmp.Ne scrutinee null_ptr "is_cons" builder in
                 ignore (Llvm.build_cond_br cmp body_bb next_bb builder);
                 Llvm.position_at_end body_bb builder;
                 let head_val = build_list_head ctx builder scrutinee in
                 let tail_val = build_list_tail ctx builder scrutinee in
                 let env' = match hp with
                   | PVar h -> StringMap.add h head_val env
                   | PWild  -> env
                   | _ -> failwith "codegen: nested head patterns in PCons not supported"
                 in
                 let env'' = match tp with
                   | PVar t  -> StringMap.add t tail_val env'
                   | PWild   -> env'
                   | PList [] -> env'
                   | _ -> failwith "codegen: nested tail patterns in PCons not supported"
                 in
                 compile_body env'';
                 Llvm.position_at_end next_bb builder;
                 compile_arms rest
             | PStr _ | PList _ ->
                 failwith "codegen: string/non-empty list literal patterns not supported in LLVM backend")
      in
      compile_arms arms;
      Llvm.position_at_end end_bb builder;
      (match !arm_results with
       | [] -> failwith "codegen: match with no reachable arms"
       | results -> Llvm.build_phi results "match_r" builder)

  | Seq (Let (name, ve), rest) ->
      let v = compile_expr ctx md builder known_fns env false ve in
      compile_expr ctx md builder known_fns (StringMap.add name v env) in_tail rest
  | Seq (a, b) ->
      ignore (compile_expr ctx md builder known_fns env false a);
      compile_expr ctx md builder known_fns env in_tail b
  | Let (_, ve) ->
      compile_expr ctx md builder known_fns env in_tail ve

  | Add (a, b) ->
      Llvm.build_fadd
        (compile_expr ctx md builder known_fns env false a)
        (compile_expr ctx md builder known_fns env false b)
        "add" builder
  | Sub (a, b) ->
      Llvm.build_fsub
        (compile_expr ctx md builder known_fns env false a)
        (compile_expr ctx md builder known_fns env false b)
        "sub" builder
  | Mul (a, b) ->
      Llvm.build_fmul
        (compile_expr ctx md builder known_fns env false a)
        (compile_expr ctx md builder known_fns env false b)
        "mul" builder
  | Div (a, b) ->
      Llvm.build_fdiv
        (compile_expr ctx md builder known_fns env false a)
        (compile_expr ctx md builder known_fns env false b)
        "div" builder
  | Eq (a, b) ->
      let f64 = Llvm.double_type ctx in
      let p = ptr_ty ctx in
      let av = compile_expr ctx md builder known_fns env false a in
      let bv = compile_expr ctx md builder known_fns env false b in
      (match Llvm.classify_type (Llvm.type_of av) with
       | Llvm.TypeKind.Pointer ->
           let strcmp_ft = Llvm.function_type (Llvm.i32_type ctx) [| p; p |] in
           let strcmp_fn = declare_ext md "strcmp" strcmp_ft in
           let cmp = Llvm.build_call strcmp_ft strcmp_fn [| av; bv |] "scmp" builder in
           let z = Llvm.const_int (Llvm.i32_type ctx) 0 in
           Llvm.build_uitofp (Llvm.build_icmp Llvm.Icmp.Eq cmp z "seq" builder) f64 "seqf" builder
       | _ ->
           Llvm.build_uitofp
             (Llvm.build_fcmp Llvm.Fcmp.Oeq av bv "eq" builder)
             f64 "eqf" builder)

  | Llvm (intrinsic, args) ->
      let f64 = Llvm.double_type ctx in
      let vargs = List.map (compile_expr ctx md builder known_fns env false) args in
      let arity = List.length vargs in
      let ft = Llvm.function_type f64 (Array.make arity f64) in
      let fn = match Llvm.lookup_function intrinsic md with
        | Some f -> f
        | None   -> Llvm.declare_function intrinsic ft md
      in
      Llvm.build_call ft fn (Array.of_list vargs) "r" builder

  | e -> failwith ("codegen: unsupported expression: " ^ string_of_expr e)

(* Compile a named function body into its pre-declared (or freshly declared) LLVM function.
   When called from compile_module, the function is already in known_fns (from the pre-pass)
   and has a mangled LLVM name to prevent libc symbol conflicts.
   When called from the JIT path, known_fns is empty and we define a fresh function. *)
let compile_fundef ctx md known_fns name args body =
  let f64 = Llvm.double_type ctx in
  let p = ptr_ty ctx in
  let fn, ft = match Hashtbl.find_opt known_fns name with
    | Some (ft, fn) ->
        (* Pre-pass already declared this function — use it directly. *)
        (fn, ft)
    | None ->
        (* JIT path: compute types from body analysis and define fresh. *)
        let arg_types = List.map
          (fun a -> if arg_is_list_in_body a body then p else f64) args in
        let ret_type = if body_returns_list StringSet.empty body then p else f64 in
        let ft = Llvm.function_type ret_type (Array.of_list arg_types) in
        let fn = Llvm.define_function name ft md in
        Hashtbl.add known_fns name (ft, fn);
        (fn, ft)
  in
  ignore ft;
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
  let result = compile_expr ctx md builder known_fns env true body in
  ignore (Llvm.build_ret result builder);
  fn

(* Find index of name in args list, or None. *)
let find_arg_index args x =
  let rec go i = function
    | [] -> None
    | a :: _ when a = x -> Some i
    | _ :: rest -> go (i+1) rest
  in
  go 0 args

(* Compile a list of top-level expressions into an LLVM module.
   Combined fixpoint: list-returning functions and list-typed args inform each other. *)
let compile_module exprs =
  let ctx = Llvm.create_context () in
  let md  = Llvm.create_module ctx "churing" in
  let known_fns : (string, Llvm.lltype * Llvm.llvalue) Hashtbl.t = Hashtbl.create 16 in
  let f64 = Llvm.double_type ctx in
  let p = ptr_ty ctx in
  let fundefs = List.filter_map (function
    | FunDef (name, args, body) -> Some (name, args, body)
    | _ -> None
  ) exprs in
  (* Combined fixpoint: list_fns (return type) and list_arg_tbl (arg types) *)
  let list_fns = ref StringSet.empty in
  let list_arg_tbl : (string * int, bool) Hashtbl.t = Hashtbl.create 16 in
  let mark fn i =
    if not (Hashtbl.mem list_arg_tbl (fn, i)) then
      (Hashtbl.replace list_arg_tbl (fn, i) true; true)
    else false
  in
  (* Seed: arg directly matched with cons/nil, or used in function-call position (closure) *)
  List.iter (fun (name, args, body) ->
    List.iteri (fun i a ->
      if arg_is_list_in_body a body || arg_is_closure_in_body a body
      then ignore (mark name i)) args
  ) fundefs;
  (* Seed: arg appears as cons tail — cons _ arg *)
  List.iter (fun (name, args, body) ->
    let rec scan = function
      | App (App (Var "cons", _), Var x) ->
          (match find_arg_index args x with Some i -> ignore (mark name i) | None -> ())
      | App (f, a) -> scan f; scan a
      | Match (e, arms) -> scan e; List.iter (fun (_, b) -> scan b) arms
      | Seq (a, b) -> scan a; scan b
      | Lam (_, b) -> scan b | Let (_, v) -> scan v | _ -> ()
    in scan body) fundefs;
  (* Interleaved fixpoint: list_fns and list_arg_tbl inform each other *)
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter (fun (name, args, body) ->
      (* is_list_arg: check if a Var in this fn's body is a list-typed arg *)
      let is_list_arg x =
        match find_arg_index args x with
        | Some i -> Hashtbl.mem list_arg_tbl (name, i)
        | None -> false
      in
      (* Update list_fns: use arg type info to detect list-returning fns *)
      if not (StringSet.mem name !list_fns) &&
         body_returns_list_v !list_fns StringSet.empty is_list_arg body then begin
        list_fns := StringSet.add name !list_fns;
        changed := true
      end;
      (* Update list_arg_tbl: arg returned as Var from a list-returning fn *)
      if body_returns_list_v !list_fns StringSet.empty is_list_arg body then begin
        let rec check_ret = function
          | Var x ->
              (match find_arg_index args x with
               | Some i -> if mark name i then changed := true
               | None -> ())
          | Match (_, arms) -> List.iter (fun (_, b) -> check_ret b) arms
          | Seq (_, b) -> check_ret b
          | _ -> ()
        in check_ret body
      end;
      (* Update list_arg_tbl: propagate via call sites *)
      let rec walk = function
        | App _ as app ->
            (match flatten_app [] app with
             | Some (gname, call_args) ->
                 List.iteri (fun j call_arg ->
                   if Hashtbl.mem list_arg_tbl (gname, j) then
                     match call_arg with
                     | Var x ->
                         (match find_arg_index args x with
                          | Some i -> if mark name i then changed := true
                          | None -> ())
                     | _ -> ()
                 ) call_args
             | None -> ());
            (match app with App (f, a) -> walk f; walk a | _ -> ())
        | Match (e, arms) -> walk e; List.iter (fun (_, b) -> walk b) arms
        | Seq (a, b) -> walk a; walk b
        | Lam (_, b) -> walk b | Let (_, v) -> walk v | _ -> ()
      in walk body
    ) fundefs
  done;
  (* Pre-pass: declare all functions with correct LLVM types.
     Names are mangled to _ch_<name> to prevent libc symbol conflicts: LLVM may lower
     llvm.floor.f64 to `call floor` on generic targets; without mangling, the linker
     resolves that to our @floor, causing infinite recursion. *)
  List.iter (fun (name, args, _body) ->
    let arg_types = List.mapi
      (fun i _ -> if Hashtbl.mem list_arg_tbl (name, i) then p else f64) args in
    let ret_type = if StringSet.mem name !list_fns then p else f64 in
    let ft = Llvm.function_type ret_type (Array.of_list arg_types) in
    let llvm_name = "_ch_" ^ name in
    let fn = Llvm.define_function llvm_name ft md in
    Llvm.set_linkage Llvm.Linkage.Internal fn;
    Hashtbl.add known_fns name (ft, fn)
  ) fundefs;
  (* Compilation pass: fill function bodies *)
  List.iter (fun (name, args, body) ->
    ignore (compile_fundef ctx md known_fns name args body)
  ) fundefs;
  (ctx, md, known_fns)

(* Generate @main function for top-level non-FunDef statements. *)
let compile_main_fn ctx md known_fns stmts =
  let f64 = Llvm.double_type ctx in
  let i32 = Llvm.i32_type ctx in
  let fn = Llvm.define_function "main" (Llvm.function_type i32 [||]) md in
  let abort_bb = Llvm.append_block ctx "abort" fn in
  let builder = Llvm.builder_at_end ctx (Llvm.entry_block fn) in
  let exit_ft = Llvm.function_type (Llvm.void_type ctx) [| i32 |] in
  let exit_fn = match Llvm.lookup_function "exit" md with
    | Some f -> f | None -> Llvm.declare_function "exit" exit_ft md in
  let ab = Llvm.builder_at_end ctx abort_bb in
  ignore (Llvm.build_call exit_ft exit_fn [| Llvm.const_int i32 1 |] "" ab);
  ignore (Llvm.build_unreachable ab);
  let _env = List.fold_left (fun env stmt ->
    match stmt with
    | FunDef _ -> env
    | Let (name, expr) ->
        let v = compile_expr ctx md builder known_fns env false expr in
        StringMap.add name v env
    | Assert expr ->
        let v = compile_expr ctx md builder known_fns env false expr in
        let cond = Llvm.build_fcmp Llvm.Fcmp.One v (Llvm.const_float f64 0.0) "ac" builder in
        let ok_bb = Llvm.append_block ctx "assert_ok" fn in
        ignore (Llvm.build_cond_br cond ok_bb abort_bb builder);
        Llvm.position_at_end ok_bb builder;
        env
    | expr ->
        ignore (compile_expr ctx md builder known_fns env false expr);
        env
  ) StringMap.empty stmts in
  ignore (Llvm.build_ret (Llvm.const_int i32 0) builder);
  fn

let setup_target_machine () =
  let triple = Llvm_target.Target.default_triple () in
  let target = Llvm_target.Target.by_triple triple in
  Llvm_target.TargetMachine.create ~triple ~cpu:"generic" ~features:""
    ~level:Llvm_target.CodeGenOptLevel.Default
    ~reloc_mode:Llvm_target.RelocMode.PIC
    ~code_model:Llvm_target.CodeModel.Default target

(* Load FunDef nodes from a stdlib file for the native compile pipeline. *)
let load_stdlib_fundefs filename =
  let possible_paths = [
    Filename.concat !source_dir "lib";
    Filename.concat (Filename.dirname !source_dir) "lib";
    Filename.concat (Filename.dirname Sys.argv.(0)) "../lib";
    Filename.concat (Filename.dirname Sys.argv.(0)) "../../src/lib";
    Sys.getcwd () ^ "/src/lib";
  ] in
  let rec find = function
    | [] -> []
    | dir :: rest ->
        let path = Filename.concat dir filename in
        if Sys.file_exists path then begin
          let ic = open_in path in
          let src = really_input_string ic (in_channel_length ic) in
          close_in ic;
          List.filter (function FunDef _ -> true | _ -> false)
            (Parser.parse src)
        end else find rest
  in
  find possible_paths

let find_runtime_lib () =
  let candidates = [
    Filename.concat (Filename.dirname Sys.argv.(0)) "libchuring_runtime_native.a";
    Filename.concat (Sys.getcwd ()) "_build/default/src/libchuring_runtime_native.a";
    Filename.concat (Filename.dirname Sys.argv.(0)) "../libchuring_runtime_native.a";
  ] in
  List.find_opt Sys.file_exists candidates

let compile_to_binary ?(output="a.out") exprs =
  ensure_initialized ();
  let stdlib = load_stdlib_fundefs "math.ch" @ load_stdlib_fundefs "prelude.ch" in
  let all_exprs = stdlib @ exprs in
  let (ctx, md, known_fns) = compile_module all_exprs in
  let non_fundefs = List.filter (function FunDef _ -> false | _ -> true) exprs in
  if non_fundefs <> [] then
    ignore (compile_main_fn ctx md known_fns non_fundefs);
  let tm = setup_target_machine () in
  Llvm.set_target_triple (Llvm_target.TargetMachine.triple tm) md;
  Llvm.set_data_layout
    (Llvm_target.DataLayout.as_string (Llvm_target.TargetMachine.data_layout tm)) md;
  Llvm_analysis.assert_valid_module md;
  let tmp_o = Filename.temp_file "churing" ".o" in
  Llvm_target.TargetMachine.emit_to_file md
    Llvm_target.CodeGenFileType.ObjectFile tmp_o tm;
  let rt_flag = match find_runtime_lib () with
    | Some p -> " " ^ Filename.quote p | None -> "" in
  let cmd = Printf.sprintf "cc %s%s -lgc -lm -o %s"
    (Filename.quote tmp_o) rt_flag (Filename.quote output) in
  (match Unix.system cmd with
   | Unix.WEXITED 0 -> ()
   | _ -> failwith ("compile_to_binary: linker failed: " ^ cmd));
  Sys.remove tmp_o;
  Llvm.dispose_module md;
  Llvm.dispose_context ctx

(* ── JIT infrastructure for interpreter fallback (llvm intrinsic calls) ─────── *)

type jit_fn =
  | F1 of nativeint
  | F2 of nativeint

let make_jit_fn fn_name arity ee =
  let addr = get_fn_addr fn_name ee in
  match arity with
  | 1 -> F1 addr
  | 2 -> F2 addr
  | n -> failwith (Printf.sprintf "codegen: LLVM JIT intrinsic arity %d not supported" n)

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
      let known_fns : (string, Llvm.lltype * Llvm.llvalue) Hashtbl.t = Hashtbl.create 4 in
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
