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

(* Cons cell struct: { double head, ptr tail }; nil = null ptr *)
let cons_cell_ty ctx =
  Llvm.struct_type ctx [| Llvm.double_type ctx; ptr_ty ctx |]

let fresh_lam () =
  let n = !lam_counter in
  incr lam_counter;
  Printf.sprintf "_lam_%d" n

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

(* Wrap a 1-arg f64→f64 named function as a closure *)
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

(* ── Static type analysis for pre-pass ─────────────────────────────────────── *)

(* Check if arg_name is used as a list (cons/nil match) in body *)
let rec arg_is_list_in_body arg_name = function
  | Match (Var x, arms) when x = arg_name ->
      List.exists (fun (pat, _) ->
        match pat with PCons _ | PList _ -> true | _ -> false) arms
  | Seq (a, b) -> arg_is_list_in_body arg_name a || arg_is_list_in_body arg_name b
  | App (f, a) -> arg_is_list_in_body arg_name f || arg_is_list_in_body arg_name a
  | Lam (x, body) when x <> arg_name -> arg_is_list_in_body arg_name body
  | Let (_, v) -> arg_is_list_in_body arg_name v
  | _ -> false

(* Vars introduced as list by a cons pattern's tail binding *)
let rec cons_tail_vars = function
  | PCons (_, PVar t) -> StringSet.singleton t
  | PCons (_, tp) -> cons_tail_vars tp
  | _ -> StringSet.empty

(* Check if body produces a list (ptr) result.
   list_fns: known list-returning function names.
   list_vars: local vars known to hold lists (from pattern bindings). *)
let rec body_returns_list_v list_fns list_vars = function
  | List _ -> true
  | Var x -> StringSet.mem x list_vars || StringSet.mem x list_fns
  | Match (_, arms) ->
      List.exists (fun (pat, body) ->
        let extra = cons_tail_vars pat in
        body_returns_list_v list_fns (StringSet.union list_vars extra) body) arms
  | Seq (_, b) -> body_returns_list_v list_fns list_vars b
  | App (f, _) ->
      (match flatten_app [] f with
       | Some ("cons", _) -> true
       | Some (name, _) -> StringSet.mem name list_fns
       | None -> false)
  | _ -> false

let body_returns_list list_fns body = body_returns_list_v list_fns StringSet.empty body

(* ── Main expression compiler ───────────────────────────────────────────────── *)
(* known_fns: name → (function_type, function_value)
   Storing ft explicitly avoids relying on Llvm.type_of which may return ptr in LLVM 18.
   in_tail: true when this expression is in tail position — enables tail call marking.
   Returns f64 for numbers/bools, closure struct for lambdas, ptr for lists. *)
let rec compile_expr ctx md builder (known_fns : (string, Llvm.lltype * Llvm.llvalue) Hashtbl.t) env in_tail = function
  | Int n   -> Llvm.const_float (Llvm.double_type ctx) (float_of_int n)
  | Float f -> Llvm.const_float (Llvm.double_type ctx) f
  | Bool b  -> Llvm.const_float (Llvm.double_type ctx) (if b then 1.0 else 0.0)

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
      let top = Hashtbl.fold (fun n _ s -> StringSet.add n s) known_fns StringSet.empty in
      let fv_set = free_vars (StringSet.add x top) body in
      let fv = StringSet.elements fv_set in
      let lam_name = fresh_lam () in
      let lft = Llvm.function_type f64 [| f64; p |] in
      let lam_fn = Llvm.define_function lam_name lft md in
      let lbb = Llvm.entry_block lam_fn in
      let lb  = Llvm.builder_at_end ctx lbb in
      Llvm.set_value_name x (Llvm.params lam_fn).(0);
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
      let result = compile_expr ctx md lb known_fns lam_env true body in
      (if Llvm.classify_type (Llvm.type_of result) <> Llvm.TypeKind.Double then
        failwith "codegen: lambda body must return f64 (list-returning lambdas deferred to #95)");
      ignore (Llvm.build_ret result lb);
      let env_ptr =
        if fv = [] then Llvm.const_null p
        else begin
          let n    = List.length fv in
          let size = Llvm.const_int (Llvm.i64_type ctx) (n * 8) in
          let mfn  = get_gc_malloc ctx md in
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
                  call_closure ctx builder fv av
                end
            | None ->
                let fv = compile_expr ctx md builder known_fns env false e_fn in
                let av = compile_expr ctx md builder known_fns env false e_arg in
                call_closure ctx builder fv av)
       | None ->
           let fv = compile_expr ctx md builder known_fns env false e_fn in
           let av = compile_expr ctx md builder known_fns env false e_arg in
           call_closure ctx builder fv av)

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
      let cmp = Llvm.build_fcmp Llvm.Fcmp.Oeq
        (compile_expr ctx md builder known_fns env false a)
        (compile_expr ctx md builder known_fns env false b)
        "eq" builder in
      Llvm.build_uitofp cmp (Llvm.double_type ctx) "eqf" builder

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

(* Compile a named function body into its pre-declared (or freshly declared) LLVM function. *)
let compile_fundef ctx md known_fns name args body =
  let f64 = Llvm.double_type ctx in
  let p = ptr_ty ctx in
  let fn, ft = match Llvm.lookup_function name md with
    | Some f ->
        (* Already declared by the pre-pass — retrieve the stored function type. *)
        (match Hashtbl.find_opt known_fns name with
         | Some (ft, _) -> (f, ft)
         | None ->
             (* JIT path: function appears in module but not in known_fns yet *)
             let arg_types = List.map
               (fun a -> if arg_is_list_in_body a body then p else f64) args in
             let ret_type = if body_returns_list StringSet.empty body then p else f64 in
             let ft = Llvm.function_type ret_type (Array.of_list arg_types) in
             Hashtbl.add known_fns name (ft, f);
             (f, ft))
    | None ->
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

(* Compile a list of top-level expressions into an LLVM module.
   Two-pass: pre-declare all functions (enabling mutual recursion), then compile bodies.
   An additional fixpoint determines which functions return list (ptr) vs f64. *)
let compile_module exprs =
  let ctx = Llvm.create_context () in
  let md  = Llvm.create_module ctx "churing" in
  let known_fns : (string, Llvm.lltype * Llvm.llvalue) Hashtbl.t = Hashtbl.create 16 in
  let f64 = Llvm.double_type ctx in
  let p = ptr_ty ctx in
  (* Fixpoint: find all list-returning functions *)
  let list_fns = ref StringSet.empty in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter (function
      | FunDef (name, _, body) ->
          if not (StringSet.mem name !list_fns) && body_returns_list !list_fns body then begin
            list_fns := StringSet.add name !list_fns;
            changed := true
          end
      | _ -> ()
    ) exprs
  done;
  (* Pre-pass: declare all functions with correct LLVM types *)
  List.iter (function
    | FunDef (name, args, body) ->
        let arg_types = List.map
          (fun a -> if arg_is_list_in_body a body then p else f64) args in
        let ret_type = if StringSet.mem name !list_fns then p else f64 in
        let ft = Llvm.function_type ret_type (Array.of_list arg_types) in
        let fn = Llvm.define_function name ft md in
        Hashtbl.add known_fns name (ft, fn)
    | _ -> ()
  ) exprs;
  (* Compilation pass: fill function bodies *)
  List.iter (function
    | FunDef (name, args, body) ->
        ignore (compile_fundef ctx md known_fns name args body)
    | _ -> ()
  ) exprs;
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
    ~reloc_mode:Llvm_target.RelocMode.Default
    ~code_model:Llvm_target.CodeModel.Default target

let compile_to_binary ?(output="a.out") exprs =
  ensure_initialized ();
  let (ctx, md, known_fns) = compile_module exprs in
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
  let cmd = Printf.sprintf "cc %s -lgc -lm -o %s" (Filename.quote tmp_o) (Filename.quote output) in
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
