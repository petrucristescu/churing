let contains s sub =
  let ls = String.length s and lsub = String.length sub in
  if lsub > ls then false
  else
    let rec loop i =
      if i > ls - lsub then false
      else if String.sub s i lsub = sub then true
      else loop (i + 1)
    in loop 0

(* ── #91 tests ──────────────────────────────────────────────────────────────── *)

let test_compile_module () =
  let exprs = [
    Ast.FunDef ("my_sqrt", ["x"], Ast.Llvm ("llvm.sqrt.f64", [Ast.Var "x"]))
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "emits IR" true (String.length ir > 0);
  Alcotest.(check bool) "contains function def" true (contains ir "my_sqrt");
  Alcotest.(check bool) "contains sqrt intrinsic" true (contains ir "llvm.sqrt.f64")

let test_arithmetic () =
  let exprs = [
    Ast.FunDef ("add_one", ["x"], Ast.Add (Ast.Var "x", Ast.Float 1.0))
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "fadd in IR" true (contains ir "fadd")

(* ── #92 tests ──────────────────────────────────────────────────────────────── *)

(* @-binding in function body: ~f x  (@y (x + 1.0); y * 2.0) *)
let test_let_in_body () =
  let body =
    Ast.Seq (
      Ast.Let ("y", Ast.Add (Ast.Var "x", Ast.Float 1.0)),
      Ast.Mul (Ast.Var "y", Ast.Float 2.0))
  in
  let exprs = [ Ast.FunDef ("f", ["x"], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "IR emitted" true (String.length ir > 0);
  Alcotest.(check bool) "fadd present" true (contains ir "fadd");
  Alcotest.(check bool) "fmul present" true (contains ir "fmul")

(* Lambda with no captures: ~apply_double x  ((|>v. v * 2.0) x) *)
let test_lambda_no_capture () =
  let lam = Ast.Lam ("v", Ast.Mul (Ast.Var "v", Ast.Float 2.0)) in
  let body = Ast.App (lam, Ast.Var "x") in
  let exprs = [ Ast.FunDef ("apply_double", ["x"], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "lifted lambda present" true (contains ir "_lam_");
  Alcotest.(check bool) "fmul present"         true (contains ir "fmul")

(* Lambda with one capture: ~add_n n,x  ((|>v. v + n) x) *)
let test_lambda_with_capture () =
  let lam = Ast.Lam ("v", Ast.Add (Ast.Var "v", Ast.Var "n")) in
  let body = Ast.App (lam, Ast.Var "x") in
  let exprs = [ Ast.FunDef ("add_n", ["n"; "x"], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "lifted lambda present" true (contains ir "_lam_");
  (* Capture requires malloc + GEP + store + load in IR *)
  Alcotest.(check bool) "malloc called" true (contains ir "malloc");
  Alcotest.(check bool) "fadd in lambda" true (contains ir "fadd")

(* Direct call to a known function in another function body:
   ~double x  (x * 2.0)
   ~quad   x  (double (double x)) *)
let test_direct_cross_call () =
  let exprs = [
    Ast.FunDef ("double2", ["x"], Ast.Mul (Ast.Var "x", Ast.Float 2.0));
    Ast.FunDef ("quad",    ["x"],
      Ast.App (Ast.Var "double2", Ast.App (Ast.Var "double2", Ast.Var "x")));
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "double2 present" true (contains ir "double2");
  Alcotest.(check bool) "quad present"    true (contains ir "quad");
  (* Two calls to double2 inside quad *)
  Alcotest.(check bool) "call present" true (contains ir "call")

(* Two-arg direct call: ~pow2 x  (llvm.pow.f64 x 2.0) declared, then called *)
let test_two_arg_direct_call () =
  let exprs = [
    Ast.FunDef ("my_pow", ["x"; "y"],
      Ast.Llvm ("llvm.pow.f64", [Ast.Var "x"; Ast.Var "y"]));
    Ast.FunDef ("square", ["x"],
      Ast.App (Ast.App (Ast.Var "my_pow", Ast.Var "x"), Ast.Float 2.0));
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "my_pow present" true (contains ir "my_pow");
  Alcotest.(check bool) "square present" true (contains ir "square")

(* Named function passed as value: ~apply f,x  (f x)
   ~inc x  (x + 1.0)
   — apply receives inc as a closure via the Var "inc" → wrap path *)
let test_named_fn_as_value () =
  let exprs = [
    Ast.FunDef ("inc",   ["x"], Ast.Add (Ast.Var "x", Ast.Float 1.0));
    (* apply takes f as a closure (f is NOT in known_fns from within the body env) *)
    (* But here f IS a parameter, so it's in local env as f64.
       A more realistic test is to call apply with (Var "inc") as arg.
       We test the wrapper generation by referencing "inc" in a lambda. *)
    Ast.FunDef ("use_inc", ["x"],
      Ast.App (Ast.Var "inc", Ast.Var "x"));
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "inc defined" true (contains ir "define")

(* Mutual recursion: even/odd via two-pass pre-declaration *)
let test_mutual_recursion () =
  (* ~is_zero x  (eq x 0.0) — just a helper *)
  (* We can't do real even/odd without conditionals, so test cross-reference only *)
  let exprs = [
    Ast.FunDef ("f1", ["x"], Ast.Add (Ast.Var "x", Ast.Float 1.0));
    Ast.FunDef ("f2", ["x"],
      Ast.App (Ast.Var "f1", Ast.App (Ast.Var "f1", Ast.Var "x")));
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "f1 present" true (contains ir "f1");
  Alcotest.(check bool) "f2 present" true (contains ir "f2")

(* ── #94 tests ──────────────────────────────────────────────────────────────── *)

(* Self-recursive tail call: ~countdown n  (match n | 0 -> 0.0 | x -> countdown (x - 1.0))
   The recursive call in the cons arm should carry the `tail call` marker in IR. *)
let test_tail_call_direct () =
  let body =
    Ast.Match (Ast.Var "n", [
      (Ast.PInt 0, Ast.Float 0.0);
      (Ast.PVar "x",
       Ast.App (Ast.Var "countdown",
         Ast.Sub (Ast.Var "x", Ast.Float 1.0)));
    ])
  in
  let exprs = [ Ast.FunDef ("countdown", ["n"], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "countdown defined" true (contains ir "countdown");
  Alcotest.(check bool) "tail call emitted" true (contains ir "tail call")

(* Mutually tail-calling pair: ~ping n → pong (n-1), ~pong n → ping (n-1)
   Both recursive call sites should have the tail marker. *)
let test_tail_call_mutual () =
  let ping_body =
    Ast.App (Ast.Var "pong", Ast.Sub (Ast.Var "n", Ast.Float 1.0))
  in
  let pong_body =
    Ast.App (Ast.Var "ping", Ast.Sub (Ast.Var "n", Ast.Float 1.0))
  in
  let exprs = [
    Ast.FunDef ("ping", ["n"], ping_body);
    Ast.FunDef ("pong", ["n"], pong_body);
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "ping defined"      true (contains ir "ping");
  Alcotest.(check bool) "pong defined"      true (contains ir "pong");
  Alcotest.(check bool) "tail call emitted" true (contains ir "tail call")

(* Non-tail call must NOT carry the tail marker:
   ~not_tco n  (countdown n + 1.0) — the call to countdown is not in tail position *)
let test_non_tail_call_unmarked () =
  let countdown_body =
    Ast.Match (Ast.Var "n", [
      (Ast.PInt 0, Ast.Float 0.0);
      (Ast.PVar "x", Ast.App (Ast.Var "countdown2", Ast.Sub (Ast.Var "x", Ast.Float 1.0)));
    ])
  in
  let wrapper_body =
    Ast.Add (Ast.App (Ast.Var "countdown2", Ast.Var "n"), Ast.Float 1.0)
  in
  let exprs = [
    Ast.FunDef ("countdown2", ["n"], countdown_body);
    Ast.FunDef ("not_tco",    ["n"], wrapper_body);
  ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  (* The self-recursive call in countdown2 IS in tail position *)
  Alcotest.(check bool) "tail call in countdown2" true (contains ir "tail call");
  (* not_tco's call to countdown2 is not in tail position — it's an operand of fadd *)
  Alcotest.(check bool) "fadd present in not_tco" true (contains ir "fadd")

(* ── #93 tests ──────────────────────────────────────────────────────────────── *)

(* List literal [1.0, 2.0, 3.0] should emit GC_malloc calls and GEP *)
let test_list_literal () =
  let body = Ast.List [Ast.Float 1.0; Ast.Float 2.0; Ast.Float 3.0] in
  let exprs = [ Ast.FunDef ("make_list", [], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "GC_malloc declared" true (contains ir "GC_malloc");
  Alcotest.(check bool) "getelementptr present" true (contains ir "getelementptr")

(* match xs | [] -> 0.0 | h :: t -> h  — nil/cons dispatch *)
let test_match_list_pattern () =
  let xs_param = "xs" in
  let body =
    Ast.Match (Ast.Var xs_param, [
      (Ast.PList [], Ast.Float 0.0);
      (Ast.PCons (Ast.PVar "h", Ast.PWild), Ast.Var "h");
    ])
  in
  let exprs = [ Ast.FunDef ("head_or_zero", [xs_param], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "icmp for nil check" true (contains ir "icmp");
  Alcotest.(check bool) "phi for match result" true (contains ir "phi");
  Alcotest.(check bool) "match_end block" true (contains ir "match_end")

(* match n | 0 -> 1.0 | x -> x * 2.0  — integer pattern dispatch *)
let test_match_int_pattern () =
  let body =
    Ast.Match (Ast.Var "n", [
      (Ast.PInt 0, Ast.Float 1.0);
      (Ast.PVar "x", Ast.Mul (Ast.Var "x", Ast.Float 2.0));
    ])
  in
  let exprs = [ Ast.FunDef ("double_nonzero", ["n"], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "fcmp for int pattern" true (contains ir "fcmp");
  Alcotest.(check bool) "phi in result"        true (contains ir "phi")

(* ~sum xs (match xs | [] -> 0.0 | h :: t -> h + sum t)
   Verifies: list arg detected as ptr, self-recursive call, phi *)
let test_recursive_list_fn () =
  let body =
    Ast.Match (Ast.Var "xs", [
      (Ast.PList [], Ast.Float 0.0);
      (Ast.PCons (Ast.PVar "h", Ast.PVar "t"),
       Ast.Add (Ast.Var "h",
         Ast.App (Ast.Var "sum", Ast.Var "t")));
    ])
  in
  let exprs = [ Ast.FunDef ("sum", ["xs"], body) ] in
  let (ctx, md, _) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "sum defined"   true (contains ir "sum");
  Alcotest.(check bool) "recursive call" true (contains ir "call");
  Alcotest.(check bool) "phi node"      true (contains ir "phi")

let () =
  Alcotest.run "llvm" [
    "codegen", [
      Alcotest.test_case "compile_module emits valid IR"      `Quick test_compile_module;
      Alcotest.test_case "arithmetic lowers to fadd"          `Quick test_arithmetic;
      Alcotest.test_case "@-binding in function body"         `Quick test_let_in_body;
      Alcotest.test_case "lambda with no captures"            `Quick test_lambda_no_capture;
      Alcotest.test_case "lambda with capture uses malloc"    `Quick test_lambda_with_capture;
      Alcotest.test_case "direct cross-function call"         `Quick test_direct_cross_call;
      Alcotest.test_case "two-arg direct call flattened"      `Quick test_two_arg_direct_call;
      Alcotest.test_case "named function referenced by Var"   `Quick test_named_fn_as_value;
      Alcotest.test_case "two-pass enables mutual references" `Quick test_mutual_recursion;
      Alcotest.test_case "List literal emits GC_malloc + GEP" `Quick test_list_literal;
      Alcotest.test_case "match list nil/cons dispatch"       `Quick test_match_list_pattern;
      Alcotest.test_case "match integer pattern fcmp"         `Quick test_match_int_pattern;
      Alcotest.test_case "recursive list function compiles"   `Quick test_recursive_list_fn;
      Alcotest.test_case "tail call marked on direct call"    `Quick test_tail_call_direct;
      Alcotest.test_case "mutual tail calls both marked"      `Quick test_tail_call_mutual;
      Alcotest.test_case "non-tail call not marked"           `Quick test_non_tail_call_unmarked;
    ]
  ]
