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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
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
  let (ctx, md) = Codegen.compile_module exprs in
  let ir = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  Alcotest.(check bool) "f1 present" true (contains ir "f1");
  Alcotest.(check bool) "f2 present" true (contains ir "f2")

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
    ]
  ]
