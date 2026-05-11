let contains s sub =
  let ls = String.length s and lsub = String.length sub in
  if lsub > ls then false
  else
    let rec loop i =
      if i > ls - lsub then false
      else if String.sub s i lsub = sub then true
      else loop (i + 1)
    in loop 0

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

let () =
  Alcotest.run "llvm" [
    "codegen", [
      Alcotest.test_case "compile_module emits valid IR" `Quick test_compile_module;
      Alcotest.test_case "arithmetic lowers to fadd"    `Quick test_arithmetic;
    ]
  ]
