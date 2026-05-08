let contains s sub =
  let ls = String.length s and lsub = String.length sub in
  if lsub > ls then false
  else
    let rec loop i =
      if i > ls - lsub then false
      else if String.sub s i lsub = sub then true
      else loop (i + 1)
    in loop 0

let test_smoke () =
  let ir = Codegen.smoke_test () in
  Alcotest.(check bool) "emits IR" true (String.length ir > 0);
  Alcotest.(check bool) "contains sqrt intrinsic" true
    (contains ir "llvm.sqrt.f64")

let () =
  Alcotest.run "llvm" [
    "smoke", [
      Alcotest.test_case "LLVM bindings functional" `Quick test_smoke
    ]
  ]
