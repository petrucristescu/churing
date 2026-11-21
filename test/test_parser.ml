let parse s = Parser.parse s

let test_int () =
  match parse "42" with
  | [Ast.Int 42] -> ()
  | _ -> Alcotest.fail "expected Int 42"

let test_long () =
  match parse "42L" with
  | [Ast.Lng 42L] -> ()
  | _ -> Alcotest.fail "expected Lng 42L"

let test_float () =
  match parse "3.14" with
  | [Ast.Float f] -> Alcotest.(check bool) "float" true (Float.equal f 3.14)
  | _ -> Alcotest.fail "expected Float 3.14"

let test_string () =
  match parse "\"hello\"" with
  | [Ast.Str "hello"] -> ()
  | _ -> Alcotest.fail "expected Str hello"

let test_string_escapes () =
  (* Churing source:  "say \"hi\"\n\tback\\slash" *)
  match parse {|"say \"hi\"\n\tback\\slash"|} with
  | [Ast.Str "say \"hi\"\n\tback\\slash"] -> ()
  | _ -> Alcotest.fail "expected escapes to be decoded"

let test_multiline_app_in_parens () =
  match parse "(f a\n  b)" with
  | [Ast.App (Ast.App (Ast.Var "f", Ast.Var "a"), Ast.Var "b")] -> ()
  | _ -> Alcotest.fail "expected f a b as one application"

let test_newline_ends_toplevel_app () =
  match parse "f a\nb" with
  | [Ast.App (Ast.Var "f", Ast.Var "a"); Ast.Var "b"] -> ()
  | _ -> Alcotest.fail "expected two top-level expressions"

let test_bool_true () =
  match parse "true" with
  | [Ast.Bool true] -> ()
  | _ -> Alcotest.fail "expected Bool true"

let test_bool_false () =
  match parse "false" with
  | [Ast.Bool false] -> ()
  | _ -> Alcotest.fail "expected Bool false"

let test_add () =
  match parse "2 + 3" with
  | [Ast.Add (Ast.Int 2, Ast.Int 3)] -> ()
  | _ -> Alcotest.fail "expected Add(2, 3)"

let test_precedence_mul_add () =
  match parse "2 * 3 + 4" with
  | [Ast.Add (Ast.Mul (Ast.Int 2, Ast.Int 3), Ast.Int 4)] -> ()
  | e -> Alcotest.fail ("expected (2*3)+4, got: " ^ Ast.string_of_exprs e)

let test_left_assoc_sub () =
  match parse "5 - 3 - 1" with
  | [Ast.Sub (Ast.Sub (Ast.Int 5, Ast.Int 3), Ast.Int 1)] -> ()
  | e -> Alcotest.fail ("expected (5-3)-1, got: " ^ Ast.string_of_exprs e)

let test_fundef () =
  match parse "~add x,y x + y" with
  | [Ast.FunDef ("add", ["x"; "y"], Ast.Add (Ast.Var "x", Ast.Var "y"))] -> ()
  | _ -> Alcotest.fail "expected FunDef add"

let test_lambda () =
  match parse "|>x. x + 1" with
  | [Ast.Lam ("x", Ast.Add (Ast.Var "x", Ast.Int 1))] -> ()
  | _ -> Alcotest.fail "expected Lam"

let test_let_int () =
  match parse "@x 5" with
  | [Ast.Let ("x", Ast.Int 5)] -> ()
  | _ -> Alcotest.fail "expected Let x = 5"

let test_assert () =
  match parse "assert true" with
  | [Ast.Assert (Ast.Bool true)] -> ()
  | _ -> Alcotest.fail "expected Assert true"

let test_eq () =
  match parse "eq 1 2" with
  | [Ast.Eq (Ast.Int 1, Ast.Int 2)] -> ()
  | _ -> Alcotest.fail "expected Eq(1, 2)"

let test_app () =
  match parse "f 1 2" with
  | [Ast.App (Ast.App (Ast.Var "f", Ast.Int 1), Ast.Int 2)] -> ()
  | _ -> Alcotest.fail "expected App(App(f, 1), 2)"

let test_import () =
  match parse "import operators" with
  | [Ast.Import "operators"] -> ()
  | _ -> Alcotest.fail "expected Import operators"

let () =
  Alcotest.run "Parser" [
    "literals", [
      Alcotest.test_case "int" `Quick test_int;
      Alcotest.test_case "long" `Quick test_long;
      Alcotest.test_case "float" `Quick test_float;
      Alcotest.test_case "string" `Quick test_string;
      Alcotest.test_case "string escapes" `Quick test_string_escapes;
      Alcotest.test_case "multi-line app in parens" `Quick test_multiline_app_in_parens;
      Alcotest.test_case "newline ends top-level app" `Quick test_newline_ends_toplevel_app;
      Alcotest.test_case "bool true" `Quick test_bool_true;
      Alcotest.test_case "bool false" `Quick test_bool_false;
    ];
    "precedence", [
      Alcotest.test_case "mul before add" `Quick test_precedence_mul_add;
      Alcotest.test_case "left assoc sub" `Quick test_left_assoc_sub;
    ];
    "definitions", [
      Alcotest.test_case "fundef" `Quick test_fundef;
      Alcotest.test_case "lambda" `Quick test_lambda;
      Alcotest.test_case "let int" `Quick test_let_int;
    ];
    "keywords", [
      Alcotest.test_case "add" `Quick test_add;
      Alcotest.test_case "assert" `Quick test_assert;
      Alcotest.test_case "eq" `Quick test_eq;
      Alcotest.test_case "import" `Quick test_import;
      Alcotest.test_case "curried app" `Quick test_app;
    ];
  ]
