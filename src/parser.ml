open Types
open Ast

exception ParseError of string * int * int

type token =
  | Integer of int
  | Long of int64
  | Float of float
  | String of string
  | Plus
  | Minus
  | Multiply
  | Divide
  | Lambda
  | Ident of string
  | At
  | Underscore
  | TypeName of string
  | Dot
  | Tilde
  | LParen
  | RParen
  | Comma
  | LBracket
  | RBracket
  | LBrace
  | RBrace
  | Colon
  | Pipe
  | Arrow
  | ColonColon
  | Newline
  | Import  (* New token for import keyword *)
  | Eof

type pos_token = token * int * int

let tokenize s =
  let is_letter c =
    ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c = '_'
  in
  let is_ident_char c =
    is_letter c || ('0' <= c && c <= '9')
  in
  let is_digit c =
    '0' <= c && c <= '9'
  in
  let is_whitespace c =
    c = ' ' || c = '\t' || c = '\r'
  in
  let rec skip_single_line_comment i line col =
    if i >= String.length s || s.[i] = '\n' then (i, line, col)
    else skip_single_line_comment (i+1) line (col+1)
  in
  let rec skip_multiline_comment i line col =
    if i >= String.length s then (i, line, col)
    else if s.[i] = ')' then (i+1, line, col+1)
    else if s.[i] = '\n' then skip_multiline_comment (i+1) (line+1) 1
    else skip_multiline_comment (i+1) line (col+1)
  in
  let rec aux i line col acc =
    if i >= String.length s then List.rev ((Eof, line, col) :: acc)
    else if s.[i] = '#' then
      if i+1 < String.length s && s.[i+1] = '(' then
        let (j, line', col') = skip_multiline_comment (i+2) line (col+2) in
        aux j line' col' acc
      else
        let (j, line', col') = skip_single_line_comment (i+1) line (col+1) in
        aux j line' col' acc
    else if s.[i] = '\n' then
      aux (i+1) (line+1) 1 ((Newline, line, col) :: acc)
    else if s.[i] = '(' then
      aux (i+1) line (col+1) ((LParen, line, col) :: acc)
    else if s.[i] = ')' then
      aux (i+1) line (col+1) ((RParen, line, col) :: acc)
    else if s.[i] = '[' then
      aux (i+1) line (col+1) ((LBracket, line, col) :: acc)
    else if s.[i] = ']' then
      aux (i+1) line (col+1) ((RBracket, line, col) :: acc)
    else if s.[i] = '{' then
      aux (i+1) line (col+1) ((LBrace, line, col) :: acc)
    else if s.[i] = '}' then
      aux (i+1) line (col+1) ((RBrace, line, col) :: acc)
    else if s.[i] = '~' then
      aux (i+1) line (col+1) ((Tilde, line, col) :: acc)
    else if s.[i] = ',' then
      aux (i+1) line (col+1) ((Comma, line, col) :: acc)
    else if i + 1 < String.length s && s.[i] = '|' && s.[i+1] = '>' then
      aux (i+2) line (col+2) ((Lambda, line, col) :: acc)
    else if s.[i] = '|' then
      aux (i+1) line (col+1) ((Pipe, line, col) :: acc)
    else if i + 1 < String.length s && s.[i] = ':' && s.[i+1] = ':' then
      aux (i+2) line (col+2) ((ColonColon, line, col) :: acc)
    else if s.[i] = ':' then
      aux (i+1) line (col+1) ((Colon, line, col) :: acc)
    else if is_whitespace s.[i] then
      aux (i+1) line (col+1) acc
    else if s.[i] = '+' then
      aux (i+1) line (col+1) ((Plus, line, col) :: acc)
    else if i + 1 < String.length s && s.[i] = '-' && s.[i+1] = '>' then
      aux (i+2) line (col+2) ((Arrow, line, col) :: acc)
    else if s.[i] = '-' then
      aux (i+1) line (col+1) ((Minus, line, col) :: acc)
    else if s.[i] = '*' then
      aux (i+1) line (col+1) ((Multiply, line, col) :: acc)
    else if s.[i] = '/' then
      aux (i+1) line (col+1) ((Divide, line, col) :: acc)
    else if s.[i] = '@' then
      aux (i+1) line (col+1) ((At, line, col) :: acc)
    else if s.[i] = '_' then
      aux (i+1) line (col+1) ((Underscore, line, col) :: acc)
    else if is_letter s.[i] then
      let j = ref (i+1) in
      while !j < String.length s && is_ident_char s.[!j] do incr j done;
      let name = String.sub s i (!j-i) in
      let token =
        match name with
        | "true" -> Ident "__church_true"
        | "false" -> Ident "__church_false"
        | "import" -> Import  (* New token for import keyword *)
        | _ -> Ident name
      in
      aux !j line (col + (!j - i)) ((token, line, col) :: acc)
    else if is_digit s.[i] then
      let j = ref i in
      while !j < String.length s && is_digit s.[!j] do incr j done;

      if !j < String.length s && s.[!j] = '.' then (
        incr j;
        if !j < String.length s && is_digit s.[!j] then (
          while !j < String.length s && is_digit s.[!j] do incr j done;
          let f = float_of_string (String.sub s i (!j-i)) in
          aux !j line (col + (!j - i)) ((Float f, line, col) :: acc)
        ) else
          raise (ParseError ("Expected digit after decimal point", line, col))
      )
      else if !j < String.length s && (s.[!j] = 'l' || s.[!j] = 'L') then
        (* This is a long integer like 123L *)
        let n = Int64.of_string (String.sub s i (!j-i)) in
        aux (!j+1) line (col + (!j - i + 1)) ((Long n, line, col) :: acc)
      else if !j < String.length s && (s.[!j] = 'f' || s.[!j] = 'F') then
        (* This is a float literal like 123f *)
        let f = float_of_string (String.sub s i (!j-i)) in
        aux (!j+1) line (col + (!j - i + 1)) ((Float f, line, col) :: acc)
      else
        (* This is a regular integer *)
        let n = int_of_string (String.sub s i (!j-i)) in
        aux !j line (col + (!j - i)) ((Integer n, line, col) :: acc)
    else if s.[i] = '.' && (i + 1) < String.length s && is_digit s.[i+1] then
      (* This is a float starting with a dot like .123 *)
      let j = ref (i+1) in
      while !j < String.length s && is_digit s.[!j] do incr j done;
      let f = float_of_string (String.sub s i (!j-i)) in
      aux !j line (col + (!j - i)) ((Float f, line, col) :: acc)
    else if s.[i] = '.' then
      aux (i+1) line (col+1) ((Dot, line, col) :: acc)
    else if s.[i] = '"' then
      let j = ref (i+1) in
      while !j < String.length s && s.[!j] <> '"' do incr j done;
      if !j >= String.length s then
        raise (ParseError ("Unterminated string literal", line, col))
      else
        let str = String.sub s (i+1) (!j - i - 1) in
        aux (!j+1) line (col + (!j - i + 1)) ((String str, line, col) :: acc)
    else
      raise (ParseError ("Unknown char: " ^ String.make 1 s.[i], line, col))
  in
  aux 0 1 1 []

(* Helper function for skipping newline tokens *)
let rec skip_newlines = function
  | (Newline, _, _) :: rest -> skip_newlines rest
  | tokens -> tokens

(* Pattern parsing for match expressions *)
let rec parse_pattern tokens =
  let pat, rest = parse_pattern_atom tokens in
  (* Check for :: (cons pattern) *)
  match rest with
  | (ColonColon, _, _) :: rest' ->
      let tail_pat, rest'' = parse_pattern rest' in
      (PCons (pat, tail_pat), rest'')
  | _ -> (pat, rest)

and parse_pattern_atom tokens =
  match tokens with
  | (Integer n, _, _) :: rest -> (PInt n, rest)
  | (String s, _, _) :: rest -> (PStr s, rest)
  | (Ident "__church_true", _, _) :: rest -> (PBool true, rest)
  | (Ident "__church_false", _, _) :: rest -> (PBool false, rest)
  | (Underscore, _, _) :: rest -> (PWild, rest)
  | (Ident name, _, _) :: rest -> (PVar name, rest)
  | (LBracket, _, _) :: rest ->
      let rec parse_list_pats toks acc =
        let toks = skip_newlines toks in
        match toks with
        | (RBracket, _, _) :: rest' -> (List.rev acc, rest')
        | _ ->
            let p, rest' = parse_pattern toks in
            let rest' = skip_newlines rest' in
            (match rest' with
             | (Comma, _, _) :: rest'' -> parse_list_pats rest'' (p :: acc)
             | (RBracket, _, _) :: rest'' -> (List.rev (p :: acc), rest'')
             | (_, line, col) :: _ -> raise (ParseError ("Expected ',' or ']' in pattern", line, col))
             | [] -> raise (ParseError ("Unterminated list pattern", 1, 1)))
      in
      let pats, rest' = parse_list_pats rest [] in
      (PList pats, rest')
  | (tok, line, col) :: _ -> raise (ParseError ("Invalid pattern", line, col))
  | [] -> raise (ParseError ("Expected pattern", 1, 1))

(* Define all parsing functions with proper mutual recursion *)
let rec parse_expr tokens = parse_fun_def tokens

and parse_fun_def tokens =
  match tokens with
  | (Tilde, _, _) :: (Import, _, _) :: (Ident library, _, _) :: rest ->
      let import_expr = Ast.Import library in
      (import_expr, rest)  (* Handle ~import syntax *)
  | (Tilde, _, _) :: (Ident "import", _, _) :: (Ident library, _, _) :: rest ->
      let import_expr = Ast.Import library in
      (import_expr, rest)  (* Alternative ~import syntax *)
  | (Tilde, _, _) :: (LParen, _, _) :: rest ->
      let body, rest' = parse_seq rest in
      (FunDef ("main", [], body), rest')
  | (Tilde, _, _) :: (Ident name, _, _) :: rest ->
      let rec parse_args toks acc =
        match toks with
        | (LParen, _, _) :: _ -> (List.rev acc, toks)
        | (Ident arg, _, _) :: (Comma, _, _) :: rest' -> parse_args rest' (arg :: acc)
        | (Ident arg, _, _) :: rest' -> (List.rev (arg :: acc), rest')
        | (tok, line, col) :: _ -> raise (ParseError ("Expected argument name", line, col))
        | [] -> raise (ParseError ("Expected argument name", 1, 1))
      in
      let args, rest' = parse_args rest [] in
      let body, rest'' =
        match rest' with
        | (LParen, _, _) :: rest'' -> parse_seq rest''
        | _ -> parse_add rest'
      in
      (FunDef (name, args, body), rest'')
  | _ -> parse_var_def tokens

and parse_seq tokens =
  let rec aux toks acc =
    let toks = skip_newlines toks in
    match toks with
    | (RParen, _, _) :: rest -> (List.rev acc, rest)
    | [] | (Eof, _, _) :: _ -> (List.rev acc, toks)
    | _ ->
        let expr, toks' = parse_expr toks in
        aux toks' (expr :: acc)
  in
  let exprs, rest = aux tokens [] in
  match exprs with
  | [] -> (Int 0, rest)
  | [e] -> (e, rest)
  | e1 :: es -> (List.fold_left (fun acc e -> Seq (acc, e)) e1 es, rest)

and parse_var_def tokens =
  match tokens with
  | (At, line, col) :: (Ident name, _, _) :: rest ->
      let value, rest' = parse_add rest in
      (Let (name, value), rest')
  | (At, _, _) :: (Underscore, _, _) :: rest ->
      (* @_ value — discard binding (run for side effects). Without this case the
         leading @ token is never consumed and the top-level loop recurses forever. *)
      let value, rest' = parse_add rest in
      (Let ("_", value), rest')
  | _ -> parse_add tokens

and parse_add tokens =
  let rec aux acc toks =
    match toks with
    | (Plus, _, _) :: rest' ->
        let rhs, rest'' = parse_mul rest' in
        aux (Add (acc, rhs)) rest''
    | (Minus, _, _) :: rest' ->
        let rhs, rest'' = parse_mul rest' in
        aux (Sub (acc, rhs)) rest''
    | _ -> (acc, toks)
  in
  let lhs, rest = parse_mul tokens in
  aux lhs rest

and parse_mul tokens =
  let rec aux acc toks =
    match toks with
    | (Multiply, _, _) :: rest' ->
        let rhs, rest'' = parse_app rest' in
        aux (Mul (acc, rhs)) rest''
    | (Divide, _, _) :: rest' ->
        let rhs, rest'' = parse_app rest' in
        aux (Div (acc, rhs)) rest''
    | _ -> (acc, toks)
  in
  let lhs, rest = parse_app tokens in
  aux lhs rest

and parse_app tokens =
  let rec aux acc toks =
    match toks with
    | (Integer _, _, _) :: _
    | (Long _, _, _) :: _
    | (Float _, _, _) :: _
    | (Ident _, _, _) :: _
    | (LParen, _, _) :: _
    | (LBracket, _, _) :: _
    | (LBrace, _, _) :: _
    | (String _, _, _) :: _ ->
        let arg, rest = parse_primary toks in
        aux (App (acc, arg)) rest
    | _ -> (acc, toks)
  in
  let atom, rest = parse_primary tokens in
  aux atom rest

and parse_primary tokens =
  match tokens with
  | (Import, line, col) :: (Ident library, _, _) :: rest ->
      (Import library, rest)  (* Handle import statements *)
  | (Import, line, col) :: (String library, _, _) :: rest ->
      (Import library, rest)  (* Handle import with string literals too *)
  | (Tilde, _, _) :: (Ident x, _, _) :: rest ->
      (Var ("~" ^ x), rest)  (* Allow ~foo as a variable reference in expressions *)
  | (Lambda, _, _) :: (Ident x, _, _) :: (Dot, _, _) :: rest ->
      let body, rest' = parse_expr (skip_newlines rest) in
      (Lam (x, body), rest')
  | (Lambda, _, _) :: (Underscore, _, _) :: (Dot, _, _) :: rest ->
      let body, rest' = parse_expr (skip_newlines rest) in
      (Lam ("_", body), rest')
  | (Ident "__church_true", _, _) :: rest ->
      (Bool true, rest)
  | (Ident "__church_false", _, _) :: rest ->
      (Bool false, rest)
  | (Integer n, _, _) :: rest -> (Int n, rest)
  | (Long n, _, _) :: rest -> (Lng n, rest)
  | (Float f, _, _) :: rest -> (Float f, rest)
  | (Ident "match", _, _) :: rest ->
      let scrutinee, rest' = parse_primary (skip_newlines rest) in
      let rec parse_arms toks acc =
        let toks = skip_newlines toks in
        match toks with
        | (Pipe, _, _) :: rest'' ->
            let pat, rest''' = parse_pattern (skip_newlines rest'') in
            let rest''' = skip_newlines rest''' in
            (match rest''' with
             | (Arrow, _, _) :: rest'''' ->
                 let body, rest''''' = parse_expr (skip_newlines rest'''') in
                 parse_arms rest''''' ((pat, body) :: acc)
             | (_, line, col) :: _ -> raise (ParseError ("Expected '->' after pattern", line, col))
             | [] -> raise (ParseError ("Expected '->' after pattern", 1, 1)))
        | _ -> (List.rev acc, toks)
      in
      let arms, rest'' = parse_arms rest' [] in
      if arms = [] then raise (ParseError ("match requires at least one arm", 1, 1));
      (Match (scrutinee, arms), rest'')
  | (Ident "try", _, _) :: rest ->
      let expr, rest' = parse_primary (skip_newlines rest) in
      let handler, rest'' = parse_primary (skip_newlines rest') in
      (Try (expr, handler), rest'')
  | (Ident "assert", _, _) :: rest ->
      let arg, rest' = parse_expr (skip_newlines rest) in
      (Assert arg, rest')
  | (Ident "eq", _, _) :: rest ->
      let a, rest' = parse_primary (skip_newlines rest) in
      let b, rest'' = parse_primary (skip_newlines rest') in
      let rest''' = skip_newlines rest'' in
      (match rest''' with
       | (String s, _, _) :: rest'''' ->
           let f, rest''''' = parse_primary (skip_newlines rest'''') in
           (App (App (Eq (a, b), Str s), f), rest''''')
       | (LParen, _, _) :: _ ->
           let t, rest'''' = parse_primary rest''' in
           let f, rest''''' = parse_primary (skip_newlines rest'''') in
           (App (App (Eq (a, b), t), f), rest''''')
       | _ -> (Eq (a, b), rest'''))
  | (Ident "llvm", _, _) :: (String intrinsic, _, _) :: rest ->
      let rec collect_args toks acc =
        match toks with
        | (Integer _, _, _) :: _
        | (Long _, _, _) :: _
        | (Float _, _, _) :: _
        | (Ident _, _, _) :: _
        | (LParen, _, _) :: _
        | (LBracket, _, _) :: _
        | (LBrace, _, _) :: _
        | (String _, _, _) :: _ ->
            let arg, rest' = parse_primary toks in
            collect_args rest' (arg :: acc)
        | _ -> (List.rev acc, toks)
      in
      let args, rest' = collect_args rest [] in
      (Llvm (intrinsic, args), rest')
  | (Ident x, _, _) :: rest -> (Var x, rest)
  | (LParen, _, _) :: rest ->
      let rest = skip_newlines rest in
      let expr, rest' = parse_expr rest in
      let rest' = skip_newlines rest' in
      (match rest' with
       | (RParen, _, _) :: rest'' -> (expr, rest'')
       | (tok, line, col) :: _ ->
           raise (ParseError ("Expected closing parenthesis", line, col))
       | [] -> raise (ParseError ("Expected closing parenthesis", 1, 1)))
  | (String s, _, _) :: rest -> (Str s, rest)
  | (LBracket, _, _) :: rest ->
      let rec parse_list_items toks acc =
        let toks = skip_newlines toks in
        match toks with
        | (RBracket, _, _) :: rest' -> (List.rev acc, rest')
        | _ ->
            let item, rest' = parse_expr toks in
            let rest' = skip_newlines rest' in
            (match rest' with
             | (Comma, _, _) :: rest'' -> parse_list_items rest'' (item :: acc)
             | (RBracket, _, _) :: rest'' -> (List.rev (item :: acc), rest'')
             | (_, line, col) :: _ -> raise (ParseError ("Expected ',' or ']' in list", line, col))
             | [] -> raise (ParseError ("Unterminated list literal", 1, 1)))
      in
      let items, rest' = parse_list_items rest [] in
      (List items, rest')
  | (LBrace, _, _) :: rest ->
      let rec parse_dict_entries toks acc =
        let toks = skip_newlines toks in
        match toks with
        | (RBrace, _, _) :: rest' -> (List.rev acc, rest')
        | (Ident key, _, _) :: rest' | (String key, _, _) :: rest' ->
            let rest' = skip_newlines rest' in
            (match rest' with
             | (Colon, _, _) :: rest'' ->
                 let value, rest''' = parse_expr (skip_newlines rest'') in
                 let rest''' = skip_newlines rest''' in
                 (match rest''' with
                  | (Comma, _, _) :: rest'''' -> parse_dict_entries rest'''' ((key, value) :: acc)
                  | (RBrace, _, _) :: rest'''' -> (List.rev ((key, value) :: acc), rest'''')
                  | (_, line, col) :: _ -> raise (ParseError ("Expected ',' or '}' in dict", line, col))
                  | [] -> raise (ParseError ("Unterminated dict literal", 1, 1)))
             | (_, line, col) :: _ -> raise (ParseError ("Expected ':' after dict key", line, col))
             | [] -> raise (ParseError ("Expected ':' after dict key", 1, 1)))
        | (_, line, col) :: _ -> raise (ParseError ("Expected key in dict", line, col))
        | [] -> raise (ParseError ("Unterminated dict literal", 1, 1))
      in
      let entries, rest' = parse_dict_entries rest [] in
      (Dict entries, rest')
  | (At, _, _) :: _ -> parse_var_def tokens
  | (tok, line, col) :: _ -> raise (ParseError ("Invalid expression", line, col))
  | [] -> raise (ParseError ("Invalid expression", 1, 1))

let rec parse_all tokens acc =
  let tokens = skip_newlines tokens in
  match tokens with
  | [] | (Eof, _, _) :: _ -> List.rev acc
  | _ ->
      let expr, rest = parse_expr tokens in
      parse_all rest (expr :: acc)

let parse input =
  let tokens =
    try tokenize input
    with ParseError (msg, line, col) ->
      Printf.printf "Tokenization error at line %d, col %d: %s\n" line col msg;
      raise (ParseError (msg, line, col))
  in
  try parse_all tokens []
  with ParseError (msg, line, col) ->
    raise (ParseError (msg, line, col))

(* --- Type inference integration --- *)
let string_of_typ t =
  let rec aux = function
    | TInt -> "Int"
    | TLong -> "Long"
    | TFloat -> "Float"
    | TBool -> "Bool"
    | TString -> "String"
    | TVar v -> "'" ^ v
    | TFun (a, b) -> "(" ^ aux a ^ " -> " ^ aux b ^ ")"
    | TList a -> "[" ^ aux a ^ "]"
    | TDict a -> "{String: " ^ aux a ^ "}"
    | TArray a -> "Array[" ^ aux a ^ "]"
    | TTensor -> "Tensor"
    | TUnknown -> "?"
  in aux t

let parse_and_infer ?(show_types=true) input =
  Infer.reset_counter ();
  let exprs = parse input in
  let env = Infer.add_operators_to_env Types.StringMap.empty in
  let rec infer_toplevel env = function
    | [] -> ()
    | expr :: rest ->
        try
          let _, typ = Infer.infer env expr in
          if show_types then
            Printf.printf "Expr: %s\nType: %s\n\n" (Ast.string_of_expr expr) (string_of_typ typ);
          let env' =
            match expr with
            | Let (name, _) -> Types.StringMap.add name (Types.Forall ([], typ)) env
            | FunDef (name, _, _) -> Types.StringMap.add name (Types.Forall ([], typ)) env
            | _ -> env
          in
          infer_toplevel env' rest
        with
        | Infer.TypeError msg -> Printf.printf "Type error: %s\n" msg; infer_toplevel env rest
  in
  infer_toplevel env exprs;
  exprs
