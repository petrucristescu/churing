let () =
  let show_ast   = ref false in
  let show_types = ref false in
  let filename   = ref "" in
  let args = Array.to_list Sys.argv |> List.tl in
  List.iter (function
    | "--version" | "-v" ->
        print_endline Version.version;
        exit 0
    | "--ast"   -> show_ast   := true
    | "--types" -> show_types := true
    | s when !filename = "" && not (String.starts_with ~prefix:"--" s) -> filename := s
    | _ -> ()
  ) args;
  if !filename = "" then (
    Printf.printf "churing %s\nUsage: churing [--ast] [--types] <file.ch>\n" Version.version;
    exit 1
  );
  let input =
    let ic = open_in !filename in
    let len = in_channel_length ic in
    let s = really_input_string ic len in
    close_in ic;
    s
  in
  Eval.source_dir := Filename.dirname !filename;
  try
    let exprs = Parser.parse_and_infer ~show_types:!show_types input in
    if !show_ast then Ast.print_ast exprs;
    Eval.eval_program exprs
  with
    | Eval.AssertionFailure msg ->
      Printf.eprintf "%s\n" msg;
      exit 1
    | Eval.RuntimeError msg ->
      Printf.eprintf "Churing Error: %s\n" msg;
      exit 1
    | Parser.ParseError (msg, line, col) ->
      Printf.eprintf "Parse error at line %d, col %d: %s\n" line col msg