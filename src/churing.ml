let read_file filename =
  let ic = open_in filename in
  let len = in_channel_length ic in
  let s = really_input_string ic len in
  close_in ic;
  s

let () =
  let args = Array.to_list Sys.argv |> List.tl in
  match args with
  | ("--version" | "-v") :: _ ->
      print_endline Version.version;
      exit 0
  | "compile" :: rest ->
      let output = ref "a.out" in
      let filename = ref "" in
      let rec parse_compile_args = function
        | "-o" :: out :: rest -> output := out; parse_compile_args rest
        | f :: rest when !filename = "" && not (String.starts_with ~prefix:"-" f) ->
            filename := f; parse_compile_args rest
        | _ :: rest -> parse_compile_args rest
        | [] -> ()
      in
      parse_compile_args rest;
      if !filename = "" then (
        Printf.eprintf "Usage: churing compile [-o output] <file.ch>\n";
        exit 1
      );
      let input = read_file !filename in
      Eval.source_dir := Filename.dirname !filename;
      (try
        let exprs = Parser.parse_and_infer ~show_types:false input in
        Codegen.compile_to_binary ~output:!output exprs
      with
        | Failure msg -> Printf.eprintf "Compile error: %s\n" msg; exit 1
        | Parser.ParseError (msg, line, col) ->
            Printf.eprintf "Parse error at line %d, col %d: %s\n" line col msg; exit 1)
  | _ ->
      let show_ast   = ref false in
      let show_types = ref false in
      let filename   = ref "" in
      List.iter (function
        | "--ast"   -> show_ast   := true
        | "--types" -> show_types := true
        | s when !filename = "" && not (String.starts_with ~prefix:"--" s) -> filename := s
        | _ -> ()
      ) args;
      if !filename = "" then (
        Printf.printf "churing %s\nUsage: churing [--ast] [--types] <file.ch>\n       churing compile [-o output] <file.ch>\n" Version.version;
        exit 1
      );
      let input = read_file !filename in
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