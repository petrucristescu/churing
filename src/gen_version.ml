let () =
  let ic = open_in Sys.argv.(1) in
  let version = String.trim (input_line ic) in
  close_in ic;
  Printf.printf "let version = %S\n" version
