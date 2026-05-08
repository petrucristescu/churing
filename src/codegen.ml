(* Churing → LLVM IR compiler.
   Issue #89: smoke test — verifies LLVM OCaml bindings are linked and functional.
   Issues #91-#95: full compiler will be built here. *)

let smoke_test () =
  let ctx = Llvm.create_context () in
  let md  = Llvm.create_module ctx "churing" in
  let f64 = Llvm.double_type ctx in
  let ft  = Llvm.function_type f64 [| f64 |] in
  let fn  = Llvm.define_function "churing_sqrt" ft md in
  let bb  = Llvm.entry_block fn in
  let b   = Llvm.builder_at_end ctx bb in
  let x   = (Llvm.params fn).(0) in
  let sq  = Llvm.declare_function "llvm.sqrt.f64" ft md in
  let r   = Llvm.build_call ft sq [| x |] "r" b in
  ignore (Llvm.build_ret r b);
  let ir  = Llvm.string_of_llmodule md in
  Llvm.dispose_module md;
  Llvm.dispose_context ctx;
  ir
