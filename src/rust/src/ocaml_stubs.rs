// Raw OCaml C ABI stubs — replaces codegen_stubs.c
// Receives boxed OCaml values (nativeint, float) and returns a boxed float.
//
// OCaml value layout used here:
//   Double_val(v)    = *(double *)(v)          -- float block body starts at v
//   Nativeint_val(v) = *((intnat *)(v + 1 word)) -- custom block: [ops_ptr][data]

type Value = usize;

unsafe fn extract_double(v: Value) -> f64 {
    *(v as *const f64)
}

unsafe fn extract_nativeint(v: Value) -> usize {
    *((v as *const usize).add(1))
}

extern "C" {
    fn caml_copy_double(x: f64) -> Value;
}

#[no_mangle]
pub unsafe extern "C" fn caml_call_f1(addr: Value, x: Value) -> Value {
    let f: fn(f64) -> f64 = std::mem::transmute(extract_nativeint(addr));
    caml_copy_double(f(extract_double(x)))
}

#[no_mangle]
pub unsafe extern "C" fn caml_call_f2(addr: Value, x: Value, y: Value) -> Value {
    let f: fn(f64, f64) -> f64 = std::mem::transmute(extract_nativeint(addr));
    caml_copy_double(f(extract_double(x), extract_double(y)))
}
