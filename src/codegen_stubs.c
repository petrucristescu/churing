#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>

/* Call a native double -> double function at the given nativeint address. */
CAMLprim value caml_call_f1(value addr_val, value x_val) {
  typedef double (*fn_t)(double);
  fn_t fn = (fn_t)Nativeint_val(addr_val);
  return caml_copy_double(fn(Double_val(x_val)));
}

/* Call a native double -> double -> double function at the given nativeint address. */
CAMLprim value caml_call_f2(value addr_val, value x_val, value y_val) {
  typedef double (*fn_t)(double, double);
  fn_t fn = (fn_t)Nativeint_val(addr_val);
  return caml_copy_double(fn(Double_val(x_val), Double_val(y_val)));
}
