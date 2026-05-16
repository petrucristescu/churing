// C-ABI string functions for the native compile pipeline.
// All heap allocations use GC_malloc so the Boehm GC can track them.
use std::ffi::CStr;

extern "C" {
    fn GC_malloc(size: usize) -> *mut u8;
}

unsafe fn alloc_str(s: &str) -> *const i8 {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let ptr = GC_malloc(len + 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
    *ptr.add(len) = 0;
    ptr as *const i8
}

unsafe fn cstr<'a>(s: *const i8) -> &'a str {
    CStr::from_ptr(s).to_str().unwrap_or("")
}

#[no_mangle]
pub unsafe extern "C" fn churing_str_length(s: *const i8) -> f64 {
    cstr(s).len() as f64
}

#[no_mangle]
pub unsafe extern "C" fn churing_concat(a: *const i8, b: *const i8) -> *const i8 {
    alloc_str(&format!("{}{}", cstr(a), cstr(b)))
}

#[no_mangle]
pub unsafe extern "C" fn churing_substring(s: *const i8, start: f64, len: f64) -> *const i8 {
    let s = cstr(s).as_bytes();
    let start = (start as usize).min(s.len());
    let end = (start + len as usize).min(s.len());
    alloc_str(std::str::from_utf8_unchecked(&s[start..end]))
}

#[no_mangle]
pub unsafe extern "C" fn churing_uppercase(s: *const i8) -> *const i8 {
    alloc_str(&cstr(s).to_uppercase())
}

#[no_mangle]
pub unsafe extern "C" fn churing_lowercase(s: *const i8) -> *const i8 {
    alloc_str(&cstr(s).to_lowercase())
}

#[no_mangle]
pub unsafe extern "C" fn churing_trim(s: *const i8) -> *const i8 {
    alloc_str(cstr(s).trim())
}

#[no_mangle]
pub unsafe extern "C" fn churing_contains(s: *const i8, sub: *const i8) -> f64 {
    if cstr(s).contains(cstr(sub)) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub unsafe extern "C" fn churing_starts_with(s: *const i8, pre: *const i8) -> f64 {
    if cstr(s).starts_with(cstr(pre)) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub unsafe extern "C" fn churing_ends_with(s: *const i8, suf: *const i8) -> f64 {
    if cstr(s).ends_with(cstr(suf)) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub unsafe extern "C" fn churing_replace(s: *const i8, from: *const i8, to: *const i8) -> *const i8 {
    alloc_str(&cstr(s).replace(cstr(from), cstr(to)))
}

#[no_mangle]
pub unsafe extern "C" fn churing_to_string(n: f64) -> *const i8 {
    let s = if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    };
    alloc_str(&s)
}

#[no_mangle]
pub unsafe extern "C" fn churing_to_float(s: *const i8) -> f64 {
    cstr(s).parse::<f64>().unwrap_or(0.0)
}

#[no_mangle]
pub unsafe extern "C" fn churing_index_of(s: *const i8, sub: *const i8) -> f64 {
    match cstr(s).find(cstr(sub)) {
        Some(idx) => idx as f64,
        None => -1.0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn churing_char_at(s: *const i8, idx: f64) -> *const i8 {
    let bytes = cstr(s).as_bytes();
    let i = idx as usize;
    if i < bytes.len() {
        alloc_str(std::str::from_utf8_unchecked(&bytes[i..i + 1]))
    } else {
        alloc_str("")
    }
}
