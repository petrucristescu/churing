type expr =
  | Int of int
  | Lng of int64
  | Float of float
  | Str of string
  | Bool of bool
  | Add of expr * expr
  | Sub of expr * expr
  | Mul of expr * expr
  | Div of expr * expr
  | Eq of expr * expr
  | Var of string
  | Lam of string * expr
  | App of expr * expr
  | Let of string * expr
  | FunDef of string * string list * expr
  | Seq of expr * expr
  | Assert of expr
  | List of expr list
  | Dict of (string * expr) list
  | Match of expr * (pattern * expr) list
  | Import of string
  | Llvm of string * expr list

and pattern =
  | PInt of int
  | PStr of string
  | PBool of bool
  | PVar of string
  | PWild
  | PList of pattern list
  | PCons of pattern * pattern  (* head :: tail destructuring *)

let rec string_of_typ = function
  | Types.TInt -> "Int"
  | Types.TBool -> "Bool"
  | Types.TString -> "String"
  | Types.TLong -> "Long"
  | Types.TFloat -> "Float"
  | Types.TUnknown -> "?"
  | Types.TVar v -> "'" ^ v
  | Types.TFun (a, b) -> "(" ^ string_of_typ a ^ " -> " ^ string_of_typ b ^ ")"
  | Types.TList a -> "[" ^ string_of_typ a ^ "]"
  | Types.TDict a -> "{String: " ^ string_of_typ a ^ "}"
  | Types.TArray a -> "Array[" ^ string_of_typ a ^ "]"
  | Types.TTensor -> "Tensor"

let rec string_of_expr = function
  | Int n -> string_of_int n
  | Lng n -> Int64.to_string n
  | Float f -> string_of_float f
  | Bool b -> string_of_bool b
  | Eq (a, b) -> "(" ^ string_of_expr a ^ " == " ^ string_of_expr b ^ ")"
  | Add (a, b) -> "(" ^ string_of_expr a ^ " + " ^ string_of_expr b ^ ")"
  | Sub (a, b) -> "(" ^ string_of_expr a ^ " - " ^ string_of_expr b ^ ")"
  | Mul (a, b) -> "(" ^ string_of_expr a ^ " * " ^ string_of_expr b ^ ")"
  | Div (a, b) -> "(" ^ string_of_expr a ^ " / " ^ string_of_expr b ^ ")"
  | Var x -> x
  | Lam (x, body) -> "|>" ^ x ^ ". " ^ string_of_expr body
  | App (f, a) -> "(" ^ string_of_expr f ^ " " ^ string_of_expr a ^ ")"
  | Str s -> "\"" ^ s ^ "\""
  | Let (name, value) ->
      "@" ^ name ^ " " ^ string_of_expr value
  | FunDef (name, args, body) ->
      "~" ^ name ^ " " ^ String.concat "," args ^ " " ^ string_of_expr body
  | Seq (a, b) -> string_of_expr a ^ ";\n" ^ string_of_expr b
  | Assert e -> "assert " ^ string_of_expr e
  | List es -> "[" ^ String.concat ", " (List.map string_of_expr es) ^ "]"
  | Dict entries ->
      "{" ^ String.concat ", " (List.map (fun (k, v) -> k ^ ": " ^ string_of_expr v) entries) ^ "}"
  | Match (e, arms) ->
      "match " ^ string_of_expr e ^ " " ^
      String.concat " " (List.map (fun (p, body) ->
        "| " ^ string_of_pattern p ^ " -> " ^ string_of_expr body) arms)
  | Import lib -> "import " ^ lib
  | Llvm (intrinsic, args) ->
      "(llvm \"" ^ intrinsic ^ "\" " ^ String.concat " " (List.map string_of_expr args) ^ ")"

and string_of_pattern = function
  | PInt n -> string_of_int n
  | PStr s -> "\"" ^ s ^ "\""
  | PBool b -> string_of_bool b
  | PVar x -> x
  | PWild -> "_"
  | PList ps -> "[" ^ String.concat ", " (List.map string_of_pattern ps) ^ "]"
  | PCons (h, t) -> string_of_pattern h ^ " :: " ^ string_of_pattern t

let string_of_exprs exprs =
  String.concat "\n" (List.map string_of_expr exprs)

let print_ast exprs =
  print_endline (string_of_exprs exprs)