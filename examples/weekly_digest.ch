# weekly_digest.ch — Weekly Sales Intelligence Report
#
# Connects to MySQL for order and customer data, computes statistics,
# segments customers, generates an LLM digest, and saves a JSON report.
#
# Showcases: @-bindings, named functions, lambdas, pattern matching,
# cons/nil patterns, list/dict literals, recursion, TCO, llvm intrinsics,
# ask, result monad, IO monad, try/catch, assert, MySQL, JSON, env vars.

# ── Config ────────────────────────────────────────────────────────────

@configRaw  (readFile (envOr "DIGEST_CONFIG" "config.json"))
@config     (fromJson configRaw)
@dbUrl      (get config "db_url")
@topN       (toInt (get config "top_products"))
@minOrders  (toFloat (get config "min_orders"))

# ── Database ──────────────────────────────────────────────────────────

@db (mysqlConnect dbUrl)

@orders (mysqlFind db
    "SELECT customer_id, amount, product, created_at FROM orders WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)")

@customers (mysqlFind db
    "SELECT id, name, email, segment FROM customers")

# ── Tail-recursive stats (TCO) ────────────────────────────────────────

# Accumulate sum and count in one pass — classic TCO pattern
~statsAcc xs s n (
    match xs
    | []      -> [s, n]
    | h :: t  -> statsAcc t (s + h) (n + 1.0)
)

~mean xs (
    @r (statsAcc xs 0.0 0.0)
    match (eq (nth 1 r) 0.0)
    | true  -> 0.0
    | false -> nth 0 r / nth 1 r
)

# Population std deviation — uses llvm sqrt intrinsic via math.ch
~stddev xs (
    @avg (mean xs)
    @sq  (map (|>x. (x - avg) * (x - avg)) xs)
    sqrt (mean sq)
)

# ── Order analysis ────────────────────────────────────────────────────

~amounts orders (map (|>o. toFloat (get o "amount")) orders)

~productCounts orders (
    foldl
        (|>acc. |>o.
            @p   (get o "product")
            @cur (if (has acc p) (get acc p) 0.0)
            set acc p (cur + 1.0))
        {}
        orders
)

# Sort products by count descending — pick top N via fold over keys
~topProducts counts n (
    @ranked (foldl
        (|>acc. |>k. cons [k, get counts k] acc)
        []
        (keys counts))
    @sorted (reverse (foldl
        (|>acc. |>item.
            match acc
            | [] -> [item]
            | h :: t ->
                match (gt (nth 1 item) (nth 1 h))
                | true  -> cons item acc
                | false -> cons h (foldl (|>a. |>x. cons x a) [item] t))
        []
        ranked))
    map (|>item. nth 0 item) (take n sorted)
)

# ── Customer segmentation ─────────────────────────────────────────────

~orderCount customerId orders (
    len (filter (|>o. eq (get o "customer_id") customerId) orders)
)

~segment c orders (
    @n (toFloat (orderCount (get c "id") orders))
    match (gt n minOrders)
    | true  ->
        match (gt n (minOrders * 3.0))
        | true  -> "vip"
        | false -> "active"
    | false ->
        match (eq n 0.0)
        | true  -> "dormant"
        | false -> "occasional"
)

~segmentCustomers customers orders (
    map (|>c. merge c {computed_segment: segment c orders}) customers
)

# ── Revenue breakdown by segment ──────────────────────────────────────

~revenueBySegment segmented orders (
    foldl
        (|>acc. |>c.
            @seg (get c "computed_segment")
            @cid (get c "id")
            @rev (sum (amounts (filter (|>o. eq (get o "customer_id") cid) orders)))
            @cur (if (has acc seg) (get acc seg) 0.0)
            set acc seg (cur + rev))
        {}
        segmented
)

# ── Safe LLM insight via result monad ────────────────────────────────

~buildPrompt stats top (
    concat
        "Weekly sales stats (JSON): "
        (concat
            (toJson stats)
            (concat
                "\nTop products: "
                (concat
                    (join ", " top)
                    "\nWrite 3 sentences of business insight for a product manager. Be specific.")))
)

~safeAsk prompt (
    try
        (ok (ask prompt))
        (|>e. err (concat "LLM unavailable: " e))
)

# ── IO pipeline — build and save report ──────────────────────────────

~saveReport path report (
    runIO (writeFileIO path (toJson report))
)

# ── Main ──────────────────────────────────────────────────────────────

~run db orders customers (
    assert (not (empty orders))
    assert (not (empty customers))

    @amts      (amounts orders)
    @segmented (segmentCustomers customers orders)
    @counts    (productCounts orders)
    @top       (topProducts counts topN)
    @revBySeg  (revenueBySegment segmented orders)

    @stats {
        period:           "last_7_days",
        total_orders:     len orders,
        total_revenue:    sum amts,
        avg_order_value:  mean amts,
        revenue_stddev:   stddev amts,
        customers_active: len (filter (|>c. not (eq (get c "computed_segment") "dormant")) segmented),
        revenue_by_segment: revBySeg
    }

    @insightResult (safeAsk (buildPrompt stats top))
    @insight (unwrapOr "No insight available." insightResult)

    @report (merge stats {
        top_products: top,
        insight:      insight,
        generated_at: concat (toString (year (now))) "-" (concat (toString (month (now))) (concat "-" (toString (day (now)))))
    })

    saveReport (envOr "DIGEST_OUTPUT" "digest.json") report
    report
)

# ── Entry point ───────────────────────────────────────────────────────

@result (try
    (run db orders customers)
    (|>err.
        merge {} {error: err}))

mysqlClose db

assert (not (has result "error"))
assert (gt (get result "total_orders") 0.0)

print (concat "Orders this week:  " (toString (get result "total_orders")))
print (concat "Total revenue:     " (toString (get result "total_revenue")))
print (concat "Avg order value:   " (toString (get result "avg_order_value")))
print (concat "Top products:      " (join ", " (get result "top_products")))
print (concat "Insight:\n" (get result "insight"))
