# Lookup tables (public data only)

Never put customer or document data here.

- `india_states_uts.txt`: closed list of states and union territories.
- `india_state_aliases.csv`: former/alternative names (`alias,official`).
- `india_post_pincodes.csv`: **India Post PIN master. Not committed until a human
  supplies it** (public dataset from India Post / data.gov.in). UTF-8 CSV with a
  header: `pincode,office,district,state`. One row per post office; `state` must
  be in the closed list or an alias. While the file is missing, the PIN lookup
  validators fail closed (`lookup_unavailable`), so no PIN can be `confirmed`.
