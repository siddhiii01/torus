# middleware/ — the custom piece

Thin service in front of LiteLLM. Node 20, ESM, Express. Listens on `:4100`.

**Provisional language choice** (Open Q E-16): Node.js, because the PDFs assume it,
the Flowgenix backend is Node/Express, and Promptfoo is Node-based. The service is
small and this is easily reversible.

## Run

```bash
npm install
cp .env.example .env                     # then edit
cp .secrets/keys.example.json .secrets/keys.json
node src/tools/mint-key.js flowgenix dev  # paste the sha256 into keys.json, keep the raw key
npm start
```

## Request lifecycle (`src/index.js`)

1. `identity.js` — `Authorization: Bearer` → `productId` (hashed lookup, never the body).
2. `profileLoader.js` — load `profiles/<productId>.json`.
3. `ragClient.js` — **only if** `profile.has_rag`. Calls the external RAG service.
   *Provisional — do not route a real product until the RAG report lands.*
4. `upstream.js` — forward the assembled request to LiteLLM; enforce the timeout.
5. `validate.js` — **only if** `profile.schema`. One retry, then annotate (no hard 5xx).
6. `logger.js` — one content-free JSON line per request.

For the **Flowgenix MVP**: no system prompt, no few-shot, no RAG, no schema — so the
path is identity → forward → log.

## Files never committed

`.env`, `.secrets/keys.json`, `logs/`. See root `.gitignore`.
