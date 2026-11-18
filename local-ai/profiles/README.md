# profiles/

One `<productId>.json` per product. The middleware loads it by the productId that the
API key resolves to. Editing a profile is a reviewed PR that must pass that product's
DeepEval regression gate before merge (`eval/`).

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `productId` | string | Must equal the filename. |
| `has_rag` | boolean | If `true`, the middleware calls the external RAG service. Requires `rag`. |
| `systemPrompt` | string \| null | Prepended as a `system` message. `null` = product sends its own complete prompt. |
| `schema` | JSON Schema \| null | If set, the model output is validated against it (1 retry, then annotate). |
| `fewShotExamples` | `[{role,content}]` | Injected right before the first user message. |
| `rag` | `{project, topK}` \| null | Required iff `has_rag`. `project` = the product's RAG project/collection name (server-resolved, never from the request). |
| `defaults` | `{model, temperature}` | Fallbacks when the request omits them. |

## Current profiles

- **`flowgenix.json`** — MVP pilot. Everything off: Flowgenix builds its own full
  extraction prompt, does its own Ajv validation, and uses no retrieval
  (`PROJECT_LOG.md` §5). The profile exists only so identity + logging + the
  cost-avoided metric work.
