/**
 * Client for the EXTERNAL, already-deployed RAG service.
 * We call it — we never embed or rebuild it. (PROJECT_LOG.md §0.1, §3.2)
 *
 * Only invoked when a product's profile has has_rag: true. Flowgenix does NOT use
 * this (has_rag: false, evidence-backed — PROJECT_LOG.md §5.6).
 *
 * Contract below is PROVISIONAL — pending the RAG codebase report (PROJECT_LOG.md
 * §8.2). Do not wire a real product through this until that report lands and the
 * request/response shapes + auth + per-product project mapping are confirmed.
 */

import { request } from "undici";

const RAG_BASE_URL = process.env.RAG_BASE_URL?.trim() || null;

/**
 * @param {{project:string, topK?:number}} rag  from profile.rag
 * @param {string} query
 * @param {string} projectKey  per-product RAG bearer key (from .secrets, not the request)
 * @returns {Promise<{chunks: Array<{id:string,text:string,metadata?:object}>}>}
 */
export async function retrieve(rag, query, projectKey) {
  if (!RAG_BASE_URL) throw new Error("RAG_BASE_URL not configured but a profile requested RAG");
  // TODO(rag-report): confirm path, body shape, auth header, and response envelope.
  const res = await request(`${RAG_BASE_URL}/v1/retrieve`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${projectKey}`,
    },
    body: JSON.stringify({ collection: rag.project, question: query, topK: rag.topK ?? 5 }),
  });
  if (res.statusCode >= 300) {
    const text = await res.body.text();
    throw new Error(`RAG service ${res.statusCode}: ${text.slice(0, 300)}`);
  }
  return res.body.json();
}
