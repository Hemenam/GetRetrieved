import test from "node:test";
import assert from "node:assert/strict";
import { ChatSession, ChatError, readToken, validateQuestion } from "../src/hrlearnium/web/assets/app.mjs";

test("local server course needs no JWT and sends no authorization header", async () => {
  const calls = [];
  const session = new ChatSession(async (url, options) => { calls.push({ url, options }); return response(); });
  session.configureLocal("captain-storm");
  assert.equal(session.local, true);
  assert.equal(session.connected, true);
  await session.ask("First question?", "explained", false, "full_context");
  await session.ask("Follow-up?");
  assert.equal(calls[0].options.headers.Authorization, undefined);
  assert.equal(calls[0].options.headers["Content-Type"], "application/json");
  assert.equal(JSON.parse(calls[1].options.body).conversation_id, "conversation-one");
  session.disconnect();
  assert.equal(session.connected, false);
  assert.throws(() => session.configureLocal("../other"), ChatError);
  assert.throws(() => session.configureLocal(null), ChatError);
});

test("local access errors do not tell the operator to generate tokens", async () => {
  const session = new ChatSession(async () => new Response("denied", { status: 403 }));
  session.configureLocal("course");
  await assert.rejects(session.ask("Question?"), /localhost/);
});

function token(extra = {}) {
  const claims = { iss: "test", aud: "course-api", sub: "learner", tenant_id: "tenant", exp: Math.floor(Date.now() / 1000) + 300, jti: "one", ...extra };
  return `test.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.test`;
}
const result = { status: "answered", answer: "متن", conversation_id: "conversation-one", request_id: "request-one", retrieval_mode: "full_context", response_mode: "explained", excerpts: [{ id: "source-one", text: "<script>not executable</script>", citation: { document_title: "Synthetic source" } }], explanation: { statements: [{ text: "پاسخ", citation_ids: ["source-one"] }] } };
const response = (body = result) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });

const metadataEvidence = { field: "chapter_count", language: "en", chapter_number: null, outline: { course_id: "course", snapshot: "snapshot-one", provenance: "imported_document_headings", documents: [{ document_id: "document-one", version: 1, chapters: [{ number: 1, title: "One", sections: ["explanation"] }] }] } };
const metadataResult = () => ({ ...result, answer_kind: "metadata", metadata: structuredClone(metadataEvidence), answer: "One recognized chapter.", excerpts: [], explanation: null });

test("metadata answers require provenance and are not source-text fallbacks", async () => {
  const session = new ChatSession(async () => response(metadataResult()));
  session.configure("course", token());
  assert.equal((await session.ask("How many chapters?")).answer_kind, "metadata");
  const invalid = new ChatSession(async () => response({ ...metadataResult(), metadata: null }));
  invalid.configure("course", token());
  await assert.rejects(invalid.ask("How many chapters?"), /provenance/);
});

function partialResult() {
  return { ...result, status: "partial", answer_kind: "multi", explanation: null, metadata: null, parts: [
    { question: "Question one?", status: "answered", answer_kind: "content", answer: result.answer, excerpt_ids: ["source-one"], explanation: result.explanation, metadata: null },
    { question: "Unknown question?", status: "refused", answer_kind: "content", answer: "Not explained in this course.", excerpt_ids: [], explanation: null, metadata: null }
  ] };
}

test("partial answers retain independent statuses and shared conversation context", async () => {
  const session = new ChatSession(async () => response(partialResult()));
  session.configure("course", token());
  const answer = await session.ask("Two questions?");
  assert.equal(answer.status, "partial");
  assert.deepEqual(answer.parts.map(p => p.status), ["answered", "refused"]);
  assert.equal(session.conversationId, "conversation-one");
});

test("a failed answer draft stays an error while a supported sibling is retained", async () => {
  const body = partialResult();
  body.parts[1].status = "error";
  body.parts[1].reason_code = "answer_verification_failed";
  body.parts[1].answer = "Could not verify this answer. Please retry.";
  const session = new ChatSession(async () => response(body));
  session.configure("course", token());
  const answer = await session.ask("Two questions?");
  assert.deepEqual(answer.parts.map(p => p.status), ["answered", "error"]);
  assert.equal(answer.status, "partial");
  body.parts[0] = { ...body.parts[1], question: "First failed question?" };
  body.excerpts = [];
  body.status = "error";
  const failed = new ChatSession(async () => response(body));
  failed.configure("course", token());
  assert.equal((await failed.ask("Two questions?")).status, "error");
});

test("mixed metadata/content answers are accepted without fake paragraph citations", async () => {
  const body = partialResult();
  body.status = "answered";
  body.parts[1] = { question: "How many chapters?", status: "answered", answer_kind: "metadata", answer: "One recognized chapter.", excerpt_ids: [], explanation: null, metadata: metadataEvidence };
  const session = new ChatSession(async () => response(body));
  session.configure("course", token());
  assert.equal((await session.ask("Two questions?")).parts[1].metadata.field, "chapter_count");
});

test("invalid partial statuses and cross-part evidence are rejected", async () => {
  for (const mutate of [b => { b.status = "answered"; }, b => { b.parts[1].excerpt_ids = ["source-one"]; }, b => { b.parts[0].explanation.statements[0].citation_ids = ["foreign-source"]; }]) {
    const body = structuredClone(partialResult()); mutate(body);
    const session = new ChatSession(async () => response(body)); session.configure("course", token());
    await assert.rejects(session.ask("Two questions?"), ChatError);
    assert.equal(session.conversationId, null);
  }
});

test("question validation counts Unicode characters, trims and enforces bounds", () => {
  assert.equal(validateQuestion("  پرسش؟  "), "پرسش؟");
  assert.equal(validateQuestion("😀".repeat(2000)).length, 4000);
  assert.throws(() => validateQuestion("a"), ChatError);
  assert.throws(() => validateQuestion("😀".repeat(2001)), ChatError);
});

test("token input accepts optional Bearer but rejects provider keys and expired tokens", () => {
  const jwt = token({ sub: "کاربر" });
  assert.equal(readToken(`Bearer ${jwt}`).token, jwt);
  assert.throws(() => readToken("sk-provider-key"), /service JWT/);
  assert.throws(() => readToken(token({ exp: 1 })), /expired/);
  assert.throws(() => readToken("aaa.bbb.ccc"), /readable/);
});

test("queries use same-origin authenticated API, modes and evaluation, then reuse conversation", async () => {
  const calls = [];
  const session = new ChatSession(async (url, options) => { calls.push({ url, options }); return response(); });
  const jwt = token();
  session.configure("captain-storm", jwt);
  await session.ask("پرسش اول", "explained", true);
  await session.ask("پرسش دوم", "verbatim");
  assert.equal(calls[0].url, "/v1/courses/captain-storm/query");
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${jwt}`);
  assert.equal(calls[0].options.redirect, "error");
  assert.equal(calls[0].options.credentials, "omit");
  assert.deepEqual(JSON.parse(calls[0].options.body), { question: "پرسش اول", response_mode: "explained", include_evaluation: true });
  assert.equal(JSON.parse(calls[1].options.body).conversation_id, "conversation-one");
  assert.equal(JSON.parse(calls[1].options.body).response_mode, "verbatim");
  assert.equal(session.pending, false);
  assert.equal(JSON.stringify(session).includes(jwt), false, "private token is not serializable");
});

test("fetch is not invoked with the session as its receiver", async () => {
  const session = new ChatSession(async function () { assert.equal(this, undefined); return response(); });
  session.configure("course", token());
  assert.equal((await session.ask("question")).status, "answered");
});

test("refreshing token keeps context; a different tenant, user or course resets it", async () => {
  const session = new ChatSession(async () => response());
  session.configure("course", token());
  await session.ask("question");
  assert.equal(session.configure("course", token({ jti: "two" })), false);
  assert.equal(session.conversationId, "conversation-one");
  assert.equal(session.configure("course", token({ sub: "other" })), true);
  assert.equal(session.conversationId, null);
  await session.ask("question");
  assert.equal(session.configure("course", token({ sub: "other", tenant_id: "other" })), true);
  await session.ask("question");
  assert.equal(session.configure("second-course", token({ sub: "other", tenant_id: "other" })), true);
  assert.equal(session.conversationId, null);
});

test("new chat and disconnect clear context without changing server data", async () => {
  const calls = [];
  const session = new ChatSession(async (url, options) => { calls.push(JSON.parse(options.body)); return response(); });
  session.configure("course", token());
  await session.ask("question");
  session.newConversation();
  await session.ask("new question");
  assert.equal("conversation_id" in calls[1], false);
  session.disconnect();
  assert.equal(session.connected, false);
  await assert.rejects(session.ask("question"), /Connect a course/);
});

for (const [status, pattern, action] of [[401, /token/, "connection"], [403, /permit/, "connection"], [404, /expired/, "new-chat"], [429, /Wait 10 seconds/, null], [503, /not a course refusal/, null], [409, /course changed/, null], [422, /could not accept/, null], [500, /request failed/, null]]) {
  test(`HTTP ${status} stays an operational error, with no automatic retry`, async () => {
    let calls = 0;
    const session = new ChatSession(async () => { calls++; return new Response("private provider error", { status, headers: { "X-Request-ID": "request-test", "Retry-After": "10" } }); });
    session.configure("course", token());
    await assert.rejects(session.ask("question"), error => error instanceof ChatError && pattern.test(error.message) && error.action === action && error.requestId === "request-test" && !error.message.includes("private provider"));
    assert.equal(calls, 1);
    assert.equal(session.conversationId, null);
    assert.equal(session.pending, false);
  });
}

test("refusal, clarification and conversational replies retain their conversation ID", async () => {
  for (const status of ["refused", "clarification", "conversation"]) {
    const session = new ChatSession(async () => response({ ...result, status, explanation: null, excerpts: [] }));
    session.configure("course", token());
    assert.equal((await session.ask("question")).status, status);
    assert.equal(session.conversationId, "conversation-one");
  }
});

test("social responses cannot contain course evidence", async () => {
  const session = new ChatSession(async () => response({ ...result, status: "conversation" }));
  session.configure("course", token());
  await assert.rejects(session.ask("سلام"), /cannot contain course evidence/);
  assert.equal(session.conversationId, null);
});

test("missing generated explanation is an error, not a source-text fallback", async () => {
  const session = new ChatSession(async () => response({ ...result, explanation: null }));
  session.configure("course", token());
  await assert.rejects(session.ask("question"), /Source text will not be substituted/);
  assert.equal(session.conversationId, null);
});

test("invalid JSON or citations do not update conversation", async () => {
  for (const fake of [() => new Response("not JSON"), () => response({ ...result, explanation: { statements: [{ text: "invented", citation_ids: ["missing"] }] } })]) {
    const session = new ChatSession(async () => fake());
    session.configure("course", token());
    await assert.rejects(session.ask("question"), ChatError);
    assert.equal(session.conversationId, null);
  }
});

test("cancelled or stale responses cannot overwrite a new conversation", async () => {
  let resolve;
  const session = new ChatSession(() => new Promise(done => { resolve = done; }));
  session.configure("course", token());
  const waiting = session.ask("question");
  await assert.rejects(session.ask("another"), /already/);
  session.newConversation();
  resolve(response());
  await assert.rejects(waiting, /Stopped waiting/);
  assert.equal(session.conversationId, null);
  assert.equal(session.pending, false);
});

test("timeout clears busy state and explains that the server may still be working", async () => {
  const session = new ChatSession((url, options) => new Promise((resolve, reject) => options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")))), 5);
  session.configure("course", token());
  await assert.rejects(session.ask("question"), /timed out/);
  assert.equal(session.pending, false);
});

const retrievalInfo = {
  default_mode: "full_context", check: "prerequisites_only", provider_verified: false,
  passage_count: 44, embedded_passage_count: 44,
  modes: ["full_context", "hybrid", "hybrid_rerank"].map(mode => ({ mode, available: true, reasons: [], warnings: [] })),
};

test("retrieval setup uses the authenticated course endpoint without creating conversation state", async () => {
  const calls = [];
  const session = new ChatSession(async (url, options) => { calls.push({ url, options }); return response(retrievalInfo); });
  const jwt = token();
  session.configure("captain-storm", jwt);
  assert.deepEqual(await session.retrievalOptions(), retrievalInfo);
  assert.equal(calls[0].url, "/v1/courses/captain-storm/retrieval-options");
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${jwt}`);
  assert.equal(calls[0].options.credentials, "omit");
  assert.equal(calls[0].options.body, undefined);
  assert.equal(session.conversationId, null);
});

for (const mode of ["full_context", "hybrid", "hybrid_rerank"]) {
  test(`explicit ${mode} reaches backend with response mode and evaluation independent`, async () => {
    let payload;
    const session = new ChatSession(async (url, options) => {
      payload = JSON.parse(options.body);
      return response({ ...result, retrieval_mode: mode });
    });
    session.configure("course", token());
    await session.ask("question", "explained", true, mode);
    assert.equal(payload.retrieval_mode, mode);
    assert.equal(payload.response_mode, "explained");
    assert.equal(payload.include_evaluation, true);
  });
}

test("a different returned mode is rejected instead of silently accepting fallback", async () => {
  const session = new ChatSession(async () => response());
  session.configure("course", token());
  await assert.rejects(session.ask("question", "explained", false, "hybrid"), /No fallback/);
  assert.equal(session.conversationId, null);
});

test("invalid mode makes no network request", async () => {
  const session = new ChatSession(() => { assert.fail("Must not fetch"); });
  session.configure("course", token());
  await assert.rejects(session.ask("question", "explained", false, "keyword"), /valid retrieval mode/);
});

test("stale retrieval checks cannot be accepted after changing course or disconnecting", async () => {
  for (const change of [session => session.configure("other-course", token()), session => session.disconnect()]) {
    let resolve;
    const session = new ChatSession(() => new Promise(done => { resolve = done; }));
    session.configure("course", token());
    const pending = session.retrievalOptions();
    change(session);
    resolve(response(retrievalInfo));
    await assert.rejects(pending, /connection changed/);
  }
});

test("starting a new chat does not invalidate course prerequisite checks", async () => {
  let resolve;
  const session = new ChatSession(() => new Promise(done => { resolve = done; }));
  session.configure("course", token());
  const pending = session.retrievalOptions();
  session.newConversation();
  resolve(response(retrievalInfo));
  assert.deepEqual(await pending, retrievalInfo);
});

test("malformed setup or missing updated endpoint cannot enable modes", async () => {
  for (const fake of [
    () => response({ ...retrievalInfo, modes: [] }),
    () => response({ ...retrievalInfo, modes: [retrievalInfo.modes[0], retrievalInfo.modes[0], retrievalInfo.modes[2]] }),
    () => response({ ...retrievalInfo, provider_verified: true }),
    () => new Response("not JSON"),
    () => new Response("missing", { status: 404 }),
  ]) {
    const session = new ChatSession(fake);
    session.configure("course", token());
    await assert.rejects(session.retrievalOptions(), ChatError);
  }
});
