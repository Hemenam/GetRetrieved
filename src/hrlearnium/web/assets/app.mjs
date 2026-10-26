// No framework, provider key, persistent browser storage, or client-side model calls.
const retrievalLabels = { full_context: "Full context", hybrid: "Hybrid", hybrid_rerank: "Hybrid + reranker", lexical: "Diagnostic keyword search" };

export class ChatError extends Error {
  constructor(message, { status = 0, requestId = null, action = null } = {}) {
    super(message);
    this.name = "ChatError";
    Object.assign(this, { status, requestId, action });
  }
}

export function validateQuestion(value) {
  const question = value.trim();
  const length = [...question].length;
  if (length < 2 || length > 2000) throw new ChatError("Write a question of 2–2,000 characters.");
  return question;
}

export function readToken(value) {
  const token = value.trim().replace(/^Bearer\s+/i, "");
  if (token.length > 12000 || !/^[\w-]+\.[\w-]+\.[\w-]+$/.test(token)) {
    throw new ChatError("Paste the service JWT printed by the token command, not a model API key.");
  }
  try {
    const part = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const bytes = Uint8Array.from(atob(part.padEnd(Math.ceil(part.length / 4) * 4, "=")), c => c.charCodeAt(0));
    const claims = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    if (typeof claims.sub !== "string" || typeof claims.tenant_id !== "string" || !Number.isFinite(claims.exp)) throw new Error();
    if (claims.exp * 1000 <= Date.now()) throw new ChatError("This token has expired. Generate and paste a fresh one.");
    // Identity hints only. Signature, scopes, lifetime and course access are verified by Python.
    const audience = Array.isArray(claims.aud) ? [...claims.aud].sort() : claims.aud;
    return { token, expiresAt: claims.exp * 1000, identity: JSON.stringify([claims.iss, audience, claims.tenant_id, claims.sub]) };
  } catch (error) {
    if (error instanceof ChatError) throw error;
    throw new ChatError("This is not a readable service JWT. Generate a fresh token with the CLI.");
  }
}

export function errorForStatus(status, requestId, retryAfter) {
  const options = { status, requestId };
  if (status === 401) return new ChatError("Your service token is invalid or expired. Generate a fresh token and update the connection.", { ...options, action: "connection" });
  if (status === 403) return new ChatError("This token does not permit queries for this course. Check the course ID and token permissions.", { ...options, action: "connection" });
  if (status === 404) return new ChatError("The conversation was not found or has expired. Start a new chat and resend your question.", { ...options, action: "new-chat" });
  if (status === 429) {
    const seconds = /^\d{1,5}$/.test(retryAfter || "") ? ` Wait ${retryAfter} seconds before retrying.` : " Retry shortly.";
    return new ChatError(`The request limit has been reached.${seconds}`, options);
  }
  if (status === 503) return new ChatError("The model or evidence service is unavailable. Check server readiness and logs, then retry. This is not a course refusal.", options);
  if (status === 409) return new ChatError("The course changed while the answer was being prepared. Please resend your question.", options);
  if (status === 422) return new ChatError("The server could not accept this question or its options. Check the input and try again.", options);
  return new ChatError("The request failed. Check the Python server and try again.", options);
}

function validateResponse(body) {
  if (!body || !["answered", "partial", "refused", "clarification", "conversation", "error"].includes(body.status) || typeof body.answer !== "string" || typeof body.conversation_id !== "string" || typeof body.request_id !== "string" || !["lexical", "hybrid", "hybrid_rerank", "full_context"].includes(body.retrieval_mode) || !Array.isArray(body.excerpts)) {
    throw new ChatError("The server returned an unexpected response. Check that this page and API use the same version.");
  }
  for (const excerpt of body.excerpts) {
    if (!excerpt || typeof excerpt.id !== "string" || typeof excerpt.text !== "string" || !excerpt.citation) throw new ChatError("The server returned invalid source evidence.");
  }
  if (new Set(body.excerpts.map(e => e.id)).size !== body.excerpts.length) throw new ChatError("Duplicate evidence identifiers.");
  if (body.answer_kind === "multi") {
    if (!Array.isArray(body.parts) || body.parts.length < 2 || body.parts.length > 8 || body.explanation != null || body.metadata != null) throw new ChatError("Invalid multi-question response.");
    const ids = new Set(body.excerpts.map(e => e.id));
    const used = new Set();
    for (const part of body.parts) {
      if (!part || typeof part.question !== "string" || !["answered", "refused", "clarification", "error"].includes(part.status) || !["content", "metadata"].includes(part.answer_kind) || !Array.isArray(part.excerpt_ids) || new Set(part.excerpt_ids).size !== part.excerpt_ids.length || part.excerpt_ids.some(id => !ids.has(id))) throw new ChatError("Invalid question-part evidence.");
      part.excerpt_ids.forEach(id => used.add(id));
      validateResponse({ ...body, ...part, parts: [], excerpts: body.excerpts.filter(e => part.excerpt_ids.includes(e.id)) });
    }
    const answered = body.parts.filter(p => p.status === "answered").length;
    const expected = answered === body.parts.length ? "answered" : answered ? "partial" : body.parts.some(p => p.status === "error") ? "error" : body.parts.every(p => p.status === "clarification") ? "clarification" : "refused";
    if (body.status !== expected || used.size !== ids.size) throw new ChatError("Inconsistent question-part statuses or evidence.");
    return body;
  }
  if (body.status === "partial" || body.parts?.length) throw new ChatError("Partial answers require separate question results.");
  if (body.answer_kind === "metadata") {
    if (body.excerpts.length || body.explanation != null) throw new ChatError("Metadata must not impersonate passage evidence.");
    if (body.status === "answered") {
      const outline = body.metadata?.outline;
      if (typeof body.metadata?.field !== "string" || !outline || typeof outline.course_id !== "string" || typeof outline.snapshot !== "string" || outline.provenance !== "imported_document_headings" || !Array.isArray(outline.documents) || !outline.documents.length || outline.documents.some(d => typeof d.document_id !== "string" || !Number.isInteger(d.version) || !Array.isArray(d.chapters))) throw new ChatError("Missing course metadata provenance.");
    } else if (body.metadata != null) throw new ChatError("Unsupported metadata answer contains evidence.");
    return body;
  }
  if (body.status !== "answered" && (body.excerpts.length || body.explanation != null)) throw new ChatError("A conversational reply or refusal cannot contain course evidence.");
  if (body.status === "answered" && body.response_mode === "explained" && !body.explanation?.statements?.length) throw new ChatError("The server did not return a generated explanation. Source text will not be substituted for it.");
  if (body.explanation) {
    const ids = new Set(body.excerpts.map(excerpt => excerpt.id));
    if (!Array.isArray(body.explanation.statements) || body.explanation.statements.some(statement => typeof statement.text !== "string" || !Array.isArray(statement.citation_ids) || !statement.citation_ids.length || statement.citation_ids.some(id => !ids.has(id)))) throw new ChatError("The server returned an invalid citation mapping.");
  }
  return body;
}

export class ChatSession {
  #token = null;
  #local = false;
  #identity = null;
  #controller = null;
  #revision = 0;
  #connectionRevision = 0;

  constructor(fetchImpl = globalThis.fetch, timeoutMs = 600000) {
    this.fetchImpl = fetchImpl;
    this.timeoutMs = timeoutMs;
    this.course = "";
    this.conversationId = null;
    this.expiresAt = 0;
    this.pending = false;
  }

  get connected() { return Boolean(this.#token) || this.#local; }
  get local() { return this.#local; }

  configureLocal(course) {
    if (this.pending) throw new ChatError("Wait for the current request before changing the connection.");
    if (typeof course !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$/.test(course)) throw new ChatError("Invalid server course ID.");
    this.disconnect();
    this.#local = true;
    this.course = course;
    this.expiresAt = Infinity;
  }

  authHeaders() { return this.#local ? {} : { Authorization: `Bearer ${this.#token}` }; }

  requestError(status, requestId, retryAfter) {
    if (this.#local && [401, 403].includes(status)) return new ChatError("Local access was denied. Use the localhost URL and the course configured on the server.", { status, requestId });
    return errorForStatus(status, requestId, retryAfter);
  }

  configure(course, tokenValue) {
    if (this.pending) throw new ChatError("Wait for the current request, or stop waiting before changing the connection.");
    course = course.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$/.test(course)) throw new ChatError("Enter a valid course ID, using letters, numbers, dots, colons, underscores or hyphens.");
    const { token, identity, expiresAt } = readToken(tokenValue);
    this.#local = false;
    this.#connectionRevision += 1;
    const reset = identity !== this.#identity || course !== this.course;
    if (reset) this.newConversation();
    this.#token = token;
    this.#identity = identity;
    this.course = course;
    this.expiresAt = expiresAt;
    return reset;
  }

  cancel() {
    this.#revision += 1;
    this.#controller?.abort();
    this.#controller = null;
    this.pending = false;
  }

  newConversation() { this.cancel(); this.conversationId = null; }

  disconnect() {
    this.#connectionRevision += 1;
    this.newConversation();
    this.#token = null;
    this.#local = false;
    this.#identity = null;
    this.expiresAt = 0;
    this.course = "";
  }

  checkConnection() {
    if (!this.connected) throw new ChatError("Connect a course with a service token first.", { action: "connection" });
    if (Date.now() >= this.expiresAt) throw new ChatError("Your service token has expired. Update it to continue this conversation.", { action: "connection" });
  }

  async retrievalOptions() {
    this.checkConnection();
    const revision = this.#connectionRevision;
    const fetchRequest = this.fetchImpl;
    let response;
    try {
      response = await fetchRequest(`/v1/courses/${encodeURIComponent(this.course)}/retrieval-options`, {
        headers: this.authHeaders(), credentials: "omit", cache: "no-store", redirect: "error",
        signal: AbortSignal.timeout(15000),
      });
    } catch { throw new ChatError("Could not check retrieval setup. Check the server and refresh status."); }
    if (revision !== this.#connectionRevision) throw new ChatError("The connection changed before the retrieval check finished.");
    if (response.status === 404) throw new ChatError("Retrieval options are not available. Restart the updated Python server, then refresh status.");
    if (!response.ok) throw this.requestError(response.status, response.headers.get("X-Request-ID"), response.headers.get("Retry-After"));
    let data;
    try { data = await response.json(); } catch { throw new ChatError("The server returned unreadable retrieval options."); }
    if (revision !== this.#connectionRevision) throw new ChatError("The connection changed before the retrieval check finished.");
    const modes = ["full_context", "hybrid", "hybrid_rerank"];
    if (!data || !Object.hasOwn(retrievalLabels, data.default_mode) || data.check !== "prerequisites_only" || data.provider_verified !== false || !Number.isInteger(data.passage_count) || !Number.isInteger(data.embedded_passage_count) || !Array.isArray(data.modes) || data.modes.length !== 3 || new Set(data.modes.map(item => item?.mode)).size !== 3 || data.modes.some(item => !modes.includes(item?.mode) || typeof item.available !== "boolean" || !Array.isArray(item.reasons) || !Array.isArray(item.warnings) || [...item.reasons, ...item.warnings].some(text => typeof text !== "string"))) {
      throw new ChatError("The server returned invalid retrieval options. Restart matching versions of the page and API.");
    }
    return data;
  }

  async ask(value, mode = "explained", includeEvaluation = false, retrievalMode = null) {
    const question = validateQuestion(value);
    this.checkConnection();
    if (this.pending) throw new ChatError("A question is already being answered.");
    if (!["verbatim", "explained"].includes(mode)) throw new ChatError("Choose a valid response mode.");
    if (retrievalMode !== null && !["full_context", "hybrid", "hybrid_rerank"].includes(retrievalMode)) throw new ChatError("Choose a valid retrieval mode.");
    const revision = ++this.#revision;
    const controller = new AbortController();
    this.#controller = controller;
    this.pending = true;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeoutMs);
    const body = { question, response_mode: mode, include_evaluation: Boolean(includeEvaluation) };
    if (retrievalMode !== null) body.retrieval_mode = retrievalMode;
    if (this.conversationId) body.conversation_id = this.conversationId;
    try {
      // Native browser fetch needs a Window/undefined receiver, not a ChatSession object.
      const fetchRequest = this.fetchImpl;
      const response = await fetchRequest(`/v1/courses/${encodeURIComponent(this.course)}/query`, {
        method: "POST",
        headers: { ...this.authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify(body), signal: controller.signal, credentials: "omit", cache: "no-store", redirect: "error",
      });
      if (controller.signal.aborted || revision !== this.#revision) throw new DOMException("Cancelled", "AbortError");
      if (!response.ok) throw this.requestError(response.status, response.headers.get("X-Request-ID"), response.headers.get("Retry-After"));
      let data;
      try { data = await response.json(); } catch { throw new ChatError("The server returned an unreadable response. Check the Python server."); }
      if (controller.signal.aborted || revision !== this.#revision) throw new DOMException("Cancelled", "AbortError");
      const answer = validateResponse(data);
      if (retrievalMode !== null && answer.retrieval_mode !== retrievalMode) throw new ChatError("The server used a different retrieval mode. No fallback is accepted; restart the updated server.");
      this.conversationId = answer.conversation_id;
      return answer;
    } catch (error) {
      if (timedOut) throw new ChatError("The request timed out. The server may still finish it; check the server before retrying.");
      if (controller.signal.aborted || error.name === "AbortError") throw new ChatError("Stopped waiting. The server may still finish this request. Your question has been restored.");
      if (error instanceof ChatError) throw error;
      throw new ChatError("Could not reach the API. Check that the Python server is running, then retry.");
    } finally {
      clearTimeout(timer);
      if (revision === this.#revision) { this.pending = false; this.#controller = null; }
    }
  }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function initChat() {
  const $ = id => document.getElementById(id);
  const session = new ChatSession();
  const localCourse = document.body.dataset.localCourse;
  if (localCourse) session.configureLocal(localCourse);
  const transcript = [];
  let epoch = 0;
  let waiting = false;
  let retrievalInfo = null;
  let checkingRetrieval = false;
  let retrievalCheckRevision = 0;
  let retrievalError = "";
  const questionInput = $("question");
  const connectionDialog = $("connection-dialog");
  const sourceDialog = $("source-dialog");

  function selectedRetrieval() {
    const mode = $("retrieval-mode").value || retrievalInfo?.default_mode;
    return retrievalInfo?.modes.find(item => item.mode === mode);
  }

  function updateRetrieval() {
    const selected = selectedRetrieval();
    $("retrieval-mode").disabled = waiting || checkingRetrieval || !session.connected;
    $("refresh-retrieval").disabled = waiting || checkingRetrieval || !session.connected;
    $("send-question").disabled = waiting || checkingRetrieval || (session.connected && !selected?.available);
    const help = $("retrieval-help");
    help.classList.toggle("blocked", Boolean(retrievalError || (selected && !selected.available)));
    if (!session.connected) help.textContent = "Connect a course to check which modes are ready to test.";
    else if (checkingRetrieval) help.textContent = "Checking course embeddings and local reranker setup. No generation or downloads.";
    else if (retrievalError) help.textContent = retrievalError;
    else if (selected) {
      const descriptions = { full_context: `All ${retrievalInfo.passage_count} chunks go to the LLM evidence selector.`, hybrid: "Keyword + embedding search before LLM evidence selection.", hybrid_rerank: "Hybrid search, then cross-encoder reranking before LLM evidence selection." };
      help.textContent = selected.available ? `${descriptions[selected.mode]} Ready to test; inference is not yet verified.` : `Needs setup: ${selected.reasons[0]} Open mode setup for all requirements.`;
    } else help.textContent = "This server default is diagnostic only. Configure an LLM backend to test these modes.";
    $("retrieval-setup").hidden = !retrievalInfo;
    $("retrieval-status-list").replaceChildren();
    if (retrievalInfo) {
      $("retrieval-mode").options[0].textContent = `Server default (${retrievalLabels[retrievalInfo.default_mode]})`;
      for (const item of retrievalInfo.modes) {
        const option = [...$("retrieval-mode").options].find(option => option.value === item.mode);
        option.textContent = `${retrievalLabels[item.mode]}${item.available ? "" : " — needs setup"}`;
        const li = element("li");
        li.append(element("strong", "", `${retrievalLabels[item.mode]}: ${item.available ? "Ready to test" : "Needs setup"}. `));
        li.append(document.createTextNode([...item.reasons, ...item.warnings].join(" ")));
        $("retrieval-status-list").append(li);
      }
    }
  }

  async function refreshRetrieval() {
    const revision = ++retrievalCheckRevision;
    checkingRetrieval = true;
    retrievalError = "";
    updateRetrieval();
    try {
      const info = await session.retrievalOptions();
      if (revision === retrievalCheckRevision) retrievalInfo = info;
    } catch (error) {
      if (revision === retrievalCheckRevision) { retrievalInfo = null; retrievalError = error.message; }
    } finally {
      if (revision === retrievalCheckRevision) { checkingRetrieval = false; updateRetrieval(); }
    }
  }

  function updateConnection() {
    const expired = session.connected && Date.now() >= session.expiresAt;
    $("connection-label").textContent = session.connected ? session.course : "Connect a course";
    $("connection-button").classList.toggle("connected", session.connected);
    $("connection-button").classList.toggle("expired", expired);
    $("token-status").textContent = session.local ? `Local testing · ${session.course} · no JWT required` : !session.connected ? "Service token required" : expired ? "Token expired · update connection" : `Token valid until ${new Date(session.expiresAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    $("disconnect").hidden = !session.connected;
  }

  function openConnection() {
    if (localCourse) return;
    $("course-id").value = session.course || $("course-id").value;
    $("service-token").value = "";
    $("connection-error").textContent = "";
    connectionDialog.showModal();
  }

  function setWaiting(value) {
    waiting = value;
    $("send-question").disabled = value;
    $("connection-button").disabled = value;
    $("cancel-request").hidden = !value;
    for (const input of document.querySelectorAll('input[name="response-mode"], #include-evaluation')) input.disabled = value;
    $("question-form").setAttribute("aria-busy", String(value));
    updateRetrieval();
  }

  function resetChat() {
    epoch += 1;
    session.newConversation();
    transcript.length = 0;
    $("messages").replaceChildren();
    $("welcome").hidden = false;
    $("export-chat").disabled = true;
    $("activity").textContent = "New conversation. Previous questions will not be sent.";
    questionInput.value = "";
    setWaiting(false);
    updateCount();
    sourceDialog.close();
    questionInput.focus();
  }

  function messageShell(role, badge = null) {
    $("welcome").hidden = true;
    const article = element("article", `message ${role}-message`);
    const heading = element("div", "message-heading");
    heading.append(element("strong", "", role === "user" ? "You" : "Course assistant"));
    if (badge) heading.append(element("span", `response-label ${badge.kind}`, badge.text));
    article.append(heading);
    $("messages").append(article);
    return article;
  }

  function showSource(excerpt, index) {
    const citation = excerpt.citation;
    $("source-title").textContent = `Source ${index}`;
    const container = $("source-content");
    container.replaceChildren();
    const title = element("h3", "source-document", citation.document_title);
    const chapter = element("p", "source-chapter", citation.chapter_title || "Course document");
    title.dir = chapter.dir = "auto";
    container.append(title, chapter);
    const location = element("div", "source-location");
    location.append(element("div", "", `Paragraphs ${citation.paragraph_start}–${citation.paragraph_end} · Document version ${citation.version}`));
    location.append(element("div", "", `Section: ${citation.section_kind}`));
    container.append(location);
    const text = element("p", "message-text", excerpt.text);
    text.dir = "auto";
    container.append(text);
    const details = element("details", "test-details");
    details.append(element("summary", "", "Source identifiers"), element("pre", "", JSON.stringify({ excerpt_id: excerpt.id, ...citation }, null, 2)));
    container.append(details);
    if (!sourceDialog.open) sourceDialog.showModal();
  }

  function sourceButton(excerpt, index, inline = false) {
    const button = element("button", inline ? "citation-button" : "source-link");
    button.type = "button";
    button.setAttribute("aria-label", `Open source ${index}`);
    if (inline) button.textContent = `[${index}]`;
    else {
      button.append(element("span", "source-number", index));
      const title = element("span", "", excerpt.citation.chapter_title || excerpt.citation.document_title);
      title.dir = "auto";
      button.append(title);
    }
    button.addEventListener("click", () => showSource(excerpt, index));
    return button;
  }

  function renderAnswer(answer, elapsed, includeDetails) {
    const labels = { answered: "Answered", partial: "Partially answered", refused: "Not answered", clarification: "Clarification needed", error: "Answer could not be verified" };
    const article = messageShell("assistant", answer.status === "conversation" ? null : { kind: answer.status, text: labels[answer.status] });
    article.querySelector(".message-heading").append(element("span", "retrieval-label", retrievalLabels[answer.retrieval_mode]));
    const excerpts = answer.excerpts;
    if (answer.answer_kind === "multi") {
      const indexById = new Map(excerpts.map((excerpt, index) => [excerpt.id, index]));
      answer.parts.forEach((part, partIndex) => {
        const heading = element("h3", "source-chapter", `${partIndex + 1}. ${part.question}`);
        heading.dir = "auto";
        article.append(heading, element("p", "small-note", `${labels[part.status]} · ${part.answer_kind === "metadata" ? "Course metadata" : "Course content"}`));
        if (part.status === "answered" && part.explanation?.statements?.length) {
          part.explanation.statements.forEach(statement => {
            const p = element("p", "statement", statement.text);
            p.dir = "auto";
            statement.citation_ids.forEach(id => { const i = indexById.get(id); p.append(sourceButton(excerpts[i], i + 1, true)); });
            article.append(p);
          });
        } else if (part.status === "answered" && part.answer_kind === "content" && answer.response_mode === "verbatim") {
          part.excerpt_ids.forEach(id => { const i = indexById.get(id); const p = element("p", "message-text", excerpts[i].text); p.dir = "auto"; article.append(p, sourceButton(excerpts[i], i + 1)); });
        } else {
          const p = element("p", "message-text", part.answer); p.dir = "auto"; article.append(p);
        }
        if (part.metadata) appendMetadata(article, part.metadata);
      });
      if (excerpts.length) { const sources = element("div", "sources-list"); excerpts.forEach((e, i) => sources.append(sourceButton(e, i + 1))); article.append(sources); }
    } else if (answer.status === "answered" && answer.explanation?.statements?.length) {
      const indexById = new Map(excerpts.map((excerpt, index) => [excerpt.id, index]));
      for (const statement of answer.explanation.statements) {
        const paragraph = element("p", "statement", statement.text);
        paragraph.dir = "auto";
        for (const id of statement.citation_ids) {
          const index = indexById.get(id);
          paragraph.append(sourceButton(excerpts[index], index + 1, true));
        }
        article.append(paragraph);
      }
      const sources = element("div", "sources-list");
      excerpts.forEach((excerpt, index) => sources.append(sourceButton(excerpt, index + 1)));
      article.append(sources);
    } else if (answer.status === "answered" && answer.response_mode === "verbatim" && excerpts.length) {
      excerpts.forEach((excerpt, index) => {
        const quote = element("div", "quote");
        const text = element("p", "message-text", excerpt.text);
        text.dir = "auto";
        quote.append(text, sourceButton(excerpt, index + 1));
        article.append(quote);
      });
    } else {
      const text = element("p", "message-text", answer.answer);
      text.dir = "auto";
      article.append(text);
    }
    if (answer.metadata) appendMetadata(article, answer.metadata);
    const details = element("details", "test-details");
    details.append(element("summary", "", `${answer.retrieval_mode.replaceAll("_", " ")} · ${(elapsed / 1000).toFixed(1)}s · Request details`));
    const metadata = { request_id: answer.request_id, conversation_id: answer.conversation_id, status: answer.status, reason_code: answer.reason_code, retrieval_mode: answer.retrieval_mode, response_mode: answer.response_mode, policy_version: answer.policy_version, browser_elapsed_ms: Math.round(elapsed) };
    if (includeDetails && answer.evaluation) metadata.evaluation = answer.evaluation;
    details.append(element("pre", "", JSON.stringify(metadata, null, 2)));
    article.append(details);
    transcript.push({ role: "assistant", response: answer, browser_elapsed_ms: Math.round(elapsed) });
    $("export-chat").disabled = false;
  }

  function appendMetadata(article, metadata) {
    const details = element("details", "test-details");
    details.append(element("summary", "", "Source: imported document metadata (not an LMS syllabus)"));
    details.append(element("pre", "", JSON.stringify(metadata, null, 2)));
    article.append(details);
  }

  function showError(error) {
    const article = messageShell("assistant", { kind: "error", text: "Request error" });
    const box = element("div", "error-message");
    box.append(element("p", "", error.message));
    if (error.requestId) box.append(element("p", "small-note", `Request ID: ${error.requestId}`));
    if (error.action) {
      const button = element("button", "quiet-button", error.action === "connection" ? "Update connection" : "Start a new chat");
      button.type = "button";
      button.addEventListener("click", error.action === "connection" ? openConnection : () => { const draft = questionInput.value; resetChat(); questionInput.value = draft; updateCount(); });
      box.append(button);
    }
    article.append(box);
    transcript.push({ role: "error", message: error.message, status: error.status, request_id: error.requestId });
    $("export-chat").disabled = false;
  }

  function scrollToComposer() {
    if (!connectionDialog.open && !sourceDialog.open) $("conversation-area").scrollTop = $("conversation-area").scrollHeight;
  }

  function updateCount() {
    const length = [...questionInput.value.trim()].length;
    $("question-count").textContent = `${length.toLocaleString()} / 2,000`;
    $("question-count").classList.toggle("invalid", length > 2000);
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 230)}px`;
  }

  $("question-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (waiting || checkingRetrieval) return;
    let question;
    try { question = validateQuestion(questionInput.value); session.checkConnection(); }
    catch (error) { $("activity").textContent = error.message; if (error.action === "connection") openConnection(); else questionInput.focus(); return; }
    if (!selectedRetrieval()?.available) { $("activity").textContent = "This retrieval mode needs setup. Check mode setup or refresh status before sending."; return; }
    const currentEpoch = epoch;
    const mode = document.querySelector('input[name="response-mode"]:checked').value;
    const includeDetails = $("include-evaluation").checked;
    const retrievalMode = $("retrieval-mode").value || null;
    const article = messageShell("user");
    const text = element("p", "message-text", question);
    text.dir = "auto";
    article.append(text);
    transcript.push({ role: "user", text: question, response_mode: mode, retrieval_mode: retrievalMode || retrievalInfo.default_mode });
    questionInput.value = "";
    updateCount();
    setWaiting(true);
    const started = performance.now();
    $("activity").textContent = "Reading the course…";
    const timer = setInterval(() => { if (epoch === currentEpoch) $("activity").textContent = `Reading the course… ${Math.floor((performance.now() - started) / 1000)}s`; }, 1000);
    scrollToComposer();
    try {
      const answer = await session.ask(question, mode, includeDetails, retrievalMode);
      if (currentEpoch !== epoch) return;
      renderAnswer(answer, performance.now() - started, includeDetails);
      $("activity").textContent = answer.answer_kind === "metadata" ? "Response received from imported document metadata. Expand its source details to inspect provenance." : answer.status === "partial" ? "Some questions were answered. Review each question's status and evidence." : "Response received. Open a citation to inspect the original text.";
    } catch (error) {
      if (currentEpoch !== epoch) return;
      showError(error);
      if (!questionInput.value.trim()) questionInput.value = question;
      updateCount();
      $("activity").textContent = "Question restored. No automatic retry was sent.";
    } finally {
      clearInterval(timer);
      if (currentEpoch === epoch) { setWaiting(false); updateConnection(); scrollToComposer(); questionInput.focus(); }
    }
  });

  questionInput.addEventListener("input", updateCount);
  questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); $("question-form").requestSubmit(); }
  });
  document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => { questionInput.value = button.dataset.question; updateCount(); questionInput.focus(); }));
  $("connection-button").addEventListener("click", openConnection);
  $("close-connection").addEventListener("click", () => connectionDialog.close());
  connectionDialog.addEventListener("close", () => { $("service-token").value = ""; });
  $("connection-form").addEventListener("submit", event => {
    event.preventDefault();
    try {
      const changed = session.configure($("course-id").value, $("service-token").value);
      if (changed) {
        retrievalInfo = null;
        const draft = questionInput.value;
        resetChat();
        questionInput.value = draft;
        updateCount();
      }
      connectionDialog.close();
      updateConnection();
      void refreshRetrieval();
      $("activity").textContent = "Checking authorized retrieval modes for this course.";
      questionInput.focus();
    } catch (error) { $("connection-error").textContent = error.message; }
  });
  $("disconnect").addEventListener("click", () => { session.disconnect(); retrievalCheckRevision++; checkingRetrieval = false; retrievalInfo = null; retrievalError = ""; resetChat(); connectionDialog.close(); updateConnection(); $("activity").textContent = "Disconnected. The service token has been cleared."; });
  $("refresh-retrieval").addEventListener("click", () => { void refreshRetrieval(); });
  $("retrieval-mode").addEventListener("change", () => {
    session.newConversation();
    updateRetrieval();
    $("activity").textContent = "Retrieval mode changed. The next question starts a new conversation for a clean comparison; earlier results stay visible.";
  });
  $("new-chat").addEventListener("click", resetChat);
  $("cancel-request").addEventListener("click", () => { session.cancel(); $("activity").textContent = "Stopping the browser wait; the server may still be working…"; });
  $("close-source").addEventListener("click", () => sourceDialog.close());
  $("export-chat").addEventListener("click", () => {
    // Explicit allowlist: never export a session object, JWT, auth header or provider key.
    const data = { exported_at: new Date().toISOString(), course_id: session.course, conversation_id: session.conversationId, messages: transcript };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const link = element("a");
    link.href = url;
    link.download = `course-chat-${session.course}-${Date.now()}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("pagehide", () => session.disconnect());
  window.addEventListener("pageshow", () => { if (localCourse && !session.connected) { session.configureLocal(localCourse); void refreshRetrieval(); } updateConnection(); });
  setInterval(updateConnection, 15000);
  updateConnection();
  updateRetrieval();
  updateCount();
  if (localCourse) void refreshRetrieval();
}

if (typeof document !== "undefined") initChat();
