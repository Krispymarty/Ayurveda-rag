'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const sources = {ipindia: 'IP India', indiacode: 'India Code', tkdl: 'TKDL'};
  const officialHosts = new Set(['ipindia.gov.in', 'www.ipindia.gov.in', 'indiacode.gov.in', 'www.indiacode.gov.in', 'indiacode.nic.in', 'www.indiacode.nic.in', 'upload.indiacode.nic.in', 'tkdl.res.in', 'www.tkdl.res.in']);
  const state = {mode: 'search', capabilities: null, ready: false, pending: false, controller: null, sequence: 0, sourceOpener: null};
  const initialResults = $('results').firstElementChild.cloneNode(true);
  const formatNumber = new Intl.NumberFormat('en-IN');
  const node = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined && text !== null) result.textContent = String(text);
    return result;
  };
  const announce = (text) => { $('announcement').textContent = text; };
  const dateLabel = (value) => {
    if (!value) return 'Not recorded';
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? 'Not recorded' : new Intl.DateTimeFormat('en-IN', {dateStyle: 'medium', timeZone: 'Asia/Kolkata'}).format(date);
  };
  function citationUrl(citation) {
    try {
      const url = new URL(citation.source_url);
      if (url.protocol !== 'https:' || !officialHosts.has(url.hostname.toLowerCase()) || url.username || url.password || (url.port && url.port !== '443')) return null;
      // Only verified PDF page anchors are synthesized; section labels are shown as text.
      if (/\.pdf$/i.test(url.pathname) && Number.isInteger(citation.page_start) && citation.page_start > 0) url.hash = `page=${citation.page_start}`;
      return url.href;
    } catch (_) { return null; }
  }
  function linkFor(citation, text, className = 'citation-link') {
    const href = citationUrl(citation);
    if (!href) return node('span', className, `${text} · citation unavailable`);
    const a = node('a', className, text);
    a.href = href;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    return a;
  }
  function anchorLabel(item) {
    const parts = [];
    if (item.section) parts.push(`Section ${item.section}`);
    if (item.rule) parts.push(`Rule ${item.rule}`);
    if (Number.isInteger(item.page_start) && item.page_start > 0) {
      parts.push(item.page_end > item.page_start ? `pp. ${item.page_start}–${item.page_end}` : `p. ${item.page_start}`);
    }
    return parts.length ? parts.join(' · ') : 'No verified section or page anchor';
  }
  function addDefinition(list, label, value) {
    list.append(node('dt', '', label), node('dd', '', value || 'Not recorded'));
  }
  function provenance(hit) {
    const details = node('details', 'provenance');
    details.append(node('summary', '', 'Archive & citation details'));
    const dl = node('dl');
    addDefinition(dl, 'Document ID', hit.document_id);
    addDefinition(dl, 'Passage ID', hit.chunk_id);
    addDefinition(dl, 'Source SHA-256', hit.raw_sha256);
    addDefinition(dl, 'Captured', dateLabel(hit.retrieved_at));
    addDefinition(dl, 'Official URL', hit.source_url);
    details.append(dl);
    return details;
  }
  function renderHit(hit, index) {
    const article = node('article', 'evidence-hit');
    const eyebrow = node('div', 'hit-eyebrow');
    eyebrow.append(node('span', 'hit-number', String(index + 1).padStart(2, '0')), node('span', 'hit-source', sources[hit.source] || 'Official evidence'));
    if (hit.legal_domain) eyebrow.append(node('span', '', hit.legal_domain));
    if (hit.evidence_role === 'institutional_background') eyebrow.append(node('span', 'hit-role', 'Institutional background'));
    if (hit.short_unit) eyebrow.append(node('span', 'hit-role', 'Short provision'));
    const title = node('h3');
    const link = linkFor(hit, hit.title || 'Untitled official document');
    const mark = node('span', 'external-mark', '↗');
    mark.setAttribute('aria-hidden', 'true');
    if (link.tagName === 'A') link.append(mark);
    title.append(link);
    const anchor = node('div', 'hit-anchor');
    anchor.append(node('span', '', anchorLabel(hit)), node('span', 'legal-status', 'Legal status unverified'));
    const passage = node('p', 'hit-passage collapsed', hit.text);
    const passageId = `passage-${state.sequence}-${index}`;
    passage.id = passageId;
    const footer = node('div', 'hit-footer');
    const expand = node('button', 'text-button hit-toggle', 'Read full passage +');
    expand.type = 'button';
    expand.setAttribute('aria-expanded', 'false');
    expand.setAttribute('aria-controls', passageId);
    expand.addEventListener('click', () => {
      const expanded = expand.getAttribute('aria-expanded') !== 'true';
      passage.classList.toggle('collapsed', !expanded);
      expand.setAttribute('aria-expanded', String(expanded));
      expand.textContent = expanded ? 'Show less −' : 'Read full passage +';
    });
    footer.append(expand, node('span', '', `Archived ${dateLabel(hit.retrieved_at)}`));
    article.append(eyebrow, title, anchor, passage, footer, provenance(hit));
    return article;
  }
  function message(title, body, error = false) {
    const block = node('div', `state-block${error ? ' error' : ''}`);
    block.append(node('h3', '', title), node('p', '', body));
    $('results').replaceChildren(block);
    return block;
  }
  function renderSearch(data, elapsed) {
    if (!Array.isArray(data.hits)) throw new Error('invalid_response');
    $('results-title').textContent = data.hits.length ? 'Evidence, with context' : 'No matching passages';
    $('results-meta').textContent = `${data.hits.length} passage${data.hits.length === 1 ? '' : 's'} · ${(elapsed / 1000).toFixed(2)} s`;
    if (!data.hits.length) {
      message('Try a more specific question.', 'Use an instrument title, section number or a different phrase. A source with no indexed documents cannot return evidence. You can also reset your filters.');
      announce('No matching passages. Try another query or reset your filters.');
      return;
    }
    const fragment = document.createDocumentFragment();
    if (data.notice) fragment.append(node('p', 'result-notice', data.notice));
    data.hits.forEach((hit, index) => fragment.append(renderHit(hit, index)));
    $('results').replaceChildren(fragment);
    announce(`${data.hits.length} evidence passages found.`);
  }
  function searchInstead() {
    const button = node('button', 'text-button', 'Search the evidence instead ↗');
    button.type = 'button';
    button.addEventListener('click', () => { setMode('search'); $('search-form').requestSubmit(); });
    return button;
  }
  function answerReason(reason, unavailable) {
    const messages = {
      current_legal_status_unverified: 'The archived evidence does not verify current legal status. Check authoritative amendments and commencement information before relying on a provision.',
      institutional_background_only: 'Only institutional background was found. It cannot establish a legal rule.',
      personal_legal_or_medical_advice: 'This service cannot provide personal legal or medical advice. You can search the official evidence for a specific instrument or provision.',
      unsafe_instruction_request: 'Ask a research question about an instrument or provision so the service can ground the response in archived evidence.',
      answer_generation_disabled: 'Answer generation is not enabled. You can search and inspect the archived source passages.',
      provider_timeout: 'The local answer provider took too long to respond. Try again or search the evidence directly.',
      model_digest_mismatch: 'The configured local answer provider could not be verified. Evidence search remains available.',
      model_identity_mismatch: 'The configured local answer provider could not be verified. Evidence search remains available.'
    };
    return messages[reason] || (unavailable ? 'The answer service is not available. Search archived passages directly.' : 'The available passages cannot support a verified response. Try a precise provision or inspect the evidence.');
  }
  function renderAnswer(data, elapsed) {
    $('results-meta').textContent = `${(elapsed / 1000).toFixed(2)} s`;
    if (data.status !== 'answered') {
      const unavailable = data.status === 'unavailable';
      $('results-title').textContent = unavailable ? 'Answer generation unavailable' : 'More evidence is needed';
      const block = message(unavailable ? 'Evidence search is still available.' : 'This question needs further verification.', answerReason(data.reason, unavailable));
      block.append(searchInstead());
      if (data.notice) block.append(node('p', 'result-notice', data.notice));
      announce(unavailable ? 'Answer generation is unavailable.' : 'The service abstained. More evidence is needed.');
      return;
    }
    // Do not show an uncited answer body when the server has no validated claims.
    const claims = Array.isArray(data.claims) ? data.claims : [];
    if (!claims.length || claims.some((claim) => !Array.isArray(claim.citations) || !claim.citations.length || claim.citations.some((citation) => !citationUrl(citation) || !citation.quote))) {
      $('results-title').textContent = 'Response could not be verified';
      message('Verified citations are required.', 'The response did not contain complete official citations. Search the source passages directly.').append(searchInstead());
      announce('Response hidden because verified citations are missing.');
      return;
    }
    $('results-title').textContent = 'A response grounded in evidence';
    const fragment = document.createDocumentFragment();
    fragment.append(node('p', 'answer-intro', 'VERIFIED EXTRACTS · CHECK THE CONTEXT'));
    fragment.append(node('p', 'result-notice', data.notice || 'These are archived source extracts, not a determination of current law or legal advice.'));
    claims.forEach((claim, claimIndex) => {
      const article = node('article', 'answer-claim');
      article.append(node('p', 'claim-text', claim.text));
      const citations = node('div', 'claim-citations');
      claim.citations.forEach((citation, citationIndex) => {
        const label = `[${claimIndex + 1}.${citationIndex + 1}] ${citation.title || 'Official source'} · ${anchorLabel(citation)}`;
        citations.append(linkFor(citation, label, 'claim-citation'));
      });
      article.append(citations);
      const details = node('details', 'citation-evidence');
      details.append(node('summary', '', 'Inspect cited extracts'));
      claim.citations.forEach((citation) => {
        details.append(node('blockquote', '', citation.quote));
        details.append(provenance(citation));
      });
      article.append(details);
      fragment.append(article);
    });
    $('results').replaceChildren(fragment);
    announce(`${claims.length} verified extract${claims.length === 1 ? '' : 's'} with official citations.`);
  }
  function answerReady() {
    const c = state.capabilities;
    return Boolean(c && c.answer_generation_enabled && c.provider_configured !== false);
  }
  function answerUnavailableMessage() {
    return state.capabilities?.answer_generation_reason === 'three_source_not_certified'
      ? 'Answers are waiting for India Code and TKDL source certification. Evidence search remains available.'
      : 'The local answer provider is not enabled or available. Evidence search remains available.';
  }
  function updateSubmit() {
    $('submit-button').disabled = !state.ready || state.pending || (state.mode === 'answer' && !answerReady());
    $('submit-label').textContent = state.pending ? (state.mode === 'answer' ? 'Checking evidence…' : 'Searching…') : (state.mode === 'answer' ? 'Ask with citations' : 'Find evidence');
  }
  function setMode(mode) {
    if (state.mode === mode) return;
    if (state.controller) state.controller.abort();
    state.sequence += 1;
    state.pending = false;
    state.mode = mode;
    $('results').setAttribute('aria-busy', 'false');
    for (const value of ['search', 'answer']) {
      $(`mode-${value}`).classList.toggle('active', mode === value);
      $(`mode-${value}`).setAttribute('aria-pressed', String(mode === value));
    }
    $('top-k').disabled = mode === 'answer';
    if (mode === 'answer') $('top-k').value = '5';
    $('question-label').textContent = mode === 'answer' ? 'What would you like to verify?' : 'What are you researching?';
    $('answer-availability').hidden = mode !== 'answer';
    $('answer-availability').textContent = answerReady() ? 'Answers use verified archived extracts with claim-level citations. Questions about current legal status or personal legal advice may require abstention.' : answerUnavailableMessage();
    $('results-title').textContent = mode === 'answer' ? 'Ask, then inspect the evidence' : 'Your research starts here';
    $('results-meta').textContent = '';
    $('results').replaceChildren(initialResults.cloneNode(true));
    updateSubmit();
    $('question').focus();
  }
  async function requestJson(url, options = {}, timeout = 15000, allowUnavailable = false) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {credentials: 'same-origin', cache: 'no-store', ...options, signal: controller.signal});
      if (!response.ok && !(allowUnavailable && response.status === 503)) throw new Error(`http_${response.status}`);
      return await response.json();
    } finally { window.clearTimeout(timer); }
  }
  async function submit(event) {
    event.preventDefault();
    if (!state.ready) { announce('Evidence search is preparing or unavailable. Your question has been kept.'); return; }
    const query = $('question').value.trim();
    if (query.length < 2 || query.length > 500) {
      $('question').setCustomValidity('Enter a research question between 2 and 500 characters.');
      $('question').reportValidity();
      return;
    }
    $('question').setCustomValidity('');
    if (state.mode === 'answer' && !answerReady()) return;
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    const sequence = ++state.sequence;
    const mode = state.mode;
    state.controller = controller;
    state.pending = true;
    updateSubmit();
    $('results').setAttribute('aria-busy', 'true');
    $('results-title').textContent = mode === 'answer' ? 'Checking the source evidence' : 'Searching the archive';
    $('results-meta').textContent = '';
    const loading = node('div', 'state-block');
    loading.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 3; i += 1) loading.append(node('div', 'loading-line'));
    $('results').replaceChildren(loading);
    announce(mode === 'answer' ? 'Checking archived evidence and citations.' : 'Searching archived evidence.');
    const source = $('source-filter').value || null;
    const domain = $('domain-filter').value || null;
    const params = new URLSearchParams({q: query, top_k: $('top-k').value});
    if (source) params.set('source', source);
    if (domain) params.set('domain', domain);
    let timedOut = false;
    const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, mode === 'answer' ? 90000 : 45000);
    const started = performance.now();
    try {
      const response = await fetch(mode === 'search' ? `/search?${params}` : '/answer', {
        method: mode === 'search' ? 'GET' : 'POST', credentials: 'same-origin', cache: 'no-store', signal: controller.signal,
        ...(mode === 'answer' ? {headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: query, top_k: 5, source, domain})} : {})
      });
      if (sequence !== state.sequence) return;
      if (!response.ok) throw new Error(`http_${response.status}`);
      const data = await response.json();
      if (sequence !== state.sequence) return;
      if (mode === 'search') renderSearch(data, performance.now() - started);
      else renderAnswer(data, performance.now() - started);
    } catch (error) {
      if (sequence !== state.sequence || (error.name === 'AbortError' && !timedOut)) return;
      $('results-title').textContent = 'Research is temporarily unavailable';
      const rateLimited = error.message === 'http_429';
      message(rateLimited ? 'Please pause before trying again.' : timedOut ? 'The request took too long.' : 'We could not complete this request.', rateLimited ? 'The service is receiving too many requests. Wait a moment and try again.' : 'Your question is still in the search field. Try again shortly, or check source status for service availability.', true);
      announce(rateLimited ? 'Too many requests. Please wait and try again.' : 'The request failed. Your question has been preserved.');
    } finally {
      window.clearTimeout(timer);
      if (sequence === state.sequence) {
        state.pending = false;
        state.controller = null;
        $('results').setAttribute('aria-busy', 'false');
        updateSubmit();
      }
    }
  }
  function filterCount() {
    const count = Number(Boolean($('source-filter').value)) + Number(Boolean($('domain-filter').value));
    $('filter-count').textContent = count ? `(${count})` : '';
  }
  function toggleSources(open, opener) {
    $('source-panel').hidden = !open;
    for (const id of ['status-toggle', 'source-detail-button']) $(id).setAttribute('aria-expanded', String(open));
    if (open) {
      state.sourceOpener = opener;
      $('source-panel-title').focus();
      $('source-panel').scrollIntoView({block: 'nearest'});
    } else if (state.sourceOpener) state.sourceOpener.focus();
  }
  function renderSources(data) {
    const rows = Array.isArray(data.sources) ? data.sources : [];
    const indexed = rows.filter((item) => Number(item.searchable_document_count) > 0);
    const total = rows.reduce((sum, item) => sum + (Number(item.searchable_document_count) || 0), 0);
    const sourceList = document.createDocumentFragment();
    const details = document.createDocumentFragment();
    for (const id of Object.keys(sources)) {
      const item = rows.find((value) => value.source_id === id);
      const count = Number(item?.searchable_document_count) || 0;
      const row = node('div', 'source-row');
      const label = node('span', '', sources[id]);
      if (id === 'tkdl') label.append(node('small', '', 'public pages'));
      row.append(label, node('span', `source-count${count ? ' available' : ''}`, count ? `${formatNumber.format(count)} documents` : 'Not indexed'));
      sourceList.append(row);
      const entry = node('article', 'source-status-entry');
      entry.append(node('h3', '', sources[id]));
      const dl = node('dl');
      addDefinition(dl, 'Searchable documents', formatNumber.format(count));
      addDefinition(dl, 'Last capture', dateLabel(item?.last_successful_sync));
      addDefinition(dl, 'Last publication', dateLabel(item?.last_successful_publication));
      addDefinition(dl, 'Access review', item?.access_review_status === 'approved' ? 'Approved public paths' : 'Unresolved · fresh capture deferred');
      addDefinition(dl, 'Evidence role', id === 'tkdl' ? 'Institutional background' : 'Official archived evidence');
      addDefinition(dl, 'Legal status', 'Unknown');
      if ((item?.pipeline_failures || []).length || (item?.recent_review_items || []).length) addDefinition(dl, 'Review items', 'Source review required before new capture');
      entry.append(dl);
      details.append(entry);
    }
    $('source-list').replaceChildren(sourceList);
    $('source-details').replaceChildren(details);
    $('coverage-note').textContent = data.index_available ? `${formatNumber.format(total)} documents searchable from ${indexed.length} of 3 sources. Three-source certification is pending.` : 'No active searchable index. Source certification is pending.';
    $('source-summary').textContent = 'Counts refer to the active searchable collection. Archived or staged files are not necessarily accepted or indexed evidence. Fresh synchronization is deferred.';
    $('generation-line').textContent = data.active_generation_id ? `Active collection: ${data.active_generation_id}` : 'No active collection is available.';
  }
  function showHealth(data) {
    const healthy = data?.status === 'ok';
    state.ready = healthy;
    const preparing = ['not_started', 'initializing'].includes(data?.readiness?.state);
    $('health-label').textContent = healthy ? 'Evidence search available' : preparing ? 'Preparing evidence search…' : 'Service unavailable';
    $('health-dot').className = `status-dot ${healthy ? 'ready' : 'error'}`;
    updateSubmit();
    if (preparing) window.setTimeout(async () => {
      try { showHealth(await requestJson('/health', {}, 5000, true)); }
      catch (_) { showHealth(null); }
    }, 2000);
  }
  async function initialize() {
    const results = await Promise.allSettled([requestJson('/health', {}, 15000, true), requestJson('/sources'), requestJson('/capabilities')]);
    const [health, sourceResult, capabilities] = results;
    showHealth(health.status === 'fulfilled' ? health.value : null);
    if (sourceResult.status === 'fulfilled') renderSources(sourceResult.value);
    else {
      $('coverage-note').textContent = 'Source counts are currently unavailable. Open source status for details.';
      $('source-summary').textContent = 'Source status could not be loaded. Please try again later.';
      $('source-list').querySelectorAll('.source-count').forEach((item) => { item.textContent = 'Unavailable'; });
    }
    if (capabilities.status === 'fulfilled') state.capabilities = capabilities.value;
    if (state.mode === 'answer') {
      $('answer-availability').textContent = answerReady() ? 'Answers use verified archived extracts with claim-level citations. Current-status questions may require abstention.' : answerUnavailableMessage();
    }
    updateSubmit();
  }
  $('search-form').addEventListener('submit', submit);
  $('question').addEventListener('input', () => { $('question').setCustomValidity(''); $('character-count').textContent = `${$('question').value.length} / 500`; });
  $('question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!state.pending) $('search-form').requestSubmit(); } });
  $('mode-search').addEventListener('click', () => setMode('search'));
  $('mode-answer').addEventListener('click', () => setMode('answer'));
  $('filter-toggle').addEventListener('click', () => { const expanded = $('filter-toggle').getAttribute('aria-expanded') !== 'true'; $('filters').hidden = !expanded; $('filter-toggle').setAttribute('aria-expanded', String(expanded)); });
  $('source-filter').addEventListener('change', filterCount);
  $('domain-filter').addEventListener('change', filterCount);
  $('reset-filters').addEventListener('click', () => { $('source-filter').value = ''; $('domain-filter').value = ''; $('top-k').value = '5'; filterCount(); });
  $('suggestions').querySelectorAll('[data-query]').forEach((button) => button.addEventListener('click', () => { $('question').value = button.dataset.query; $('question').dispatchEvent(new Event('input')); $('search-form').requestSubmit(); }));
  for (const id of ['status-toggle', 'source-detail-button']) $(id).addEventListener('click', () => toggleSources($('source-panel').hidden, $(id)));
  $('close-sources').addEventListener('click', () => toggleSources(false));
  $('source-panel').addEventListener('keydown', (event) => { if (event.key === 'Escape') toggleSources(false); });
  initialize();
})();
