'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  let language = 'en';
  try { if (localStorage.getItem('ayurveda-language') === 'hi') language = 'hi'; } catch (_) { /* Preference storage is optional. */ }
  const tr = (en, hi) => language === 'hi' ? hi : en;
  const languageButton = document.querySelector('.language-button');
  const placeholders = [...document.querySelectorAll('[data-placeholder-hi]')].map(el => [el, el.placeholder]);
  function setLanguage(next) {
    language = next;
    document.documentElement.lang = next;
    document.querySelectorAll('[data-en][data-hi]').forEach(element => {
      // The two formatting tags come from static translations. Render them
      // explicitly so source text and translated copy never enter innerHTML.
      const parts = element.dataset[next].split(/(<em>[^<]*<\/em>|<br>)/);
      element.replaceChildren(...parts.map(part => {
        if (part === '<br>') return document.createElement('br');
        if (part.startsWith('<em>') && part.endsWith('</em>')) {
          const emphasis = document.createElement('em');
          emphasis.textContent = part.slice(4, -5);
          return emphasis;
        }
        return document.createTextNode(part);
      }));
    });
    for (const [element, english] of placeholders) element.placeholder = tr(english, element.dataset.placeholderHi);
    for (const [attribute, selector] of [['aria-label', 'aria'], ['alt', 'alt']]) {
      document.querySelectorAll('[data-' + selector + '-en][data-' + selector + '-hi]').forEach(element =>
        element.setAttribute(attribute, element.getAttribute('data-' + selector + '-' + next)));
    }
    if (languageButton) {
      languageButton.querySelector('.language-label').textContent = tr('हिन्दी', 'English');
      languageButton.setAttribute('aria-label', tr('Switch language to Hindi', 'Switch language to English'));
    }
    try { localStorage.setItem('ayurveda-language', next); } catch (_) { /* Preference storage is optional. */ }
  }
  if (languageButton) languageButton.addEventListener('click', () => setLanguage(tr('hi', 'en')));
  setLanguage(language);
  if (!$('search-form')) return;
  const sources = {ipindia: 'IP India', indiacode: 'India Code', tkdl: 'TKDL'};
  const officialHosts = new Set(['ipindia.gov.in', 'www.ipindia.gov.in', 'indiacode.gov.in', 'www.indiacode.gov.in', 'indiacode.nic.in', 'www.indiacode.nic.in', 'upload.indiacode.nic.in', 'tkdl.res.in', 'www.tkdl.res.in']);
  const state = {demoEnabled: false, ready: false, pending: false, controller: null, sequence: 0, sourceOpener: null, healthPending: false, healthTimer: null};
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
  function linkFor(citation, text, className = 'official-link') {
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
    const details = node('details', 'provenance-box');
    details.append(node('summary', '', 'Archive & citation details'));
    const dl = node('dl', 'provenance-list');
    addDefinition(dl, 'Document ID', hit.document_id);
    addDefinition(dl, 'Passage ID', hit.chunk_id);
    addDefinition(dl, 'Source SHA-256', hit.raw_sha256);
    addDefinition(dl, 'Captured', dateLabel(hit.retrieved_at));
    addDefinition(dl, 'Official URL', hit.source_url);
    details.append(dl);
    return details;
  }
  function renderHit(hit, index) {
    const article = node('article', 'evidence-card');
    const eyebrow = node('div', 'evidence-eyebrow');
    eyebrow.append(node('span', 'hit-badge', String(index + 1).padStart(2, '0')), node('span', 'source-pill', sources[hit.source] || 'Official evidence'));
    if (hit.legal_domain) eyebrow.append(node('span', '', hit.legal_domain));
    if (hit.evidence_role === 'institutional_background') eyebrow.append(node('span', 'hit-role', 'Institutional background'));
    if (hit.short_unit) eyebrow.append(node('span', 'hit-role', 'Short provision'));
    const title = node('h3', 'evidence-title');
    const link = linkFor(hit, hit.title || 'Untitled official document');
    const mark = node('span', 'external-mark', '↗');
    mark.setAttribute('aria-hidden', 'true');
    if (link.tagName === 'A') link.append(mark);
    title.append(link);
    const anchor = node('div', 'evidence-anchor');
    anchor.append(node('span', '', anchorLabel(hit)), node('span', 'status-pill', 'Legal status unknown'));
    const passage = node('p', 'evidence-quote collapsed', hit.text);
    const passageId = `passage-${state.sequence}-${index}`;
    passage.id = passageId;
    const footer = node('div', 'card-footer');
    const expand = node('button', 'text-button toggle-quote', 'Read full passage +');
    expand.type = 'button';
    expand.setAttribute('aria-expanded', 'false');
    expand.setAttribute('aria-controls', passageId);
    expand.addEventListener('click', () => {
      const expanded = expand.getAttribute('aria-expanded') !== 'true';
      passage.classList.toggle('collapsed', !expanded);
      expand.setAttribute('aria-expanded', String(expanded));
      expand.textContent = expanded ? 'Show less −' : 'Read full passage +';
    });
    footer.append(expand, linkFor(hit, 'Open official document ↗', 'text-button official-document'));
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
    $('results-title').textContent = data.hits.length ? tr('Verified evidence passages', 'सत्यापित साक्ष्य अंश') : tr('No matching passages', 'कोई संबंधित अंश नहीं मिला');
    $('results-meta').textContent = `${data.hits.length} passage${data.hits.length === 1 ? '' : 's'} · ${(elapsed / 1000).toFixed(2)} s`;
    if (!data.hits.length) {
      message('Try a different research phrase.', 'Use an instrument title, section number or broader topic. Reset filters if needed. No result does not establish that a legal rule is absent.');
      announce('No matching passages. Try another query or reset your filters.');
      return;
    }
    const fragment = document.createDocumentFragment();
    if (data.notice) fragment.append(node('p', 'result-notice', data.notice));
    data.hits.forEach((hit, index) => fragment.append(renderHit(hit, index)));
    $('results').replaceChildren(fragment);
    announce(`${data.hits.length} evidence passages found.`);
  }
  function demoMode() { return state.demoEnabled && $('research-mode').value === 'answer'; }
  function addEvidenceAction(parent, selection) {
    const button = node('button', 'text-button evidence-action', tr('Inspect matching passages', 'मिलते-जुलते अंश देखें'));
    button.type = 'button';
    button.addEventListener('click', () => {
      $('question').value = selection.query;
      $('question').dispatchEvent(new Event('input'));
      $('domain-filter').value = selection.domain;
      $('source-filter').value = 'ipindia';
      $('research-mode').value = 'search';
      $('research-mode').dispatchEvent(new Event('change'));
      filterCount();
      runQuery(selection, {inspect: true});
    });
    parent.append(button);
  }
  function renderAnswer(data, elapsed, selection) {
    const unavailable = data.status === 'unavailable';
    const titles = {personal_legal_or_medical_advice: 'Research or personal advice?', current_legal_status_unverified: 'Current legal status is not established'};
    $('results-title').textContent = data.status === 'answered' ? tr('Verified research extracts', 'सत्यापित शोध अंश') : unavailable ? 'AI service temporarily unavailable' : titles[data.reason] || 'No supporting extract selected';
    $('results-meta').textContent = `IP India demo · ${Math.round(elapsed)} ms · ${data.provider?.model || 'No model call'}`;
    const fragment = document.createDocumentFragment();
    if (data.notice) fragment.append(node('p', 'result-notice', data.notice));
    if (!data.claims?.length) {
      const reasons = {
        personal_legal_or_medical_advice: 'The request-policy check classified this question as personal legal or medical advice. If you meant archived research, ask what a named section or rule says.',
        current_legal_status_unverified: 'The archive cannot establish whether a provision is currently in force.',
        insufficient_relevant_evidence: 'The model did not select enough directly relevant archived evidence.',
        provider_rate_limited: 'The model provider has reached its request limit. Please wait before trying again.',
        answer_daily_limit: 'This site has reached its daily limit for AI answers.',
        answer_browser_limit: 'This browser has reached its hourly limit for AI answers.',
        service_busy: 'Another AI request is in progress.',
        answer_session_required: 'The AI browser session expired. You can try again or inspect the archive.',
        answer_quota_unavailable: 'AI answers are temporarily unavailable.',
        provider_auth_failed: 'The model provider rejected the configured credentials. The service owner needs to check its configuration.',
        provider_request_rejected: 'The model provider rejected this request. No quotation was published; this does not mean the archive has no relevant provision.'
      };
      const block = node('div', 'state-block');
      block.append(node('p', '', `${reasons[data.reason] || `No extract was published: ${(data.reason || 'insufficient evidence').replaceAll('_', ' ')}.`}`));
      addEvidenceAction(block, selection);
      fragment.append(block);
    }
    (data.claims || []).forEach((claim) => {
      const article = node('article', 'evidence-card');
      article.append(node('blockquote', 'evidence-quote', claim.text));
      (claim.citations || []).forEach((citation) => {
        article.append(linkFor(citation, `${citation.title || 'Official source'} · ${anchorLabel(citation)}`), provenance(citation));
      });
      fragment.append(article);
    });
    $('results').replaceChildren(fragment);
    announce(`${data.claims?.length || 0} verified extracts. ${data.reason || ''}`);
  }
  function updateSubmit() {
    $('submit-button').disabled = !state.ready || state.pending;
    $('consent-label').hidden = !demoMode();
    if (state.pending) $('submit-label').textContent = tr('Searching…', 'खोज जारी है…');
    else $('submit-label').textContent = demoMode() ? tr('Ask AyurIP', 'AyurIP से पूछें') : tr('Find evidence', 'साक्ष्य खोजें');
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
  function submit(event) {
    event.preventDefault();
    return runQuery({query: $('question').value.trim(), source: $('source-filter').value,
                     domain: $('domain-filter').value, topK: $('top-k').value});
  }
  async function runQuery(selection, {inspect = false} = {}) {
    if (!state.ready || state.pending) { announce('Evidence search is preparing or unavailable. Your research question has been kept.'); return; }
    const {query, source, domain, topK} = selection;
    const answerMode = !inspect && demoMode();
    if (query.length < 2 || query.length > 500) {
      $('question').setCustomValidity('Enter a research question between 2 and 500 characters.');
      $('question').reportValidity();
      return;
    }
    $('question').setCustomValidity('');
    if (answerMode && !$('demo-consent').checked) {
      $('demo-consent').setCustomValidity('Confirm non-confidential input and cloud-processing consent for the demo.');
      $('demo-consent').reportValidity();
      return;
    }
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    const sequence = ++state.sequence;
    state.controller = controller;
    state.pending = true;
    updateSubmit();
    $('results').setAttribute('aria-busy', 'true');
    $('results-title').textContent = 'Searching the archive';
    $('results-meta').textContent = '';
    const loading = node('div', 'state-block');
    loading.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 3; i += 1) loading.append(node('div', 'loading-line'));
    $('results').replaceChildren(loading);
    announce('Searching archived evidence.');
    if ($('research-results')) {
      $('research-results').hidden = false;
      $('edit-search').hidden = false;
      $('results-title').focus({preventScroll: true});
      $('research-results').scrollIntoView({block: 'start'});
    }
    const params = new URLSearchParams({q: query, top_k: inspect ? '5' : topK});
    if (inspect) {
      params.set('adapt_hindi', 'true');
      params.set('source', 'ipindia');
    } else if (source) params.set('source', source);
    if (domain) params.set('domain', domain);
    let timedOut = false;
    const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, answerMode ? 105000 : 45000);
    const started = performance.now();
    try {
      const response = await fetch(answerMode ? '/answer' : `/search?${params}`, {
        credentials: 'same-origin', cache: 'no-store', signal: controller.signal,
        ...(answerMode ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
          question: query, source: source || 'ipindia', domain: domain || null,
          top_k: 5, acknowledged_nonconfidential: $('demo-consent').checked,
          acknowledged_external_processing: $('demo-consent').checked
        })} : {})
      });
      if (sequence !== state.sequence) return;
      if (!response.ok && !(answerMode && [403, 429, 503].includes(response.status))) throw new Error(`http_${response.status}`);
      const data = await response.json();
      if (sequence !== state.sequence) return;
      if (answerMode) {
        if (!data.status) Object.assign(data, {status: 'unavailable', reason: data.detail});
        renderAnswer(data, performance.now() - started, selection);
        if (data.reason === 'answer_session_required') loadCapabilities();
      } else renderSearch(data, performance.now() - started);
    } catch (error) {
      if (sequence !== state.sequence || (error.name === 'AbortError' && !timedOut)) return;
      $('results-title').textContent = 'Research is temporarily unavailable';
      const rateLimited = error.message === 'http_429';
      const block = message(rateLimited ? 'Please pause before trying again.' : timedOut ? 'The request took too long.' : 'We could not complete this request.', rateLimited ? 'The service is receiving too many requests. Wait a moment and try again.' : 'Your research question is still in the field. Check service status or try again shortly.', true);
      announce(rateLimited ? 'Too many requests. Please wait and try again.' : 'The request failed. Your research question has been preserved.');
      if (answerMode) addEvidenceAction(block, selection);
      if (error.message === 'http_503') checkHealth();
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
    for (const id of ['status-toggle', 'source-detail-button']) $(id)?.setAttribute('aria-expanded', String(open));
    document.querySelectorAll('[data-open-sources]').forEach((button) => button.setAttribute('aria-expanded', String(open)));
    if (open) {
      state.sourceOpener = opener;
      $('source-panel-title').focus();
      $('source-panel').scrollIntoView({block: 'nearest'});
      loadSources();
      if (!state.ready) checkHealth();
    } else if (state.sourceOpener) state.sourceOpener.focus();
  }
  function renderSources(data) {
    const rows = Array.isArray(data.sources) ? data.sources : [];
    const total = rows.reduce((sum, item) => sum + (Number(item.searchable_document_count) || 0), 0);
    const sourceList = document.createDocumentFragment();
    const details = document.createDocumentFragment();
    for (const id of Object.keys(sources)) {
      const item = rows.find((value) => value.source_id === id);
      const count = Number(item?.searchable_document_count) || 0;
      const row = node('div', 'source-row');
      const label = node('span', '', sources[id]);
      if (id === 'tkdl') label.append(node('small', '', 'public pages'));
      row.append(label, node('span', `source-count${count ? ' available' : ''}`, count ? `${formatNumber.format(count)} documents` : 'Not available'));
      sourceList.append(row);
      const entry = node('article', 'source-status-entry');
      entry.append(node('h3', '', sources[id]));
      const dl = node('dl');
      addDefinition(dl, 'Searchable documents', formatNumber.format(count));
      addDefinition(dl, 'Last capture', dateLabel(item?.last_successful_capture || item?.last_successful_sync));
      addDefinition(dl, 'Last publication', dateLabel(item?.last_successful_publication));
      addDefinition(dl, 'Source coverage', count ? 'Archived evidence available' : 'Not admitted to this release');
      addDefinition(dl, 'Evidence role', id === 'tkdl' ? 'Institutional background' : 'Official archived evidence');
      addDefinition(dl, 'Legal status', 'Unknown');
      entry.append(dl);
      details.append(entry);
    }
    $('source-list').replaceChildren(sourceList);
    $('source-details').replaceChildren(details);
    if ($('archive-total')) {
      const ipindia = rows.find((item) => item.source_id === 'ipindia');
      $('archive-total').textContent = data.index_available ? formatNumber.format(Number(ipindia?.searchable_document_count) || 0) : '—';
    }
    $('coverage-note').textContent = data.index_available ? `${formatNumber.format(total)} archived documents. Each result includes its official source and capture details.` : 'The evidence collection is temporarily unavailable. Please check service status.';
    $('source-summary').textContent = 'Explore archived IP India documents. India Code and TKDL are not included in this collection. The archive is not currently refreshing; legal status remains unknown.';
    $('generation-line').textContent = data.active_generation_id ? `Collection identifier: ${data.active_generation_id}` : 'No active collection is available.';
  }
  async function loadSources() {
    try { renderSources(await requestJson('/sources')); }
    catch (_) {
      if ($('archive-total')) $('archive-total').textContent = '—';
      $('coverage-note').textContent = 'Source counts are temporarily unavailable. This release supports archived IP India research only.';
      $('source-summary').textContent = 'Source status could not be loaded. Close and reopen this panel to retry.';
      $('source-list').querySelectorAll('.source-count').forEach((item) => { item.textContent = 'Status unavailable'; });
    }
  }
  function showHealth(data) {
    const wasReady = state.ready;
    state.ready = data?.status === 'ok';
    const preparing = ['not_started', 'initializing'].includes(data?.readiness?.state);
    $('health-label').textContent = state.ready ? 'Evidence search available' : preparing ? 'Preparing evidence search…' : 'Service unavailable';
    $('health-dot').className = `service-dot ${state.ready ? 'ready' : 'offline'}`;
    $('service-message').textContent = state.ready ? 'Search the archive. Expand a passage and inspect its official source.' : preparing ? 'The service is validating the archive before accepting searches. Your question will stay in the field.' : 'Evidence search is temporarily unavailable. Your research question has been kept.';
    $('service-notice').classList.toggle('unavailable', !state.ready);
    $('retry-service').hidden = state.ready;
    updateSubmit();
    if (state.ready && !wasReady) loadSources();
    window.clearTimeout(state.healthTimer);
    if (!state.ready) state.healthTimer = window.setTimeout(checkHealth, preparing ? 3000 : 15000);
  }
  async function checkHealth() {
    if (state.healthPending) return;
    state.healthPending = true;
    $('retry-service').disabled = true;
    try {
      const wasReady = state.ready;
      showHealth(await requestJson('/health', {}, 10000, true));
      if (state.ready && !wasReady) loadCapabilities();
    } catch (_) { showHealth(null); }
    finally { state.healthPending = false; $('retry-service').disabled = false; }
  }
  $('search-form').addEventListener('submit', submit);
  $('question').addEventListener('input', () => { $('question').setCustomValidity(''); $('character-count').textContent = `${$('question').value.length} / 500`; });
  $('question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!state.pending) $('search-form').requestSubmit(); } });
  $('source-filter').addEventListener('change', filterCount);
  $('domain-filter').addEventListener('change', filterCount);
  $('reset-filters').addEventListener('click', () => { $('source-filter').value = ''; $('domain-filter').value = ''; $('top-k').value = '5'; filterCount(); });
  $('suggestions').querySelectorAll('[data-query]').forEach((button) => button.addEventListener('click', () => { $('question').value = button.dataset.query; $('question').dispatchEvent(new Event('input')); $('search-form').requestSubmit(); }));
  for (const id of ['status-toggle', 'source-detail-button']) $(id)?.addEventListener('click', () => toggleSources($('source-panel').hidden, $(id)));
  languageButton?.addEventListener('click', updateSubmit);
  $('close-sources').addEventListener('click', () => toggleSources(false));
  $('source-panel').addEventListener('keydown', (event) => { if (event.key === 'Escape') toggleSources(false); });
  $('retry-service').addEventListener('click', () => { checkHealth(); loadSources(); });
  if ($('edit-search')) $('edit-search').addEventListener('click', () => {
    $('question').scrollIntoView({block: 'center'});
    $('question').focus({preventScroll: true});
  });
  $('research-mode').addEventListener('change', () => { $('demo-consent').checked = false; updateSubmit(); });
  $('demo-consent').addEventListener('change', () => $('demo-consent').setCustomValidity(''));
  async function loadCapabilities() {
    try {
      const capabilities = await requestJson('/capabilities');
      state.demoEnabled = capabilities.demo === true && capabilities.answer_generation_enabled === true;
      $('demo-controls').hidden = !state.demoEnabled;
    } catch (_) {
      state.demoEnabled = false;
      $('demo-controls').hidden = true;
    }
    updateSubmit();
  }
  checkHealth();
})();
