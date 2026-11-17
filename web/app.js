'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
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
    anchor.append(node('span', '', anchorLabel(hit)), node('span', 'legal-status', 'Legal status unknown'));
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
    $('results-title').textContent = data.hits.length ? 'Evidence, with context' : 'No matching passages';
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
  function renderAnswer(data, elapsed) {
    $('results-title').textContent = data.status === 'answered' ? 'Verified research extracts' : 'No supported answer';
    $('results-meta').textContent = `IP India demo · ${Math.round(elapsed)} ms · ${data.provider?.model || 'No model call'}`;
    const fragment = document.createDocumentFragment();
    fragment.append(node('p', 'result-notice', data.notice));
    if (!data.claims?.length) fragment.append(node('p', 'state-block', `No extract was published: ${(data.reason || 'insufficient evidence').replaceAll('_', ' ')}. Try a specific archived provision or switch to evidence search.`));
    (data.claims || []).forEach((claim) => {
      const article = node('article', 'evidence-hit');
      article.append(node('blockquote', 'hit-passage', claim.text));
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
    $('submit-label').textContent = state.pending ? (demoMode() ? 'Selecting extracts…' : 'Searching…') : demoMode() ? 'Ask AyurIP' : 'Find evidence';
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
    if (!state.ready || state.pending) { announce('Evidence search is preparing or unavailable. Your research question has been kept.'); return; }
    const query = $('question').value.trim();
    if (query.length < 2 || query.length > 500) {
      $('question').setCustomValidity('Enter a research question between 2 and 500 characters.');
      $('question').reportValidity();
      return;
    }
    $('question').setCustomValidity('');
    if (demoMode() && !$('demo-consent').checked) {
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
      $('edit-search').hidden = false;
      $('results-title').focus({preventScroll: true});
      $('research-results').scrollIntoView({block: 'start'});
    }
    const params = new URLSearchParams({q: query, top_k: $('top-k').value});
    if ($('source-filter').value) params.set('source', $('source-filter').value);
    if ($('domain-filter').value) params.set('domain', $('domain-filter').value);
    let timedOut = false;
    const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, demoMode() ? 105000 : 45000);
    const started = performance.now();
    try {
      const answerMode = demoMode();
      const response = await fetch(answerMode ? '/answer' : `/search?${params}`, {
        credentials: 'same-origin', cache: 'no-store', signal: controller.signal,
        ...(answerMode ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
          question: query, source: $('source-filter').value || 'ipindia', domain: $('domain-filter').value || null,
          top_k: 5, acknowledged_nonconfidential: $('demo-consent').checked,
          acknowledged_external_processing: $('demo-consent').checked
        })} : {})
      });
      if (sequence !== state.sequence) return;
      if (!response.ok) throw new Error(`http_${response.status}`);
      const data = await response.json();
      if (sequence !== state.sequence) return;
      if (answerMode) renderAnswer(data, performance.now() - started);
      else renderSearch(data, performance.now() - started);
    } catch (error) {
      if (sequence !== state.sequence || (error.name === 'AbortError' && !timedOut)) return;
      $('results-title').textContent = 'Research is temporarily unavailable';
      const rateLimited = error.message === 'http_429';
      message(rateLimited ? 'Please pause before trying again.' : timedOut ? 'The request took too long.' : 'We could not complete this request.', rateLimited ? 'The service is receiving too many requests. Wait a moment and try again.' : 'Your research question is still in the field. Check service status or try again shortly.', true);
      announce(rateLimited ? 'Too many requests. Please wait and try again.' : 'The request failed. Your research question has been preserved.');
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
    for (const id of ['status-toggle', 'source-detail-button']) $(id).setAttribute('aria-expanded', String(open));
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
    $('health-dot').className = `status-dot ${state.ready ? 'ready' : 'error'}`;
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
    try { showHealth(await requestJson('/health', {}, 10000, true)); }
    catch (_) { showHealth(null); }
    finally { state.healthPending = false; $('retry-service').disabled = false; }
  }
  $('search-form').addEventListener('submit', submit);
  $('question').addEventListener('input', () => { $('question').setCustomValidity(''); $('character-count').textContent = `${$('question').value.length} / 500`; });
  $('question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!state.pending) $('search-form').requestSubmit(); } });
  $('filter-toggle').addEventListener('click', () => { const expanded = $('filter-toggle').getAttribute('aria-expanded') !== 'true'; $('filters').hidden = !expanded; $('filter-toggle').setAttribute('aria-expanded', String(expanded)); });
  $('source-filter').addEventListener('change', filterCount);
  $('domain-filter').addEventListener('change', filterCount);
  $('reset-filters').addEventListener('click', () => { $('source-filter').value = ''; $('domain-filter').value = ''; $('top-k').value = '5'; filterCount(); });
  $('suggestions').querySelectorAll('[data-query]').forEach((button) => button.addEventListener('click', () => { $('question').value = button.dataset.query; $('question').dispatchEvent(new Event('input')); $('search-form').requestSubmit(); }));
  for (const id of ['status-toggle', 'source-detail-button']) $(id).addEventListener('click', () => toggleSources($('source-panel').hidden, $(id)));
  document.querySelectorAll('[data-open-sources]').forEach((button) => button.addEventListener('click', () => toggleSources(true, button)));
  document.querySelectorAll('[data-research-query]').forEach((link) => link.addEventListener('click', (event) => {
    event.preventDefault();
    $('question').value = link.dataset.researchQuery;
    $('question').dispatchEvent(new Event('input'));
    $('research').scrollIntoView({block: 'start'});
    $('question').focus({preventScroll: true});
  }));
  const menu = $('nav-toggle');
  function closeMenu(restoreFocus = false) {
    if (!menu) return;
    menu.setAttribute('aria-expanded', 'false');
    $('site-nav').classList.remove('is-open');
    $('nav-sign').textContent = '＋';
    if (restoreFocus) menu.focus();
  }
  if (menu) {
    menu.addEventListener('click', () => {
      const open = menu.getAttribute('aria-expanded') !== 'true';
      menu.setAttribute('aria-expanded', String(open));
      $('site-nav').classList.toggle('is-open', open);
      $('nav-sign').textContent = open ? '−' : '＋';
    });
    $('site-nav').querySelectorAll('a').forEach((link) => link.addEventListener('click', () => closeMenu()));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') closeMenu(true);
    });
    window.matchMedia('(min-width: 901px)').addEventListener('change', () => closeMenu());
  }
  $('close-sources').addEventListener('click', () => toggleSources(false));
  $('source-panel').addEventListener('keydown', (event) => { if (event.key === 'Escape') toggleSources(false); });
  $('retry-service').addEventListener('click', () => { checkHealth(); loadSources(); });
  if ($('edit-search')) $('edit-search').addEventListener('click', () => {
    $('question').scrollIntoView({block: 'center'});
    $('question').focus({preventScroll: true});
  });
  const navLinks = [...document.querySelectorAll('#site-nav a[href^="#"]')];
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const target = entry.target.id === 'research-results' ? '#research' : `#${entry.target.id}`;
        navLinks.forEach((link) => {
          if (link.hash === target) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
    }, {rootMargin: '-100px 0px -55% 0px', threshold: 0});
    navLinks.forEach((link) => { const section = document.querySelector(link.hash); if (section) observer.observe(section); });
    if ($('research-results')) observer.observe($('research-results'));
  }
  $('research-mode').addEventListener('change', () => { $('demo-consent').checked = false; updateSubmit(); });
  $('demo-consent').addEventListener('change', () => $('demo-consent').setCustomValidity(''));
  requestJson('/capabilities').then((capabilities) => {
    state.demoEnabled = capabilities.demo === true;
    $('demo-controls').hidden = !state.demoEnabled;
    updateSubmit();
  }).catch(() => {});
  checkHealth();
  loadSources();
})();
