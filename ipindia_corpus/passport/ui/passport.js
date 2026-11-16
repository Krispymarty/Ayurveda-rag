'use strict';

(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId('passport-form');
  const fields = byId('intake-fields');
  const submit = byId('submit-passport');
  const results = byId('passport-results');
  const resultContent = byId('result-content');
  const error = byId('form-error');
  const status = byId('form-status');
  const sessionStatus = byId('session-status');
  const retry = byId('session-retry');
  const description = byId('description');
  let csrfToken = '';
  let serviceReady = false;
  let busy = false;
  let sessionPending = false;
  let sessionModel = null;
  let sessionProvider = null;
  let externalProcessing = false;

  const asText = (value, fallback = 'Not provided') => {
    if (typeof value === 'string' && value.trim()) return value;
    if (typeof value === 'number' && Number.isFinite(value)) return String(value);
    return fallback;
  };
  const arrayOf = (value) => Array.isArray(value) ? value : [];
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const paragraph = (parent, text, className) => {
    const node = element('p', text, className);
    parent.append(node);
    return node;
  };
  const updateCount = () => {
    byId('description-count').textContent = `${description.value.length.toLocaleString('en-IN')} / 1,600 characters`;
    description.setCustomValidity('');
  };
  const showError = (message) => {
    error.textContent = message;
    error.hidden = false;
  };
  const clearError = () => {
    error.textContent = '';
    error.hidden = true;
  };
  const updateControls = () => {
    fields.disabled = busy;
    submit.disabled = busy || !serviceReady;
    retry.disabled = busy || sessionPending;
    form.setAttribute('aria-busy', String(busy));
    results.setAttribute('aria-busy', String(busy));
    sessionStatus.parentElement.setAttribute('aria-busy', String(sessionPending));
    submit.firstElementChild.textContent = busy ? 'Building your evidence trail…' : 'Create Evidence Passport';
  };

  const readinessMessage = (reason) => {
    const messages = {
      search_initializing: 'Research resources are being checked. Wait a moment, then retry the connection. Your form is still here.',
      search_initialization_timeout: 'The readiness check exceeded its deadline and may still be finishing. Retry the connection later. Your form is still here.',
      search_dependency_unavailable: 'The local search service is unavailable. Start or restore it, then retry the connection. Your form is still here.',
      active_generation_changed: 'The searchable archive changed. Verify the expected generation, then restart only the private Passport server. Do not reset or change the corpus. Your form is still here.',
      gemini_key_missing: 'The Gemini API key is not configured on the server. Configure GEMINI_API_KEY locally, then retry the connection. Do not enter an API key in this form.',
      gemini_configuration_invalid: 'The Gemini provider configuration is invalid. Check the server Gemini key, model and provider settings, then retry. Your form is still here.',
      provider_auth_failed: 'Google Gemini rejected the server credentials. Check the server API key and its permissions, then retry. Do not enter credentials in this form.',
      provider_rate_limited: 'Google Gemini has limited these requests. Wait before trying again and check the configured API quota. Your form is still here.',
      provider_model_unavailable: 'The selected Gemini model is unavailable to this API configuration. Check the server model setting and access, then retry.',
      provider_model_changed: 'The provider model version changed. Verify the configured Gemini version before restarting the private Passport service. Your form is still here.',
    };
    return messages[reason] || 'Research resources are not ready. Retry the connection to check again. Your form is still here.';
  };
  const requireReconnect = (message) => {
    csrfToken = '';
    serviceReady = false;
    retry.hidden = false;
    sessionStatus.textContent = message;
    sessionStatus.parentElement.classList.add('unavailable');
  };

  async function requestJSON(path, options = {}, timeoutMs = 180000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        credentials: 'same-origin',
        cache: 'no-store',
        redirect: 'error',
        ...options,
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        let message = 'The research service could not complete this request. Please try again.';
        if (response.status === 403) {
          requireReconnect('Your private session needs to be renewed. Retry the connection before submitting again.');
          message = 'Your session expired or could not be verified. Retry the connection; your form is still here.';
        } else if (response.status === 429) {
          message = 'The research service is busy. Please wait a moment before trying again.';
        } else if (response.status === 400 || response.status === 422) {
          message = 'The service could not accept the intake. Check the required fields and character limits, then try again.';
        } else if (response.status === 503) {
          message = readinessMessage(payload && payload.reason);
          requireReconnect(message);
        }
        throw new Error(message);
      }
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new Error('The service returned an unreadable response. Please try again.');
      }
      return payload;
    } catch (failure) {
      if (failure.name === 'AbortError') throw new Error('The request took too long. Your form is still here; please try again.');
      if (failure instanceof TypeError) throw new Error('Could not connect to the private research service. Check that it is running, then retry the connection.');
      throw failure;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function connectSession() {
    if (sessionPending || busy) return;
    sessionPending = true;
    serviceReady = false;
    csrfToken = '';
    retry.hidden = true;
    sessionStatus.textContent = 'Connecting to the private research service…';
    sessionStatus.parentElement.classList.remove('unavailable');
    updateControls();
    try {
      const session = await requestJSON('/api/session', {}, 10000);
      sessionModel = session.model;
      sessionProvider = session.provider && typeof session.provider === 'object' ? session.provider : null;
      const remote = session.external_processing === true || (sessionProvider && sessionProvider.external_processing === true);
      if (remote !== externalProcessing) byId('external-acknowledgment').checked = false;
      externalProcessing = Boolean(remote);
      byId('external-processing-notice').hidden = !externalProcessing;
      byId('external-acknowledgment-row').hidden = !externalProcessing;
      byId('external-acknowledgment').required = externalProcessing;
      csrfToken = typeof session.csrf_token === 'string' ? session.csrf_token : '';
      serviceReady = session.ready === true && csrfToken.length > 0;
      if (serviceReady) {
        sessionStatus.textContent = externalProcessing ? 'Private session ready with Google Gemini. Review external processing and give consent before submitting.' : 'Private session ready. Submit only non-confidential information.';
      } else {
        sessionStatus.textContent = readinessMessage(session.reason);
        sessionStatus.parentElement.classList.add('unavailable');
        retry.hidden = false;
      }
    } catch (failure) {
      sessionStatus.textContent = failure.message;
      sessionStatus.parentElement.classList.add('unavailable');
      retry.hidden = false;
    } finally {
      sessionPending = false;
      updateControls();
    }
  }

  function officialSourceURL(value) {
    if (typeof value !== 'string') return null;
    try {
      const url = new URL(value);
      if (url.protocol !== 'https:' || !['ipindia.gov.in', 'www.ipindia.gov.in'].includes(url.hostname) ||
          url.username || url.password || (url.port && url.port !== '443')) return null;
      return url.href;
    } catch (_) {
      return null;
    }
  }

  function section(parent, heading) {
    const node = element('section', undefined, 'result-section');
    node.append(element('h3', heading));
    parent.append(node);
    return node;
  }
  function addList(parent, values, emptyMessage, className = 'result-list') {
    const strings = arrayOf(values).filter((value) => typeof value === 'string' && value.trim());
    if (!strings.length) {
      paragraph(parent, emptyMessage);
      return;
    }
    const list = element('ul', undefined, className);
    strings.forEach((value) => list.append(element('li', value)));
    parent.append(list);
  }
  function labeledParagraph(parent, label, value) {
    const node = paragraph(parent, undefined);
    node.append(element('span', label, 'context-label'), document.createTextNode(asText(value)));
  }
  function addMetadata(list, label, value, mono = false) {
    list.append(element('dt', label), element('dd', asText(value), mono ? 'mono' : undefined));
  }

  function renderPassport(data) {
    const fragment = document.createDocumentFragment();
    const notices = element('div', undefined, 'result-notices');
    paragraph(notices, 'Informational evidence only. Legal relevance is unreviewed. This passport makes no determination of patent eligibility, legal compliance, or current law.');
    paragraph(notices, 'An exact quote verifies the cited text only. It does not establish that the source is complete, current, legally controlling, or applicable to your invention.');
    const statuses = {
      ready: 'Evidence available',
      partial: 'Partial evidence',
      blocked: 'Research blocked',
      unavailable: 'Research unavailable',
      needs_clarification: 'Clarification needed',
    };
    const statusLabel = statuses[data.status] || 'Status unconfirmed';
    if (data.status !== 'ready') paragraph(notices, 'Some research remains unresolved. Read the limitations and verification steps before relying on any passage.');
    fragment.append(notices);

    const summary = section(fragment, 'Your invention context');
    paragraph(summary, 'User-provided description; not an independent assessment.');
    paragraph(summary, asText(data.invention_summary), 'invention-summary');
    paragraph(summary, 'Jurisdiction: India · Sources: archived IP India patent passages. Full TKDL records are not covered.');

    const evidenceRecords = arrayOf(data.verified_evidence).filter((record) => record && typeof record === 'object');
    const evidenceNodes = new Map();
    const issuesSection = section(fragment, 'Issues to investigate');
    const issues = arrayOf(data.issues).filter((issue) => issue && typeof issue === 'object');
    if (!issues.length) paragraph(issuesSection, 'No supported issues were returned. This does not mean there are no relevant patent or traditional-knowledge issues.');
    const evidenceSection = section(fragment, 'Exact-quote evidence & provenance');
    if (!evidenceRecords.length) paragraph(evidenceSection, 'No verified evidence was returned. No conclusion can be drawn from this absence.');
    evidenceRecords.forEach((record, index) => {
      const details = element('details', undefined, 'evidence-entry');
      details.id = `passport-evidence-${index + 1}`;
      const evidenceId = asText(record.evidence_id, `Evidence ${index + 1}`);
      if (typeof record.evidence_id === 'string' && !evidenceNodes.has(record.evidence_id)) evidenceNodes.set(record.evidence_id, details);
      const summaryNode = element('summary', `${evidenceId} — ${asText(record.document_title, 'Untitled source')}`);
      details.append(summaryNode);
      const body = element('div', undefined, 'evidence-body');
      paragraph(body, 'Exact source quote', 'quote-label');
      body.append(element('blockquote', asText(record.exact_quote, 'No exact quote supplied. Treat this evidence as incomplete.')));
      const url = officialSourceURL(record.official_url);
      if (url) {
        const link = element('a', 'Open official IP India source ↗', 'evidence-source-link');
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.referrerPolicy = 'no-referrer';
        body.append(link);
      } else {
        paragraph(body, 'Official link unavailable: the supplied URL did not pass the HTTPS IP India source check.', 'source-link-unavailable');
      }
      const metadata = element('dl', undefined, 'provenance-table');
      addMetadata(metadata, 'Document', record.document_title);
      addMetadata(metadata, 'Section', record.section);
      addMetadata(metadata, 'Rule', record.rule);
      const start = asText(record.page_start, 'Unknown');
      const end = asText(record.page_end, 'Unknown');
      addMetadata(metadata, 'PDF page reference', start === end ? start : `${start}–${end}`);
      addMetadata(metadata, 'Source status', 'Unknown — current legal force not verified');
      addMetadata(metadata, 'Legal relevance', 'Unreviewed');
      addMetadata(metadata, 'Evidence ID', record.evidence_id, true);
      addMetadata(metadata, 'Document ID', record.document_id, true);
      addMetadata(metadata, 'Chunk ID', record.chunk_id, true);
      addMetadata(metadata, 'Raw source SHA-256', record.raw_sha256, true);
      addMetadata(metadata, 'Supplied official URL', record.official_url, true);
      body.append(metadata);
      details.append(body);
      evidenceSection.append(details);
    });
    issues.forEach((issue, index) => {
      const card = element('article', undefined, 'issue-card');
      card.append(element('span', `${String(index + 1).padStart(2, '0')} / ${asText(issue.issue_id, 'Issue')}`, 'issue-label'));
      card.append(element('h4', asText(issue.title, 'Research issue')));
      labeledParagraph(card, 'Why it may matter', issue.why_it_may_matter);
      labeledParagraph(card, 'Your context', issue.user_context);
      labeledParagraph(card, 'Evidence status', issue.evidence_status);
      const limitations = Array.isArray(issue.limitations) ? issue.limitations.map((item) => asText(item, '')).filter(Boolean).join('\n') : issue.limitations;
      labeledParagraph(card, 'Limitations', asText(limitations, 'No issue-specific limitation was supplied. Source completeness, current legal force, and legal relevance remain unverified.'));
      const references = element('div', undefined, 'evidence-links');
      references.append(element('span', 'Supporting evidence:'));
      const evidenceIds = arrayOf(issue.evidence_ids).filter((value) => typeof value === 'string');
      if (!evidenceIds.length) references.append(element('span', 'No linked verified evidence.'));
      evidenceIds.forEach((id) => {
        const target = evidenceNodes.get(id);
        if (!target) {
          references.append(element('span', `${id} — referenced evidence was not returned.`));
          return;
        }
        const link = element('a', id, 'evidence-ref');
        link.href = `#${target.id}`;
        link.addEventListener('click', () => {
          target.open = true;
          target.querySelector('summary').focus();
        });
        references.append(link);
      });
      card.append(references);
      issuesSection.append(card);
    });

    const questions = section(fragment, 'Unresolved questions');
    addList(questions, data.unresolved_questions, 'No specific questions were returned. Legal relevance, corpus completeness, and current source status still require verification.');
    const unsupported = section(fragment, 'Unsupported checks remain unresolved');
    paragraph(unsupported, 'This passport does not assess patent eligibility or grant prospects, freedom to operate, regulatory approval, clinical efficacy or safety, TKDL records, access-and-benefit-sharing (ABS) requirements, or whether sources reflect current law.');
    const domains = arrayOf(data.unsupported_domains).filter((domain) => domain && typeof domain === 'object');
    if (domains.length) {
      const list = element('ul', undefined, 'result-list unsupported-list');
      domains.forEach((domain) => {
        const item = element('li');
        item.append(element('strong', asText(domain.domain, 'Unsupported domain')), document.createTextNode(asText(domain.limitation, 'This check was not performed.')));
        list.append(item);
      });
      unsupported.append(list);
    }
    const steps = section(fragment, 'Recommended verification steps');
    addList(steps, data.recommended_verification_steps, 'Review the original official sources and consult a qualified Indian patent professional to verify relevance, completeness, and current status.');
    const metadata = element('div', undefined, 'result-meta');
    paragraph(metadata, `Corpus generation: ${asText(data.generation_id, 'Not supplied')}`);
    paragraph(metadata, 'Overall status: informational evidence only · Legal relevance review: unreviewed');
    const model = data.model || sessionModel;
    if (typeof model === 'string') paragraph(metadata, `Research model: ${model}`);
    else if (model && typeof model === 'object') {
      paragraph(metadata, `Research model: ${asText(model.name || model.model)}`);
      if (typeof model.provider === 'string') paragraph(metadata, `Inference provider: ${model.provider}`);
      const modelVersion = model.model_version || model.version;
      if (typeof modelVersion === 'string') paragraph(metadata, `Provider model version: ${modelVersion}`);
      const modelDigest = model.digest || model.model_digest;
      if (typeof modelDigest === 'string') paragraph(metadata, `Model digest: ${modelDigest}`);
      if (typeof model.planner_version === 'string') paragraph(metadata, `Planner version: ${model.planner_version}`);
      if (Number.isInteger(model.calls_completed) && Number.isInteger(model.calls_attempted)) {
        paragraph(metadata, `Model calls completed / attempted: ${model.calls_completed} / ${model.calls_attempted}`);
      }
    }
    const timings = data.timings_ms;
    if (timings && typeof timings === 'object' && !Array.isArray(timings)) {
      const entries = Object.entries(timings).filter(([, value]) => typeof value === 'number' && Number.isFinite(value));
      if (entries.length) paragraph(metadata, `Service timings (ms): ${entries.map(([key, value]) => `${key}: ${value}`).join(' · ')}`);
    }
    fragment.append(metadata);
    byId('result-status').textContent = statusLabel;
    resultContent.replaceChildren(fragment);
    results.hidden = false;
    byId('results-title').focus();
  }

  description.addEventListener('input', updateCount);
  retry.addEventListener('click', connectSession);
  byId('example-fill').addEventListener('click', () => {
    form.reset();
    description.value = 'I am researching a broad extraction process for Ashwagandha intended to improve consistency between batches. I would like to understand which patent and traditional-knowledge evidence may be relevant to investigating this process.';
    byId('novelty').value = 'new_process';
    byId('goal').value = 'patent_research';
    byId('ingredients').value = 'Ashwagandha (Withania somnifera)';
    byId('traditional-use').value = 'yes';
    byId('additional-context').value = 'Public, illustrative research example only. No recipe, ratio, operating conditions, or confidential technical detail.';
    clearError();
    updateCount();
    resultContent.replaceChildren();
    results.hidden = true;
    status.textContent = 'Non-confidential example filled. Review it and confirm the acknowledgment before submitting.';
    description.focus();
  });
  byId('clear-form').addEventListener('click', () => {
    form.reset();
    updateCount();
    clearError();
    resultContent.replaceChildren();
    results.hidden = true;
    byId('result-status').textContent = '';
    status.textContent = 'Form and results cleared.';
    description.focus();
  });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    clearError();
    if (!serviceReady || !csrfToken) {
      showError('Connect to the private research service before submitting.');
      return;
    }
    const trimmedDescription = description.value.trim();
    if (trimmedDescription.length < 20 || trimmedDescription.length > 1600) {
      description.setCustomValidity('Enter a non-confidential description of 20–1,600 characters, excluding surrounding spaces.');
    }
    if (!form.reportValidity()) return;
    const intake = {
      description: trimmedDescription,
      novelty: byId('novelty').value,
      goal: byId('goal').value,
      jurisdiction: byId('jurisdiction').value,
      ingredients: byId('ingredients').value.trim(),
      traditional_use: byId('traditional-use').value,
      public_disclosure: byId('public-disclosure').value,
      additional_context: byId('additional-context').value.trim(),
      acknowledged_nonconfidential: byId('acknowledgment').checked,
      acknowledged_external_processing: externalProcessing && byId('external-acknowledgment').checked,
    };
    busy = true;
    resultContent.replaceChildren();
    results.hidden = true;
    status.textContent = 'Research in progress. Checking the available evidence and preparing source references…';
    updateControls();
    try {
      const passport = await requestJSON('/api/passport', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Passport-Token': csrfToken },
        body: JSON.stringify(intake),
      });
      renderPassport(passport);
      status.textContent = 'Evidence Passport returned. Review each issue, exact quote, and unresolved check below.';
    } catch (failure) {
      status.textContent = '';
      showError(failure.message || 'The research request could not be completed. Please try again.');
      retry.hidden = false;
    } finally {
      busy = false;
      updateControls();
    }
  });
  updateCount();
  connectSession();
})();
