'use strict';

(() => {
  const sources = {
    ipindia: 'IP India',
    indiacode: 'India Code',
    tkdl: 'TKDL'
  };

  const officialHosts = new Set([
    'ipindia.gov.in',
    'www.ipindia.gov.in',
    'indiacode.gov.in',
    'www.indiacode.gov.in',
    'indiacode.nic.in',
    'www.indiacode.nic.in',
    'upload.indiacode.nic.in',
    'tkdl.res.in',
    'www.tkdl.res.in'
  ]);

  const state = {
    apiBase: (window.location.port === '8000') ? '' : 'http://127.0.0.1:8000',
    serviceReady: false,
    servicePending: false,
    demoEnabled: false,
    activeType: 'question',
    currentResults: null,
    controller: null
  };

  const languageButton = document.querySelector('.language-button');
  let savedLanguage = null;
  try { savedLanguage = localStorage.getItem('ayurveda-language'); } catch { savedLanguage = null; }
  let language = savedLanguage === 'hi' ? 'hi' : 'en';

  function tr(en, hi) {
    return language === 'hi' ? hi : en;
  }

  function setLanguage(next) {
    language = next;
    document.documentElement.lang = next;

    document.querySelectorAll('[data-en][data-hi]').forEach((element) => {
      const value = element.dataset[next];
      if (value.includes('<em>') || value.includes('<br>')) element.innerHTML = value;
      else element.textContent = value;
    });

    document.querySelectorAll('[data-placeholder-hi]').forEach((element) => {
      element.placeholder = next === 'hi'
        ? element.dataset.placeholderHi
        : element.dataset.placeholderEn || element.defaultValue || element.getAttribute('data-original-placeholder') || element.placeholder;
    });

    document.querySelectorAll('[data-aria-hi]').forEach((element) => {
      element.setAttribute('aria-label', next === 'hi' ? element.dataset.ariaHi : element.dataset.ariaEn);
    });

    document.querySelectorAll('[data-alt-hi]').forEach((element) => {
      element.alt = next === 'hi' ? element.dataset.altHi : element.dataset.altEn;
    });

    if (languageButton) {
      const label = languageButton.querySelector('.language-label');
      if (label) label.textContent = next === 'en' ? 'हिन्दी' : 'English';
      languageButton.setAttribute('aria-label', next === 'en' ? 'Switch language to Hindi' : 'Switch language to English');
    }

    document.title = next === 'en'
      ? (document.body.dataset.page === 'home' ? 'AyuSerach — Ayurvedic research with sources in view' : 'Research workspace — AyuSerach')
      : (document.body.dataset.page === 'home' ? 'AyuSerach — स्रोत आधारित शोध' : 'शोध कार्यक्षेत्र — AyuSerach');

    updateServiceBadge();
    if (state.currentResults) {
      renderResults(state.currentResults);
    }

    try { localStorage.setItem('ayurveda-language', next); } catch { /* Ignore local storage errors */ }
  }

  document.querySelectorAll('[data-placeholder-hi]').forEach((element) => {
    element.dataset.originalPlaceholder = element.placeholder;
  });

  if (languageButton) {
    languageButton.addEventListener('click', () => setLanguage(language === 'en' ? 'hi' : 'en'));
  }

  function updateServiceBadge() {
    const pill = document.querySelector('#service-pill');
    const dot = document.querySelector('#service-dot');
    const text = document.querySelector('#service-status');
    if (!pill || !dot || !text) return;

    if (state.serviceReady) {
      dot.className = 'service-dot ready';
      text.textContent = tr('Archive Active (Ready)', 'संग्रह सक्रिय (तैयार)');
    } else {
      dot.className = 'service-dot offline';
      text.textContent = tr('Backend Offline (Preview)', 'बैकएंड ऑफ़लाइन (पूर्वावलोकन)');
    }
  }

  async function checkBackendHealth() {
    try {
      const response = await fetch(`${state.apiBase}/health`, { method: 'GET', cache: 'no-store' });
      if (response.ok) {
        const data = await response.json();
        state.serviceReady = (data.status === 'ok');
      } else {
        state.serviceReady = false;
      }
    } catch {
      state.serviceReady = false;
    }
    updateServiceBadge();

    try {
      const capResponse = await fetch(`${state.apiBase}/capabilities`, { method: 'GET', cache: 'no-store' });
      if (capResponse.ok) {
        const caps = await capResponse.json();
        state.demoEnabled = (caps.demo === true || caps.answer_generation_enabled === true);
        const consentRow = document.querySelector('#demo-consent-row');
        if (consentRow) consentRow.hidden = !state.demoEnabled;
      }
    } catch {
      /* Capabilities check optional */
    }
  }

  // Helper DOM creator
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function formatCitationUrl(item) {
    try {
      const url = new URL(item.source_url);
      if (url.protocol !== 'https:' || !officialHosts.has(url.hostname.toLowerCase())) return null;
      if (/\.pdf$/i.test(url.pathname) && Number.isInteger(item.page_start) && item.page_start > 0) {
        url.hash = `page=${item.page_start}`;
      }
      return url.href;
    } catch {
      return null;
    }
  }

  function anchorLabel(item) {
    const parts = [];
    if (item.section) parts.push(`${tr('Section', 'धारा')} ${item.section}`);
    if (item.rule) parts.push(`${tr('Rule', 'नियम')} ${item.rule}`);
    if (Number.isInteger(item.page_start) && item.page_start > 0) {
      parts.push(item.page_end > item.page_start ? `pp. ${item.page_start}–${item.page_end}` : `p. ${item.page_start}`);
    }
    return parts.length ? parts.join(' · ') : tr('General provision', 'सामान्य प्रावधान');
  }

  function renderHitCard(hit, index) {
    const card = el('article', 'evidence-card');

    const eyebrow = el('div', 'evidence-eyebrow');
    eyebrow.append(
      el('span', 'hit-badge', `#${String(index + 1).padStart(2, '0')}`),
      el('span', 'source-pill', sources[hit.source] || hit.source || 'Official Archive')
    );
    if (hit.legal_domain) {
      eyebrow.append(el('span', 'domain-pill', hit.legal_domain));
    }
    eyebrow.append(el('span', 'status-pill', tr('Status: Unverified', 'स्थिति: असत्यापित')));

    const titleH3 = el('h3', 'evidence-title');
    const officialHref = formatCitationUrl(hit);
    if (officialHref) {
      const link = el('a', 'official-link', hit.title || tr('Untitled official record', 'शीर्षकहीन आधिकारिक दस्तावेज़'));
      link.href = officialHref;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      const mark = el('span', 'ext-arrow', ' ↗');
      mark.setAttribute('aria-hidden', 'true');
      link.append(mark);
      titleH3.append(link);
    } else {
      titleH3.textContent = hit.title || tr('Official document', 'आधिकारिक दस्तावेज़');
    }

    const anchorBar = el('div', 'evidence-anchor', anchorLabel(hit));

    const quoteBlock = el('blockquote', 'evidence-quote collapsed', hit.text);
    const quoteId = `quote-passage-${index}`;
    quoteBlock.id = quoteId;

    const footBar = el('div', 'card-footer');
    const toggleBtn = el('button', 'text-button toggle-quote', tr('Read full passage +', 'पूरा अंश पढ़ें +'));
    toggleBtn.type = 'button';
    toggleBtn.setAttribute('aria-expanded', 'false');
    toggleBtn.setAttribute('aria-controls', quoteId);
    toggleBtn.addEventListener('click', () => {
      const isExpanded = toggleBtn.getAttribute('aria-expanded') === 'true';
      quoteBlock.classList.toggle('collapsed', isExpanded);
      toggleBtn.setAttribute('aria-expanded', String(!isExpanded));
      toggleBtn.textContent = isExpanded
        ? tr('Read full passage +', 'पूरा अंश पढ़ें +')
        : tr('Show less −', 'कम दिखाएँ −');
    });
    footBar.append(toggleBtn);

    if (officialHref) {
      const directDoc = el('a', 'text-button doc-link', tr('Open official source ↗', 'मूल आधिकारिक स्रोत खोलें ↗'));
      directDoc.href = officialHref;
      directDoc.target = '_blank';
      directDoc.rel = 'noopener noreferrer';
      footBar.append(directDoc);
    }

    // Provenance metadata
    const provDetails = el('details', 'provenance-box');
    const provSummary = el('summary', '', tr('Archive & citation provenance', 'संग्रह और उद्धरण स्रोत विवरण'));
    provDetails.append(provSummary);

    const dl = el('dl', 'provenance-list');
    function addDl(dtText, ddText) {
      dl.append(el('dt', '', dtText), el('dd', '', ddText || '—'));
    }
    addDl(tr('Document ID', 'दस्तावेज़ आईडी'), hit.document_id);
    addDl(tr('Passage Chunk ID', 'अंश पहचानकर्ता'), hit.chunk_id);
    addDl(tr('SHA-256 Fingerprint', 'डिजिटल हस्ताक्षर (SHA-256)'), hit.raw_sha256);
    addDl(tr('Captured Date', 'संग्रहीत तिथि'), hit.retrieved_at ? hit.retrieved_at.split('T')[0] : '—');
    addDl(tr('Official URL', 'आधिकारिक वेब पता'), hit.source_url);
    provDetails.append(dl);

    card.append(eyebrow, titleH3, anchorBar, quoteBlock, footBar, provDetails);
    return card;
  }

  function renderResults(data) {
    state.currentResults = data;
    const section = document.querySelector('#evidence-section');
    const container = document.querySelector('#evidence-container');
    const meta = document.querySelector('#results-meta');
    if (!section || !container) return;

    section.hidden = false;
    container.replaceChildren();

    const hits = data.hits || [];
    if (hits.length === 0) {
      meta.textContent = tr('0 passages found', 'कोई संबंधित अंश नहीं मिला');
      const emptyState = el('div', 'empty-results');
      emptyState.append(
        el('h3', '', tr('No matching passages found', 'कोई उपयुक्त साक्ष्य अंश नहीं मिला')),
        el('p', '', tr(
          'Try broadening your research question, or search directly for a section (e.g. "Patents Act section 3").',
          'अपने शोध प्रश्न को व्यापक बनाएं, या सीधे किसी धारा की खोज करें (जैसे "पेटेंट अधिनियम धारा 3")।'
        ))
      );
      container.append(emptyState);
      return;
    }

    meta.textContent = `${hits.length} ${tr('verified passages retrieved', 'सत्यापित अंश प्राप्त हुए')} · ${tr('Source: IP India / Government Archive', 'स्रोत: आईपी इंडिया / सरकारी संग्रह')}`;
    hits.forEach((hit, index) => {
      container.append(renderHitCard(hit, index));
    });

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Setup research form logic
  const researchForm = document.querySelector('#research-form');
  if (researchForm) {
    const typeButtons = [...document.querySelectorAll('.input-type')];
    const panels = [...document.querySelectorAll('.input-panel')];
    const message = document.querySelector('#form-message');
    const linkField = document.querySelector('#source-link');
    const linkError = document.querySelector('#link-error');
    const documentField = document.querySelector('#document');
    const selectedFile = document.querySelector('#file-selected');
    const extractedTermsBox = document.querySelector('#file-extracted-terms');
    const questionField = document.querySelector('#question');
    const charCount = document.querySelector('#char-count');
    const submitBtn = document.querySelector('#submit-button');
    const submitBtnText = document.querySelector('#submit-button-text');
    const clearResultsBtn = document.querySelector('#clear-results-button');

    if (clearResultsBtn) {
      clearResultsBtn.addEventListener('click', () => {
        const section = document.querySelector('#evidence-section');
        if (section) section.hidden = true;
        state.currentResults = null;
      });
    }

    if (questionField && charCount) {
      questionField.addEventListener('input', () => {
        charCount.textContent = `${questionField.value.length} / 500`;
      });
    }

    linkField?.addEventListener('input', () => {
      linkField.removeAttribute('aria-invalid');
      if (linkError) linkError.textContent = '';
    });

    // Client-side document reading & term extraction
    documentField?.addEventListener('change', () => {
      if (!documentField.files.length) {
        if (selectedFile) selectedFile.textContent = '';
        if (extractedTermsBox) extractedTermsBox.hidden = true;
        return;
      }

      const file = documentField.files[0];
      const fileSize = (file.size / 1024).toFixed(1);
      if (selectedFile) {
        selectedFile.textContent = `${tr('Selected file', 'चुनी गई फ़ाइल')}: ${file.name} (${fileSize} KB)`;
      }

      if (file.name.endsWith('.txt')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = String(e.target?.result || '');
          const matches = text.match(/\b(?:section\s+\d+|rule\s+\d+|patent|ayurveda|traditional knowledge|formulation|tkdl|ip\s+india)\b/gi);
          if (matches && matches.length && extractedTermsBox) {
            const unique = [...new Set(matches.map(m => m.trim()))].slice(0, 5);
            extractedTermsBox.replaceChildren();
            extractedTermsBox.append(el('span', 'extracted-label', tr('Key terms found in document:', 'दस्तावेज़ में मिले मुख्य शब्द:')));
            unique.forEach(term => {
              const chip = el('button', 'suggestion-chip', term);
              chip.type = 'button';
              chip.addEventListener('click', () => {
                if (questionField) {
                  questionField.value = term;
                  const qBtn = typeButtons.find(b => b.dataset.type === 'question');
                  qBtn?.click();
                  questionField.focus();
                }
              });
              extractedTermsBox.append(chip);
            });
            extractedTermsBox.hidden = false;
          }
        };
        reader.readAsText(file.slice(0, 10000));
      }
    });

    typeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        state.activeType = button.dataset.type;
        typeButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
        panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === state.activeType));
        message?.classList.remove('show');
        linkField?.removeAttribute('aria-invalid');
        if (linkError) linkError.textContent = '';
        const activeInput = panels.find((p) => p.dataset.panel === state.activeType)?.querySelector('textarea,input');
        activeInput?.focus();
      });
    });

    // Query suggestion chips
    document.querySelectorAll('.suggestion-chip[data-query]').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (questionField) {
          questionField.value = btn.dataset.query;
          const qBtn = typeButtons.find(b => b.dataset.type === 'question');
          qBtn?.click();
          questionField.focus();
          questionField.dispatchEvent(new Event('input'));
        }
      });
    });

    // Submission handler
    researchForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const currentPanel = panels.find((panel) => panel.dataset.panel === state.activeType);
      const field = currentPanel?.querySelector('textarea,input');

      let queryText = '';
      if (state.activeType === 'document') {
        if (!documentField.files.length) {
          showFormMessage(tr('Please select a document to continue.', 'कृपया आगे बढ़ने के लिए एक दस्तावेज़ चुनें।'), true);
          documentField.focus();
          return;
        }
        queryText = documentField.files[0].name.replace(/\.[^/.]+$/, '').replace(/[-_]/g, ' ');
      } else if (state.activeType === 'link') {
        if (!field.value.trim()) {
          showFormMessage(tr('Please enter a link to examine.', 'कृपया जाँच के लिए एक लिंक डालें।'), true);
          field.focus();
          return;
        }
        if (!field.checkValidity()) {
          field.setAttribute('aria-invalid', 'true');
          if (linkError) linkError.textContent = tr('Enter a valid web address, such as https://ipindia.gov.in', 'मान्य वेब पता दर्ज करें, जैसे https://ipindia.gov.in');
          field.focus();
          return;
        }
        queryText = field.value.trim();
      } else {
        queryText = questionField?.value.trim() || '';
        if (queryText.length < 2) {
          showFormMessage(tr('Add a question between 2 and 500 characters.', 'कृपया 2 से 500 अक्षरों के बीच शोध प्रश्न लिखें।'), true);
          questionField?.focus();
          return;
        }
      }

      // Check if backend service is reachable
      const sourceFilter = document.querySelector('#filter-source')?.value || '';
      const domainFilter = document.querySelector('#filter-domain')?.value || '';
      const topK = document.querySelector('#filter-top-k')?.value || '5';
      const consentChecked = document.querySelector('#demo-consent')?.checked || false;

      // Abort previous call
      if (state.controller) state.controller.abort();
      state.controller = new AbortController();

      setPending(true);
      showFormMessage(tr('Searching verified evidence archive...', 'सत्यापित साक्ष्य संग्रह में खोज की जा रही है...'), false);

      try {
        const useAnswerMode = state.demoEnabled && consentChecked;
        const endpoint = useAnswerMode
          ? `${state.apiBase}/answer`
          : `${state.apiBase}/search?q=${encodeURIComponent(queryText)}&top_k=${topK}${sourceFilter ? `&source=${sourceFilter}` : ''}${domainFilter ? `&domain=${encodeURIComponent(domainFilter)}` : ''}`;

        const fetchOptions = {
          signal: state.controller.signal,
          headers: useAnswerMode ? { 'Content-Type': 'application/json' } : {},
          ...(useAnswerMode ? {
            method: 'POST',
            body: JSON.stringify({
              question: queryText,
              source: sourceFilter || 'ipindia',
              domain: domainFilter || null,
              top_k: parseInt(topK, 10),
              acknowledged_nonconfidential: true,
              acknowledged_external_processing: true
            })
          } : { method: 'GET' })
        };

        const response = await fetch(endpoint, fetchOptions);
        if (!response.ok) {
          throw new Error(`http_${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
        showFormMessage(tr('Verified evidence retrieved from archive.', 'संग्रह से सत्यापित साक्ष्य प्राप्त हुए।'), false);
      } catch (err) {
        if (err.name === 'AbortError') return;

        // If backend was unreachable or returned error
        showFormMessage(
          tr(
            'The evidence search service is currently offline or unreachable at ' + state.apiBase + '. You can verify claims directly in the official sources below.',
            'साक्ष्य खोज सेवा वर्तमान में ' + state.apiBase + ' पर ऑफ़लाइन है। आप नीचे दिए गए आधिकारिक स्रोतों में सीधे दावे सत्यापित कर सकते हैं।'
          ),
          true
        );
      } finally {
        setPending(false);
      }
    });

    function setPending(pending) {
      state.servicePending = pending;
      if (submitBtn) submitBtn.disabled = pending;
      if (submitBtnText) {
        submitBtnText.textContent = pending
          ? tr('Searching archive…', 'खोज जारी है…')
          : tr('Start research', 'शोध शुरू करें');
      }
    }

    function showFormMessage(msg, isError) {
      if (!message) return;
      message.textContent = msg;
      message.classList.toggle('error', Boolean(isError));
      message.classList.add('show');
    }
  }

  // Initialize
  setLanguage(language);
  checkBackendHealth();
})();
