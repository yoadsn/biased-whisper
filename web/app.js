const OPTIONS = {
  objective: ['mass', 'pair', 'mass_vs_card'],
  nudge_mode: ['linear', 'iterative'],
  param_filter: ['bitfit', 'decoder_no_embed', 'decoder', 'encoder', 'all_no_embed', 'all'],
  normalize: ['rms', 'none'],
  stat: ['mean_all', 'mean_present', 'median_present'],
};
const HELP = {
  limit: 'how many corpus samples to process',
  mistake_prob: 'target prob = 1 - this (the nudge goal)',
  sample_pct: 'upstream params sample pct: top-% of each sample\'s nudge kept',
  common_pct: 'Common upstream params percent of the union (by count)',
  agree_top_pct: 'top agreement params percent per voting round',
  group_size: 'vectors summed per voting round',
  rounds: 'random voting rounds (also used for the null)',
  fdr: 'empirical false-discovery rate for the vote threshold',
  param_filter: 'which parameters are knobs (bitfit = biases+LayerNorm)',
  objective: 'mass: digit-token mass vs rest; pair: digit tok vs cardinal tok',
  nudge_mode: 'linear: min-norm 1st-order; iterative: damped Gauss-Newton',
  strengths: 'comma-separated multipliers of the mean nudge',
  controls: 'shuffled_values and/or random_indices',
  stat: 'which per-candidate statistic to apply',
  skip_satisfied: 'only compute a nudge from samples that need it',
};

const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function app() {
  return {
    tabs: ['overview', 'corpus', 'phase1', 'phase2', 'phase3', 'jobs', 'knowledge'],
    tab: location.hash.slice(1) || 'overview',
    settings: {}, device: '', jobs: [], jobTypes: {}, cfgs: {}, OPTIONS, HELP,
    toast: '',
    // corpus
    corpus: { items: [], total: 0, stats: { sources: [], total: 0 }, q: '', source: '', digitsOnly: false, offset: 0, limit: 30, sel: null, hf: [], explore: null, curlSearch: '', curlAudio: '', probe: null, order: 'id' },
    // phase 1
    p1: { run: '', rows: [], total: 0, verdicts: {}, filter: '', sel: null, usable: null },
    p2: { run: '', chosen: [], summary: null, type: 'to_cardinal', cands: [], candTotal: 0, candOffset: 0, p1runs: [], verdicts: ['cardinal', 'human_good'], usable: null },
    p3: { run: '', table: [], outputs: {}, sample: null },
    logJob: '', logText: '', logOff: 0,
    kn: { list: [], doc: null },

    async init() {
      window.addEventListener('hashchange', () => (this.tab = location.hash.slice(1) || 'overview'));
      this.$watch('tab', t => { location.hash = t; this.onTab(t); });
      await Promise.all([this.loadSettings(), this.loadTypes()]);
      await this.pollJobs();
      this.onTab(this.tab);
      setInterval(() => this.pollJobs(), 1500);
    },
    async api(path, method = 'GET', body, quiet = false) {
      const r = await fetch(path, { method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
      if (!r.ok) { const t = await r.text(); if (!quiet) this.say(`${r.status}: ${t.slice(0, 200)}`); throw new Error(t); }
      return r.json();
    },
    say(m) { this.toast = m; setTimeout(() => (this.toast = ''), 5000); },

    // ---------------- settings & jobs
    async loadSettings() { const r = await this.api('/api/settings'); this.settings = r.settings; this.device = r.resolved_device; },
    async saveSettings() { this.settings = await this.api('/api/settings', 'POST', this.settings); await this.loadSettings(); this.say('saved'); },
    async loadTypes() {
      this.jobTypes = await this.api('/api/job-types');
      for (const [k, v] of Object.entries(this.jobTypes)) this.cfgs[k] = JSON.parse(JSON.stringify(v.defaults));
    },
    async pollJobs() {
      const prev = this.jobs;
      this.jobs = await this.api('/api/jobs');
      const active = this.jobs.filter(j => ['running', 'queued'].includes(j.status));
      // live refresh of the selected run's results
      for (const [sel, type, fn] of [[this.p1.run, 'phase1', () => this.loadP1()], [this.p2.run, 'phase2', () => this.loadP2()], [this.p3.run, 'phase3', () => this.loadP3()]]) {
        const j = this.jobs.find(x => x.id === sel);
        const was = prev.find(x => x.id === sel);
        if (j && (['running'].includes(j.status) || (was && was.status !== j.status))) fn();
      }
      if (this.logJob) this.pollLog();
      return active;
    },
    byType(t) { return this.jobs.filter(j => j.type === t); },
    active() { return this.jobs.filter(j => ['running', 'queued'].includes(j.status)); },
    job(id) { return this.jobs.find(j => j.id === id); },
    pct(j) { return j && j.progress.total ? Math.round(100 * j.progress.done / j.progress.total) : 0; },
    async launch(type, extra = {}) {
      const j = await this.api('/api/jobs', 'POST', { type, config: { ...this.parseCfg(type), ...extra } });
      this.say(`started ${j.id}`);
      await this.pollJobs();
      return j;
    },
    async cancel(id) { await this.api(`/api/jobs/${id}/cancel`, 'POST'); },
    watchLog(id) { this.logJob = id; this.logOff = 0; this.logText = ''; this.pollLog(); },
    async pollLog() {
      const r = await this.api(`/api/jobs/${this.logJob}/log?offset=${this.logOff}`);
      if (r.text) { this.logText += r.text; this.logOff = r.offset; this.$nextTick(() => { const e = document.getElementById('logbox'); if (e) e.scrollTop = e.scrollHeight; }); }
    },

    // ---------------- config forms
    fields(type) { return Object.keys(this.cfgs[type] || {}).filter(k => !this.hidden(type).includes(k)); },
    hidden(type) { return { phase1: ['sample_ids'], phase2: ['phase1_runs', 'verdicts'], phase3: ['phase2_run'] }[type] || []; },
    kind(type, k) {
      const d = this.jobTypes[type].defaults[k], v = this.cfgs[type][k];
      if (OPTIONS[k]) return 'select';
      if (typeof d === 'boolean') return 'bool';
      if (Array.isArray(d)) return 'array';
      if (typeof d === 'number') return 'number';
      return 'text';
    },
    arrText(type, k) { const v = this.cfgs[type][k]; return Array.isArray(v) ? v.join(', ') : (v ?? ''); },
    setArr(type, k, s) { this.cfgs[type][k] = s.split(',').map(x => x.trim()).filter(Boolean).map(x => (isNaN(+x) ? x : +x)); },
    setText(type, k, s) {
      s = s.trim();
      if (s === '') this.cfgs[type][k] = null;
      else if (/^-?\d+$/.test(s) && this.jobTypes[type].defaults[k] === null && !/model|regex|dataset/.test(k)) this.cfgs[type][k] = +s;
      else this.cfgs[type][k] = s;
    },
    parseCfg(type) { return JSON.parse(JSON.stringify(this.cfgs[type])); },

    // ---------------- navigation
    onTab(t) {
      if (t === 'corpus') this.loadCorpus(true);
      if (t === 'phase1') { this.loadUsable(); if (!this.p1.run) this.pickLatest('phase1'); }
      if (t === 'phase2') { this.initP2(); if (!this.p2.run) this.pickLatest('phase2'); }
      if (t === 'phase3') { if (!this.p3.run) this.pickLatest('phase3'); }
      if (t === 'knowledge') this.loadKnowledge();
    },
    pickLatest(type) {
      const j = this.byType(type)[0];
      if (!j) return;
      if (type === 'phase1') this.selectP1(j.id);
      if (type === 'phase2') this.selectP2(j.id);
      if (type === 'phase3') this.selectP3(j.id);
    },

    // ---------------- corpus
    async loadCorpus(reset = false) {
      if (reset) this.corpus.offset = 0;
      const c = this.corpus;
      const qs = new URLSearchParams({ q: c.q, source: c.source, digits_only: c.digitsOnly, limit: c.limit, offset: c.offset, order: c.order });
      const r = await this.api('/api/corpus?' + qs);
      c.items = r.items; c.total = r.total;
      c.stats = await this.api('/api/corpus/stats');
      c.hf = await this.api('/api/corpus/hf-datasets');
      c.explore = await this.api('/api/explore/config');
    },
    async openSample(id) { this.corpus.sel = await this.api(`/api/corpus/${id}`); },
    async saveSampleText() { const s = this.corpus.sel; await this.api(`/api/corpus/${s.id}`, 'PUT', { text: s.text }); this.say('text saved'); this.loadCorpus(); },
    async deleteSample(id) { if (!confirm('Delete sample ' + id + '?')) return; await this.api('/api/corpus', 'DELETE', [id]); this.corpus.sel = null; this.loadCorpus(); },
    async importCurl() {
      this.corpus.explore = await this.api('/api/explore/curl', 'POST', { search_curl: this.corpus.curlSearch, audio_curl: this.corpus.curlAudio });
      this.corpus.curlSearch = this.corpus.curlAudio = ''; this.say('explore config saved');
    },
    async saveExplore() {
      const e = this.corpus.explore;
      this.corpus.explore = await this.api('/api/explore/config', 'POST', { search_url: e.search_url, audio_url: e.audio_url, mapping: e.mapping, page_size: +e.page_size, query: e.query, mode: e.mode });
      this.say('saved');
    },
    async exploreCheck() { const r = await this.api('/api/explore/check'); this.say(r.ok ? 'explore auth OK' : 'explore: ' + r.reason); },
    async exploreProbe() { this.corpus.probe = await this.api('/api/explore/probe'); },

    // ---------------- phase 1
    async loadUsable() {
      this.p1.runs = this.byType('phase1');
    },
    selectP1(id) { this.p1.run = id; this.p1.sel = null; this.loadP1(); this.watchLog(id); },
    async loadP1() {
      if (!this.p1.run) return;
      const r = await this.api(`/api/jobs/${this.p1.run}/rows?limit=2000`);
      this.p1.rows = r.items; this.p1.total = r.total;
      this.p1.verdicts = await this.api(`/api/phase1/${this.p1.run}/verdicts`);
      if (this.p1.sel) this.p1.sel = this.p1.rows.find(x => x.sample_id === this.p1.sel.sample_id) || null;
    },
    p1rows() { const f = this.p1.filter; return this.p1.rows.filter(r => !f || this.eff(r) === f || r.verdict === f); },
    eff(r) { const h = this.p1.verdicts[r.sample_id]; return h ? 'human_' + h : r.verdict; },
    p1counts() { const c = {}; for (const r of this.p1.rows) c[this.eff(r)] = (c[this.eff(r)] || 0) + 1; return Object.entries(c); },
    async setVerdict(r, v) {
      await this.api(`/api/phase1/${this.p1.run}/verdict`, 'POST', { sample_id: r.sample_id, verdict: v });
      this.p1.verdicts = await this.api(`/api/phase1/${this.p1.run}/verdicts`);
    },
    hlDigits(t) { return esc(t).replace(/(\S*\d\S*)/g, '<mark class="dig">$1</mark>'); },
    hlWords(t, spans) {
      const set = new Set((spans || []).flatMap(s => s.number_words || []));
      return esc(t).split(/(\s+)/).map(w => (set.has(w) ? `<mark class="word">${w}</mark>` : w)).join('');
    },
    pct1(x) { return x == null ? '—' : (100 * x).toFixed(1) + '%'; },
    f(x, d = 3) { return x == null ? '—' : Number(x).toFixed(d); },

    // ---------------- phase 2
    initP2() {
      this.p2.p1runs = this.byType('phase1').filter(j => j.status === 'done');
      if (!this.p2.sel_init) { this.p2.sel_init = true; const l = this.p2.p1runs[0]; if (l) this.p2.chosen = [l.id]; }
      this.p2.chosen = this.p2.chosen || [];
      this.updateUsable();
    },
    toggleP1(id) { const c = this.p2.chosen; const i = c.indexOf(id); i < 0 ? c.push(id) : c.splice(i, 1); this.updateUsable(); },
    async updateUsable() {
      if (!this.p2.chosen?.length) { this.p2.usable = 0; return; }
      const r = await this.api(`/api/phase1/usable?runs=${this.p2.chosen.join(',')}&verdicts=${this.p2.verdicts.join(',')}`);
      this.p2.usable = r.count;
    },
    toggleVerdict(v) { const a = this.p2.verdicts; const i = a.indexOf(v); i < 0 ? a.push(v) : a.splice(i, 1); this.updateUsable(); },
    launchP2() { return this.launch('phase2', { phase1_runs: this.p2.chosen, verdicts: this.p2.verdicts }); },
    selectP2(id) { this.p2.run = id; this.loadP2(); this.watchLog(id); },
    async loadP2() {
      const id = this.p2.run; if (!id) return;
      try { this.p2.summary = await this.api(`/api/jobs/${id}/file/summary.json`, 'GET', null, true); } catch { this.p2.summary = null; }
      this.p2.cands = []; this.p2.candOffset = 0;
      if (this.p2.summary) this.loadCands();
    },
    async loadCands() {
      const t = this.p2.type;
      try {
        const r = await this.api(`/api/phase2/${this.p2.run}/candidates?type=${t}&offset=${this.p2.candOffset}&limit=50`, 'GET', null, true);
        this.p2.cands = r.items; this.p2.candTotal = r.total;
      } catch { this.p2.cands = []; this.p2.candTotal = 0; }
    },
    ty() { return this.p2.summary?.types?.[this.p2.type]; },
    histBars(h, h2) {
      if (!h) return [];
      const m = Math.max(1, ...h, ...(h2 || []));
      return h.map((v, i) => ({ x: i * 14, h: Math.round(80 * Math.log1p(v) / Math.log1p(m)), h2: h2 ? Math.round(80 * Math.log1p(h2[i]) / Math.log1p(m)) : 0 }));
    },
    histSvg(t) {
      if (!t || !t.hist_real) return '';
      const m = Math.max(1, ...t.hist_real, ...t.hist_null), h = v => Math.round(80 * Math.log1p(v) / Math.log1p(m));
      const vmax = t.hist_edges[t.hist_edges.length - 1];
      let g = t.hist_real.map((v, i) => `<rect x="${i * 14}" y="${90 - h(v)}" width="12" height="${h(v)}" fill="var(--c1)" opacity=".75"/><rect x="${i * 14 + 2}" y="${90 - h(t.hist_null[i])}" width="8" height="${h(t.hist_null[i])}" fill="var(--c2)" opacity=".85"/>`).join('');
      if (t.threshold_votes) { const x = t.threshold_votes / vmax * 560; g += `<line x1="${x}" x2="${x}" y1="0" y2="92" stroke="var(--bad)" stroke-dasharray="3"/>`; }
      return `<svg width="570" height="100" viewBox="0 0 570 100">${g}<text x="0" y="99">0</text><text x="520" y="99">${Math.round(vmax)}</text></svg>`;
    },
    barList(obj, n = 14) {
      const e = Object.entries(obj || {}).slice(0, n); const m = Math.max(1e-30, ...e.map(x => x[1]));
      return e.map(([k, v]) => ({ k, v, w: 100 * v / m }));
    },

    // ---------------- phase 3
    selectP3(id) { this.p3.run = id; this.loadP3(); this.watchLog(id); },
    async loadP3() {
      const id = this.p3.run; if (!id) return;
      try { this.p3.table = await this.api(`/api/jobs/${id}/file/table.json`, 'GET', null, true); } catch { this.p3.table = []; }
      const r = await this.api(`/api/jobs/${id}/rows?file=outputs.jsonl&limit=100000`);
      const out = {};
      for (const o of r.items) (out[o.sample_id] ||= {})[o.cond + (o.strength ? ' x' + o.strength : '')] = o.text;
      this.p3.outputs = out;
      if (!this.p3.sample) this.p3.sample = Object.keys(out)[0] || null;
    },
    p3base() { return this.p3.table.find(r => r.cond === 'baseline'); },
    async launchP3() { return this.launch('phase3', { phase2_run: this.p2.run || this.byType('phase2').find(j => j.status === 'done')?.id }); },

    // ---------------- knowledge
    async loadKnowledge() { this.kn.list = await this.api('/api/knowledge'); if (!this.kn.doc && this.kn.list.length) this.openDoc(this.kn.list.find(d => d.path === 'index.md')?.path || this.kn.list[0].path); },
    async openDoc(p) { this.kn.doc = await this.api('/api/knowledge/doc?path=' + encodeURIComponent(p)); },
    md(t) {
      // tiny markdown renderer: headings, code fences, lists, tables, bold/code/links
      const lines = (t || '').split('\n'); let out = '', inCode = false, list = false, tbl = false;
      const inl = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, a, h) => /^https?:/.test(h) ? `<a href="${h}" target="_blank">${a}</a>` : `<a href="#knowledge" onclick="event.preventDefault();document.dispatchEvent(new CustomEvent('kn-open',{detail:'${h.replace(/^\//, '').replace(/^\.\//, '')}'}))">${a}</a>`);
      const closeList = () => { if (list) { out += '</ul>'; list = false; } if (tbl) { out += '</table>'; tbl = false; } };
      for (const l of lines) {
        if (l.startsWith('```')) { closeList(); out += inCode ? '</pre>' : '<pre>'; inCode = !inCode; continue; }
        if (inCode) { out += esc(l) + '\n'; continue; }
        let m;
        if ((m = l.match(/^(#{1,4})\s+(.*)/))) { closeList(); out += `<h${m[1].length + 1}>${inl(m[2])}</h${m[1].length + 1}>`; }
        else if (/^\s*[-*]\s+/.test(l)) { if (!list) { closeList(); out += '<ul>'; list = true; } out += `<li>${inl(l.replace(/^\s*[-*]\s+/, ''))}</li>`; }
        else if (/^\|.*\|$/.test(l.trim())) { if (/^\|[\s:|-]+\|$/.test(l.trim())) continue; if (!tbl) { closeList(); out += '<table>'; tbl = true; } out += '<tr>' + l.trim().slice(1, -1).split('|').map(c => `<td>${inl(c.trim())}</td>`).join('') + '</tr>'; }
        else if (l.trim() === '') { closeList(); }
        else { closeList(); out += `<p>${inl(l)}</p>`; }
      }
      closeList(); return out;
    },
    fmtTime(t) { return t ? new Date(t * 1000).toLocaleTimeString() : ''; },
    dur(j) { return j.started ? Math.round((j.finished || Date.now() / 1000) - j.started) + 's' : ''; },
  };
}
document.addEventListener('alpine:init', () => {});
document.addEventListener('kn-open', e => { const root = document.querySelector('[x-data]')._x_dataStack[0]; root.openDoc(e.detail); });
