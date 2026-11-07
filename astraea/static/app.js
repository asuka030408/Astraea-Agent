const timeline = document.getElementById('timeline');
const form = document.getElementById('composer');
const input = document.getElementById('message');
const send = document.getElementById('send');
let sessionId = null;
let busy = false;
let currentRequest = null;
const history = document.getElementById('history');
const menuToggle = document.getElementById('menuToggle');
const contextDialog = document.getElementById('contextDialog');
const archiveNames = {tool_output: '较早的工具结果', failed_attempt: '失败的工具尝试', summarized_history: '已摘要的早期对话'};

function contextCard(label, content) {
  const card = document.createElement('article');
  card.className = 'contextCard';
  const heading = document.createElement('div');
  heading.className = 'contextCardLabel';
  heading.textContent = label;
  const body = document.createElement('pre');
  body.textContent = content;
  card.append(heading, body);
  return card;
}

async function refreshContext() {
  if (!contextDialog.open || !sessionId) return;
  const selectedId = sessionId;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(selectedId)}/context`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '无法读取上下文');
    if (!contextDialog.open || selectedId !== sessionId) return;
    const percent = (100 * data.tokens_estimate / data.context_window).toFixed(1);
    document.getElementById('contextMeta').textContent = `T${data.tier} · 约 ${data.tokens_estimate} / ${data.context_window} tokens · ${percent}%`;
    const active = document.getElementById('activeMessages');
    active.replaceChildren();
    if (!data.messages.length) active.append(contextCard('当前为空', '发送第一条消息后，这里会显示活动上下文。'));
    for (const [index, message] of data.messages.entries()) {
      const calls = message.tool_calls?.length ? `\n\n工具调用：\n${JSON.stringify(message.tool_calls, null, 2)}` : '';
      active.append(contextCard(`${index + 1}. ${message.role}`, message.content + calls));
    }
    const archives = document.getElementById('archiveMessages');
    archives.replaceChildren();
    if (!data.archives.length) archives.append(contextCard('暂无归档', '当前对话还没有内容被移到外部归档。'));
    for (const [index, fragment] of data.archives.entries()) {
      archives.append(contextCard(`${index + 1}. ${archiveNames[fragment.kind] || fragment.kind}`, fragment.text));
    }
  } catch (error) {
    document.getElementById('contextMeta').textContent = error.message;
  }
}

document.getElementById('openContext').addEventListener('click', () => {
  if (!sessionId) return;
  contextDialog.showModal();
  refreshContext();
});
document.getElementById('closeContext').addEventListener('click', () => contextDialog.close());
setInterval(refreshContext, 3000);

function setSidebarOpen(open) {
  document.body.classList.toggle('sidebarOpen', open);
  menuToggle.setAttribute('aria-expanded', String(open));
  menuToggle.setAttribute('aria-label', open ? '关闭对话历史' : '打开对话历史');
}

menuToggle.addEventListener('click', () => setSidebarOpen(!document.body.classList.contains('sidebarOpen')));
document.getElementById('sidebarBackdrop').addEventListener('click', () => setSidebarOpen(false));
document.addEventListener('keydown', event => { if (event.key === 'Escape') setSidebarOpen(false); });

function welcome() {
  timeline.innerHTML = '<div class="welcome"><span class="star large">✦</span><h2>从一个任务开始</h2><p>可以让 Astraea 搜索、读取或修改工作区文件，并运行测试。</p></div>';
}

async function refreshHistory() {
  const response = await fetch('/api/sessions');
  if (!response.ok) throw new Error('无法读取对话历史');
  const data = await response.json();
  history.replaceChildren();
  for (const item of data.sessions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'historyItem' + (item.session_id === sessionId ? ' active' : '');
    button.textContent = item.title;
    button.title = item.title;
    button.disabled = busy;
    button.addEventListener('click', () => openSession(item.session_id));
    history.append(button);
  }
}

async function openSession(id) {
  if (busy) return;
  const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '无法打开对话');
  sessionId = id;
  localStorage.setItem('astraeaSessionId', id);
  timeline.replaceChildren();
  for (const message of data.messages) entry(message.role === 'human' ? '你' : 'Astraea', message.content, message.role === 'human' ? 'user' : '');
  if (!data.messages.length) welcome();
  document.getElementById('workspace').textContent = `工作区 · ${data.workspace}`;
  document.getElementById('status').textContent = `T${data.context_tier} · 上下文 ${data.context_percent}%`;
  await refreshHistory();
  if (contextDialog.open) refreshContext();
  setSidebarOpen(false);
}

function entry(label, text, kind = '') {
  const box = document.createElement('div');
  box.className = `entry ${kind}`;
  const name = document.createElement('div');
  name.className = 'label';
  name.textContent = label;
  const body = document.createElement('div');
  body.textContent = text;
  box.append(name, body);
  timeline.append(box);
  timeline.scrollTop = timeline.scrollHeight;
  return box;
}

function showEvents(events) {
  if (!events.length) return;
  const details = document.createElement('details');
  details.className = 'events';
  const summary = document.createElement('summary');
  summary.textContent = `${events.length} 条工具记录`;
  details.append(summary);
  for (const event of events) {
    const row = document.createElement('div');
    row.className = 'event';
    const title = document.createElement('strong');
    title.textContent = `${event.kind === 'tool_call' ? '调用' : '结果'} · ${event.name}`;
    const content = document.createElement('pre');
    content.textContent = typeof event.detail === 'string' ? event.detail : JSON.stringify(event.detail, null, 2);
    row.append(title, content);
    details.append(row);
  }
  timeline.append(details);
}

async function startSession() {
  const response = await fetch('/api/session', {method: 'POST'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '无法创建会话');
  sessionId = data.session_id;
  localStorage.setItem('astraeaSessionId', sessionId);
  document.getElementById('workspace').textContent = `工作区 · ${data.workspace}`;
  document.getElementById('status').textContent = 'T0 · 上下文 0%';
  welcome();
  await refreshHistory();
  if (contextDialog.open) refreshContext();
}

document.getElementById('newChat').addEventListener('click', async () => {
  if (busy) return;
  try { await startSession(); setSidebarOpen(false); } catch (error) { entry('错误', error.message); }
});

send.addEventListener('click', () => { if (busy) currentRequest?.abort(); });

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy) return;
  const message = input.value.trim();
  if (!message || !sessionId) return;
  timeline.querySelector('.welcome')?.remove();
  entry('你', message, 'user');
  input.value = '';
  busy = true;
  currentRequest = new AbortController();
  send.type = 'button';
  send.textContent = '停止 ■';
  document.getElementById('newChat').disabled = true;
  history.querySelectorAll('button').forEach(button => button.disabled = true);
  document.getElementById('status').textContent = 'Astraea 正在处理…';
  const answer = entry('Astraea', '');
  const answerBody = answer.lastElementChild;
  const events = [];
  let completed = false;
  let answerSaved = false;
  try {
    const response = await fetch('/api/chat/stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, message}), signal: currentRequest.signal});
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || '请求失败');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      pending += decoder.decode(value, {stream: true});
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.type === 'token') { answerBody.textContent += item.text; timeline.scrollTop = timeline.scrollHeight; }
        if (item.type === 'answer') { answerBody.textContent = item.text; answerSaved = true; }
        if (item.type === 'tool_call' || item.type === 'tool_result')
          events.push({kind: item.type, name: item.name, detail: item.detail});
        if (item.type === 'error') throw new Error(item.message);
        if (item.type === 'done') {
          completed = true;
          document.getElementById('status').textContent = `T${item.context_tier} · 上下文 ${item.context_percent}%${item.memory_saved === false ? ' · Memory 整理失败' : ''}`;
        }
      }
    }
    if (!completed) throw new Error('连接中断，本轮未完成');
    showEvents(events);
    await refreshHistory();
    if (contextDialog.open) refreshContext();
  } catch (error) {
    if (error.name === 'AbortError') {
      answerBody.textContent += answerSaved ? '\n[回答已保存；已停止等待后续整理]' : '\n[已中断，本轮未保存]';
      document.getElementById('status').textContent = answerSaved ? '回答已保存' : '已中断';
    } else {
      entry('错误', error.message);
      document.getElementById('status').textContent = '请求失败';
    }
  } finally {
    busy = false;
    currentRequest = null;
    send.type = 'submit';
    send.textContent = '发送 ↗';
    document.getElementById('newChat').disabled = false;
    history.querySelectorAll('button').forEach(button => button.disabled = false);
    input.focus();
  }
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});

const settingsDialog = document.getElementById('settingsDialog');
const settingsForm = document.getElementById('settingsForm');
const settingsHint = document.getElementById('settingsHint');
let settingsInitial = {};
const settingFields = {
  ASTRAEA_API_KEY: 'cfgApiKey',
  ASTRAEA_BASE_URL: 'cfgBaseUrl',
  ASTRAEA_MODEL: 'cfgModel',
  ASTRAEA_COMPACTOR_MODEL: 'cfgCompactor',
  ASTRAEA_CONTEXT_WINDOW: 'cfgWindow',
  ASTRAEA_WORKSPACE: 'cfgWorkspace',
  ASTRAEA_EMBEDDING_API_KEY: 'cfgEmbedKey',
  ASTRAEA_EMBEDDING_BASE_URL: 'cfgEmbedUrl',
  ASTRAEA_EMBEDDING_MODEL: 'cfgEmbedModel',
  ASTRAEA_EMBEDDING_DIMENSIONS: 'cfgEmbedDims',
};

async function loadSettingsView() {
  const response = await fetch('/api/settings');
  if (!response.ok) throw new Error('无法读取设置');
  const data = await response.json();
  for (const [key, id] of Object.entries(settingFields)) {
    const element = document.getElementById(id);
    if (!element) continue;
    if (element.type === 'password') {
      element.value = '';
      element.placeholder = data.values[key] ? `已保存 ${data.values[key]}（留空不修改）` : '尚未填写';
    } else {
      element.value = data.values[key] || '';
    }
    settingsInitial[key] = element.value;
  }
  document.getElementById('settingsMeta').textContent =
    `对话模型 ${data.configured ? '已配置' : '未配置'} · 记忆向量 ${data.embedding_configured ? '已配置' : '未配置'}`;
  return data;
}

async function saveSettings() {
  const values = {};
  for (const [key, id] of Object.entries(settingFields)) {
    const element = document.getElementById(id);
    const text = element.value.trim();
    if (element.type === 'password') { if (text) values[key] = text; }
    else if (text !== (settingsInitial[key] || '')) values[key] = text;
  }
  const button = document.getElementById('saveSettings');
  button.disabled = true;
  settingsHint.textContent = '正在保存…';
  try {
    const response = await fetch('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({values})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '保存失败');
    await loadSettingsView();
    settingsHint.textContent = data.configured ? '已保存；新建对话即生效' : '已保存，但还需要填写对话模型 API Key';
    if (data.configured && !sessionId) {
      if (settingsDialog.open) settingsDialog.close();
      await startSession();
    }
  } catch (error) {
    settingsHint.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

document.getElementById('openSettings').addEventListener('click', async () => {
  if (!settingsDialog.open) settingsDialog.showModal();
  settingsHint.textContent = '';
  try { await loadSettingsView(); } catch (error) { settingsHint.textContent = error.message; }
});
document.getElementById('closeSettings').addEventListener('click', () => settingsDialog.close());
document.getElementById('saveSettings').addEventListener('click', saveSettings);
settingsForm.addEventListener('submit', event => { event.preventDefault(); saveSettings(); });

(async () => {
  try {
    const config = await loadSettingsView();
    if (!config.configured) {
      document.getElementById('workspace').textContent = '尚未配置 API Key · 请点右上角「设置」填写';
      if (!settingsDialog.open) settingsDialog.showModal();
      settingsHint.textContent = '首次使用：请填写对话模型的 API Key；记忆向量 Key 可选';
      return;
    }
    const previous = localStorage.getItem('astraeaSessionId');
    if (previous) {
      try { await openSession(previous); return; } catch { localStorage.removeItem('astraeaSessionId'); }
    }
    await startSession();
  } catch (error) { entry('错误', error.message); }
})();
