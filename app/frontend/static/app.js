/* 极简前端：用 fetch 读取 SSE 流，把事件渲染到聊天框；YAML 同步到右侧。 */

const sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", sessionId);

const $messages = document.getElementById("messages");
const $form = document.getElementById("chat-form");
const $input = document.getElementById("input");
const $sendBtn = document.getElementById("send-btn");
const $status = document.getElementById("status");
const $yamlOut = document.getElementById("yaml-out");
const $copyBtn = document.getElementById("copy-btn");
const $meta = document.getElementById("meta");

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = "message " + role;
  el.textContent = text;
  $messages.appendChild(el);
  $messages.scrollTop = $messages.scrollHeight;
  return el;
}

function appendEvent(text) {
  const el = document.createElement("div");
  el.className = "message event";
  el.textContent = "· " + text;
  $messages.appendChild(el);
  $messages.scrollTop = $messages.scrollHeight;
}

function renderYaml(text) {
  $yamlOut.textContent = text;
  if (window.hljs) hljs.highlightElement($yamlOut);
}

function appendMeta(label, content) {
  const div = document.createElement("div");
  const span = document.createElement("span");
  span.className = "label";
  span.textContent = label + ":";
  div.appendChild(span);
  div.appendChild(document.createTextNode(content));
  $meta.appendChild(div);
}

$copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($yamlOut.textContent);
    $copyBtn.textContent = "已复制";
    setTimeout(() => ($copyBtn.textContent = "复制"), 1200);
  } catch (e) {
    $copyBtn.textContent = "复制失败";
  }
});

async function sendMessage(message) {
  $sendBtn.disabled = true;
  $status.textContent = "请求中...";
  $meta.innerHTML = "";
  appendMessage("user", message);
  const assistantEl = appendMessage("assistant", "");

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, stream: true }),
    });

    if (!resp.ok || !resp.body) {
      assistantEl.textContent = "请求失败：HTTP " + resp.status;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n\n");
      buf = lines.pop() || "";
      for (const block of lines) {
        if (!block.startsWith("data:")) continue;
        const json = block.slice(5).trim();
        if (!json) continue;
        try {
          const ev = JSON.parse(json);
          handleChunk(ev, assistantEl);
        } catch (e) {
          console.error("parse error", e, json);
        }
      }
    }
    $status.textContent = "完成";
  } catch (e) {
    assistantEl.textContent = "出错：" + e.message;
    $status.textContent = "出错";
  } finally {
    $sendBtn.disabled = false;
  }
}

function handleChunk(ev, assistantEl) {
  switch (ev.type) {
    case "plan":
      appendEvent(`[plan] intent=${ev.content} ${ev.meta?.queries ? "queries=" + ev.meta.queries : ""}`);
      break;
    case "retrieval":
      appendEvent(`[retrieval] ${ev.content} ${ev.meta?.sources ? "sources=" + ev.meta.sources : ""}`);
      break;
    case "token":
      assistantEl.textContent = ev.content;
      break;
    case "yaml":
      renderYaml(ev.content);
      break;
    case "validation":
      appendEvent(`[validation] ${ev.content}`);
      if (ev.meta?.warnings) appendMeta("警告", ev.meta.warnings);
      if (ev.meta?.errors) appendMeta("错误", ev.meta.errors);
      break;
    case "error":
      appendEvent(`[error] ${ev.content}`);
      break;
    case "done":
      $status.textContent = "已完成";
      break;
  }
}

$form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = $input.value.trim();
  if (!message) return;
  $input.value = "";
  sendMessage(message);
});

$input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    $form.requestSubmit();
  }
});
