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

  // 流式累加状态：每个节点 (generator / fixer) 维护独立缓冲，
  // fixer 进来时清掉旧 generator 的草稿，避免错位。
  const tokenState = { node: null, buffer: "" };
  const t0 = performance.now();
  let firstTokenAt = null;

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
          if (ev.type === "token" && firstTokenAt === null) {
            firstTokenAt = performance.now();
            $status.textContent = `首 token: ${((firstTokenAt - t0) / 1000).toFixed(2)}s`;
          }
          handleChunk(ev, assistantEl, tokenState);
        } catch (e) {
          console.error("parse error", e, json);
        }
      }
    }
    const total = ((performance.now() - t0) / 1000).toFixed(2);
    const ttft = firstTokenAt ? ((firstTokenAt - t0) / 1000).toFixed(2) : "-";
    $status.textContent = `完成 · 首 token ${ttft}s · 总耗时 ${total}s`;
  } catch (e) {
    assistantEl.textContent = "出错：" + e.message;
    $status.textContent = "出错";
  } finally {
    $sendBtn.disabled = false;
  }
}

// 增量提取 yaml 代码块（即使代码块还没闭合，也把当前已收到的内容渲染出来）
const YAML_FENCE_RE = /```ya?ml\s*\n([\s\S]*?)(?:```|$)/m;

function maybeUpdateYamlFromBuffer(buffer) {
  const m = buffer.match(YAML_FENCE_RE);
  if (m && m[1]) {
    renderYaml(m[1].trimEnd());
  }
}

function handleChunk(ev, assistantEl, tokenState) {
  switch (ev.type) {
    case "plan":
      appendEvent(`[plan] intent=${ev.content} ${ev.meta?.queries ? "queries=" + ev.meta.queries : ""}`);
      break;
    case "retrieval":
      appendEvent(`[retrieval] ${ev.content} ${ev.meta?.sources ? "sources=" + ev.meta.sources : ""}`);
      break;
    case "token": {
      const node = ev.meta?.node || "generator";
      // 切换到新节点（如 fixer），重置缓冲与显示区域
      if (tokenState.node !== node) {
        tokenState.node = node;
        tokenState.buffer = "";
        if (node === "fixer") {
          appendEvent("[fixer] 修正 YAML...");
        }
      }
      tokenState.buffer += ev.content;
      assistantEl.textContent = tokenState.buffer;
      $messages.scrollTop = $messages.scrollHeight;
      maybeUpdateYamlFromBuffer(tokenState.buffer);
      break;
    }
    case "yaml":
      // 后端在节点结束时给出权威 yaml（已格式化），覆盖前端的增量结果
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
      // 状态行已经显示首 token / 总耗时；这里不要覆盖
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
