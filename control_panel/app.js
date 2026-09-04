// Control panel — Phase 4. Lets the researcher pick which single modality is
// active on the mirror, live, via POST /api/config (no restart, and every
// change is logged server-side with a timestamp — see config_manager.py).
// Exactly one of "proxemic" / "gesture" / "object" is active at a time —
// enforced by the backend's active_modality field, not just this radio UI.

const SELECTABLE_MODALITIES = ["proxemic", "gesture", "object"];

let config = null;
const statusEl = document.getElementById("status");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function syncSelection() {
  const active = config.active_modality;
  for (const modality of SELECTABLE_MODALITIES) {
    document.getElementById(`select-${modality}`).checked = modality === active;
  }
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    config = await res.json();
    syncSelection();
    setStatus("Connected");
  } catch (err) {
    setStatus(`Could not reach backend: ${err.message}`, true);
  }
}

async function selectModality(modality) {
  const previous = config.active_modality;
  config.active_modality = modality;
  setStatus("Saving…");
  try {
    const note = `Switched active modality to ${modality} via control panel`;
    const res = await fetch(`/api/config?note=${encodeURIComponent(note)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const data = await res.json();
    config = data.config;
    syncSelection();
    setStatus("Saved");
  } catch (err) {
    config.active_modality = previous; // revert the optimistic selection
    syncSelection();
    setStatus(`Failed to save: ${err.message}`, true);
  }
}

// Keeps this panel in sync if the config changes elsewhere — another panel
// instance, or content_config.json edited directly — reusing the same
// config_update broadcast the mirror frontend already listens for.
function connectSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "config_update") {
      config = msg.config;
      syncSelection();
    }
  };
  ws.onclose = () => setTimeout(connectSocket, 1000);
}

async function init() {
  await loadConfig();
  for (const modality of SELECTABLE_MODALITIES) {
    document.getElementById(`select-${modality}`).addEventListener("change", (e) => {
      if (e.target.checked) selectModality(modality);
    });
  }
  connectSocket();
}

init();
