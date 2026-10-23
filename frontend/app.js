const TEXT_API = "http://127.0.0.1:8001";
const IMAGE_API = "http://127.0.0.1:8002";

const LABELS = { clean: "Safe", review: "Needs review", reject: "Rejected" };

const $ = (selector) => document.querySelector(selector);

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function percent(value) {
  return Math.round(value * 100) + "%";
}

function pretty(name) {
  return name.replace(/_/g, " ");
}

function chip(status) {
  return make("span", "chip " + status, LABELS[status] || "Not checked");
}

function verdict(status, reason) {
  const box = make("div", "verdict " + status);
  box.append(make("div", "verdict-title", LABELS[status] || status), make("div", "verdict-reason", reason));
  return box;
}

function table(headers, rows) {
  const t = make("table");
  const thead = make("thead");
  const head = make("tr");
  headers.forEach((h) => head.append(make("th", "", h)));
  thead.append(head);
  t.append(thead);

  const body = make("tbody");
  rows.forEach((cells) => {
    const tr = make("tr");
    cells.forEach((cell, i) => {
      const td = make("td", i === cells.length - 1 && headers.length === 2 ? "num" : "");
      if (cell instanceof Node) td.append(cell);
      else td.textContent = cell;
      tr.append(td);
    });
    body.append(tr);
  });
  t.append(body);
  return t;
}

function render(nodes) {
  const box = $("#result");
  box.replaceChildren(...nodes);
  box.hidden = false;
}

function showError(message) {
  render([make("div", "error", message)]);
}

function setBusy(busy) {
  $("#btn-text").disabled = busy;
  $("#btn-image").disabled = busy;
}

/* service status */
async function checkService(url, element, name) {
  try {
    const res = await fetch(url + "/health");
    if (!res.ok) throw new Error();
    element.className = "svc online";
    element.textContent = name + ": online";
  } catch (err) {
    element.className = "svc offline";
    element.textContent = name + ": offline";
  }
}
checkService(TEXT_API, $("#status-text"), "Text service");
checkService(IMAGE_API, $("#status-image"), "Image service");

/* tabs */
function openTab(name) {
  ["text", "image"].forEach((t) => {
    $("#tab-" + t).setAttribute("aria-selected", String(t === name));
    $("#panel-" + t).hidden = t !== name;
  });
  $("#result").hidden = true;
}
$("#tab-text").addEventListener("click", () => openTab("text"));
$("#tab-image").addEventListener("click", () => openTab("image"));

/* image preview */
$("#file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("#file-name").textContent = file.name;
  const preview = $("#preview");
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

/* text check */
$("#btn-text").addEventListener("click", async () => {
  const text = $("#text-input").value.trim();
  if (!text) return showError("Enter some text to check.");

  setBusy(true);
  try {
    const res = await fetch(TEXT_API + "/check-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("status " + res.status);
    const data = await res.json();

    render([
      verdict(data.status, "Vulgar probability: " + percent(data.probability)),
    ]);
  } catch (err) {
    showError("Could not reach the text service at " + TEXT_API + ". Make sure it is running.");
  }
  setBusy(false);
});

/* image check */
$("#btn-image").addEventListener("click", async () => {
  const file = $("#file-input").files[0];
  if (!file) return showError("Choose an image first.");

  setBusy(true);
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(IMAGE_API + "/check-image", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "status " + res.status);
    }
    const data = await res.json();

    const img = data.image;
    const textResult = data.ocr.result;
    const textStatus = textResult && textResult.status;

    let reason = "No issue found in the image or the text inside it.";
    if (img.status !== "clean") reason = "Image flagged as " + pretty(img.label) + ".";
    else if (textStatus === "reject" || textStatus === "review") reason = "Text inside the image was flagged.";

    const imageDetail = img.status === "clean"
      ? "Looks safe"
      : pretty(img.label) + " (" + percent(img.unsafe_score) + ")";

    let textCell = make("span", "", "No readable text");
    let textDetail = "";
    if (textResult && textResult.probability !== undefined) {
      textCell = chip(textStatus);
      textDetail = "Vulgar probability " + percent(textResult.probability);
    } else if (textStatus === "unavailable") {
      textCell = chip("unavailable");
      textDetail = "Text service not reachable";
    }

    const scores = Object.entries(img.scores)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => [pretty(name), percent(value)]);

    render([
      verdict(data.status, reason),
      make("h2", "", "Results"),
      table(["Check", "Result", "Detail"], [
        ["Image", chip(img.status), imageDetail],
        ["Text in image", textCell, textDetail],
      ]),
      make("h2", "", "Image class scores"),
      table(["Class", "Score"], scores),
      make("h2", "", "Text found in the image"),
      make("pre", "extracted", data.ocr.text || "Nothing found."),
    ]);
  } catch (err) {
    showError("Could not check the image: " + err.message + ". Make sure the image service at " + IMAGE_API + " is running.");
  }
  setBusy(false);
});