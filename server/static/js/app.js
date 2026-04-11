(function () {
  const $ = (id) => document.getElementById(id);

  /**
   * Hugging Face Spaces (and other reverse proxies) serve the app under a path like
   * /spaces/user/repo — fetch("/step") would hit the site root and fail. Strip the /ui
   * segment from the current path to get the API base.
   */
  function apiBase() {
    const p = (window.location.pathname || "/").replace(/\/+$/, "") || "/";
    if (p === "/ui" || p.endsWith("/ui")) {
      const b = p.slice(0, -3);
      return b === "" ? "" : b;
    }
    const m = p.match(/^(.*)\/ui(\/|$)/);
    return m ? m[1] : "";
  }

  function apiUrl(path) {
    const base = apiBase();
    const suffix = path.startsWith("/") ? path : "/" + path;
    return base ? base + suffix : suffix;
  }

  const productInput = $("product-name");
  const ingredientsInput = $("ingredients");
  const taskSelect = $("task-id");
  const fileInput = $("file-input");
  const btnOcr = $("btn-ocr");
  const btnCheck = $("btn-check");
  const ocrStatus = $("ocr-status");
  const formStatus = $("form-status");
  const results = $("results");
  const verdictEl = $("verdict");
  const flaggedBlock = $("flagged-block");
  const flaggedList = $("flagged-list");
  const aiBlock = $("ai-block");
  const aiText = $("ai-text");
  const metaEl = $("meta");

  function parseIngredients(raw) {
    return raw
      .split(/[\n,;•·|]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function setOcrStatus(msg, isError) {
    ocrStatus.textContent = msg || "";
    ocrStatus.style.color = isError ? "var(--danger)" : "var(--warn)";
  }

  function setFormStatus(msg, isError) {
    formStatus.textContent = msg || "";
    formStatus.style.color = isError ? "var(--danger)" : "var(--warn)";
  }

  async function runOcr(file) {
    if (typeof Tesseract === "undefined") {
      setOcrStatus("OCR library failed to load. Check your network.", true);
      return;
    }
    btnOcr.disabled = true;
    setOcrStatus("Running OCR (first run may download language data)…");

    try {
      const {
        data: { text },
      } = await Tesseract.recognize(file, "eng", {
        logger: (m) => {
          if (m.status === "recognizing text") {
            setOcrStatus(`OCR… ${Math.round(m.progress * 100)}%`);
          }
        },
      });

      const cleaned = text.replace(/\r/g, "\n").trim();
      if (!cleaned) {
        setOcrStatus("No text found in image. Try a clearer photo.", true);
        return;
      }

      const current = ingredientsInput.value.trim();
      ingredientsInput.value = current
        ? current + "\n" + cleaned
        : cleaned;
      setOcrStatus("Text added to ingredients. Edit if needed.");
    } catch (e) {
      setOcrStatus(e.message || "OCR failed.", true);
    } finally {
      btnOcr.disabled = false;
    }
  }

  btnOcr.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    fileInput.value = "";
    if (f) runOcr(f);
  });

  function showResults(data) {
    const obs = data.observation || {};
    const verdict = (obs.verdict || "").toUpperCase();
    const dangerous = verdict === "DANGEROUS";

    results.classList.add("visible");
    verdictEl.textContent = dangerous ? "DANGEROUS" : "SAFE";
    verdictEl.className = "verdict " + (dangerous ? "dangerous" : "safe");

    const flagged = obs.flagged_ingredients || [];
    if (flagged.length) {
      flaggedBlock.style.display = "block";
      flaggedList.innerHTML = "";
      flagged.forEach((line) => {
        const li = document.createElement("li");
        li.textContent = line;
        flaggedList.appendChild(li);
      });
    } else {
      flaggedBlock.style.display = "none";
    }

    const ai = obs.ai_analysis || "";
    aiText.textContent = ai;
    aiBlock.style.display = ai ? "block" : "none";

    const info = data.info || {};
    metaEl.textContent = [
      info.task_id ? `Task: ${info.task_id}` : "",
      typeof data.reward === "number" ? `Signal: ${data.reward.toFixed(4)}` : "",
      info.flagged_count != null ? `Flagged: ${info.flagged_count}` : "",
    ]
      .filter(Boolean)
      .join(" · ");
  }

  btnCheck.addEventListener("click", async () => {
    const product_name = productInput.value.trim() || "Unnamed product";
    const ingredients = parseIngredients(ingredientsInput.value);

    if (!ingredients.length) {
      setFormStatus("Add at least one ingredient, or use OCR on a label photo.", true);
      return;
    }

    setFormStatus("");
    btnCheck.disabled = true;
    results.classList.remove("visible");

    const task_id = taskSelect.value || "food_check_easy";

    try {
      const resetRes = await fetch(
        apiUrl("/reset?task_id=" + encodeURIComponent(task_id)),
        { method: "POST" }
      );
      if (!resetRes.ok) {
        throw new Error("Reset failed: " + resetRes.status);
      }

      const stepRes = await fetch(apiUrl("/step"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_name,
          ingredients,
          task_id,
        }),
      });

      if (!stepRes.ok) {
        const errText = await stepRes.text();
        throw new Error(errText || "Check failed: " + stepRes.status);
      }

      const data = await stepRes.json();
      showResults(data);
      setFormStatus("");
    } catch (e) {
      setFormStatus(e.message || "Request failed", true);
    } finally {
      btnCheck.disabled = false;
    }
  });
})();
