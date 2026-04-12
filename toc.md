---
layout: course_page
title: Table of Contents
---

# Table of Contents

[Looking for the 2025 version? It's archived here.](archive/2025/toc.md)

<a id="export-lectures" href="#">Export lectures to markdown</a>

[0. Introduction](section/0/notes.md) | [Slides](section/0/slides.pdf) | [Notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/0/notebook.ipynb)
   > Course content, a deliverable, and spam classification in PyTorch.

[1. Optimization and PyTorch Basics in 1D](section/1/notes.md)
  > Optimization setup, minimizers and stationarity, 1D gradient descent, diagnostics, step-size tuning, and PyTorch autodiff basics.

[2. Stochastic Optimization Basics in 1D](section/2/notes.md) 
   > Empirical risk, SGD updates, step-size schedules, noise floors, unbiasedness and variance, minibatches, and validation diagnostics.

[3. Optimization and PyTorch basics in higher dimensions](section/3/notes.md) | [Live demo](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/3/live-demo.ipynb)
  > Lift optimization to $\mathbb{R}^d$, derive gradient descent from the local model, and tour PyTorch tensors, efficiency, dtypes, and devices.

[4. Loss functions and models for regression and classification problems](section/4/notes.md) | [Live demo](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/4/live-demo.ipynb)
   > Formulate ML objectives, choose losses for regression/classification, and build/train linear and convolutional models in PyTorch.

[5. A step-by-step introduction to transformer models](section/5/notes.md)
   > Building transformers from scratch: embeddings, attention, residual connections, and next-token prediction on Shakespeare.

[6. A step-by-step introduction to diffusion models](section/6/notes.md)
   > Diffusion models from first principles: forward process, reverse process, noise prediction, U-Net, sampling, DDIM, conditional generation, and FID.

[7. Reinforcement learning for language models](section/7/notes.md)
   > The REINFORCE gradient estimator, baselines, KL penalties, rejection sampling, gradient weight rescaling, and a reward shaping experiment on Shakespeare.

[8. More on optimizers](section/8/notes.md)
   > Algorithm modifiers (momentum, schedulers, gradient clipping), techniques that change the problem (LoRA, quantization, weight decay), the optimizer zoo (SignSGD, Signum, AdaGrad, RMSProp, Adam, AdamW), coordinate-wise scaling, Newton's method, and Muon.

[9. Benchmarking Optimizers](section/9/notes.md)
   > How to compare optimizers fairly: time-to-result, why tuning is inseparable from the optimizer, and the AlgoPerf benchmark.

   
---

Some 2025 content below; yet to be deleted.

[11. A Playbook for Tuning Deep Learning Models](section/11/notes.md) | [Cheatsheet](section/11/cheatsheet.md)
   > A systematic process for [tuning deep learning models](https://github.com/google-research/tuning_playbook)

[12. Scaling Transformers: Parallelism Strategies from the Ultrascale Playbook](section/12/notes.md) | [Cheatsheet](section/12/cheatsheet.md)
   > How do we scale training of transformers to 100s of billions of parameters?

[Recap](section/recap/notes.md) | [Cheatsheet](section/recap/cheatsheet.md)
   > A recap of the course.

<script>
(() => {
  const button = document.getElementById("export-lectures");
  if (!button) return;

  const REPO_OWNER = "damek";
  const REPO_NAME = "STAT-4830";
  const BRANCH = "main";

  function normalizePath(href) {
    const url = new URL(href, window.location.href);
    const pathname = url.pathname || "";
    const withoutPrefix = pathname.replace(/^\/?STAT-4830\//, "");
    const mdPath = withoutPrefix.endsWith(".md")
      ? withoutPrefix
      : withoutPrefix.endsWith(".html")
        ? withoutPrefix.replace(/\.html$/, ".md")
        : `${withoutPrefix}.md`;
    return mdPath.replace(/^\//, "");
  }

  async function fetchRaw(path) {
    const cleanPath = normalizePath(path);
    const primary = `https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${BRANCH}/${cleanPath}`;
    const fallback = `https://cdn.jsdelivr.net/gh/${REPO_OWNER}/${REPO_NAME}@${BRANCH}/${cleanPath}`;

    const tryFetch = async (url) => {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`${res.status} ${url}`);
      return res.text();
    };

    try {
      return await tryFetch(primary);
    } catch (e) {
      return await tryFetch(fallback);
    }
  }

  async function exportLectures() {
    const links = Array.from(
      document.querySelectorAll('a[href*="section/"][href*="notes"]')
    );
    if (!links.length) {
      alert("No lecture notes found on this page.");
      return;
    }

    const originalLabel = button.textContent;
    button.textContent = "Exporting...";

    try {
      const parts = [];
      for (const link of links) {
        const title = (link.textContent || link.href).trim();
        const href = link.getAttribute("href") || "";
        const text = await fetchRaw(href);
        parts.push(`\n\n---\n\n# ${title}\n\n${text.trim()}\n`);
      }

      const blob = new Blob(parts, { type: "text/markdown" });
      const download = document.createElement("a");
      download.href = URL.createObjectURL(blob);
      download.download = "lectures-export.md";
      download.click();
      URL.revokeObjectURL(download.href);
    } catch (err) {
      console.error(err);
      alert(`Export failed: ${err.message}`);
    } finally {
      button.textContent = originalLabel;
    }
  }

  button.addEventListener("click", (e) => {
    e.preventDefault();
    exportLectures();
  });
})();
</script>