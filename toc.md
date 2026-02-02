---
layout: course_page
title: Table of Contents
---

# Table of Contents

[2025 version of the course](archive/2025/toc.md)

<a id="export-lectures" href="#">Export lectures to markdown</a>

[0. Introduction](section/0/notes.md) | [Slides](section/0/slides.pdf) | [Notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/0/notebook.ipynb)
   > Course content, a deliverable, and spam classification in PyTorch.

[1. Optimization and PyTorch Basics in 1D](section/1/notes.md)
  > Optimization setup, minimizers and stationarity, 1D gradient descent, diagnostics, step-size tuning, and PyTorch autodiff basics.

[2. Stochastic Optimization Basics in 1D](section/2/notes.md) 
   > Empirical risk, SGD updates, step-size schedules, noise floors, unbiasedness and variance, minibatches, and validation diagnostics.

[3. Optimization and PyTorch basics in higher dimensions](section/3/notes.md)
  > Lift optimization to $\mathbb{R}^d$, derive gradient descent from the local model, and tour PyTorch tensors, efficiency, dtypes, and devices.

---

Some 2025 content below; yet to be deleted.

[4. How to compute gradients in PyTorch](section/4/notes.md) | [Slides](section/4/slides.pdf) | [Notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/4/notebook.ipynb)
   > Introduction to PyTorch's automatic differentiation system.

[5. How to think about derivatives through best linear approximation](section/5/notes.md)
   > How to think about derivatives through best linear approximation.

[6. Stochastic gradient descent: A first look](section/6/notes.md)
   > A first look at stochastic gradient descent through the mean estimation problem.

[7. Stochastic gradient descent: insights from the Noisy Quadratic Model](section/7/notes.md)
   > When should we use exponential moving averages, momentum, and preconditioning?

[8. Stochastic Gradient Descent: The general problem and implementation details](section/8/notes.md) | [Notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/8/notebook.ipynb)
   > Stochastic optimization problems, SGD, tweaks, and implementation in PyTorch

[9. Adaptive Optimization Methods](section/9/notes.md) | [Notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/9/notebook.ipynb) | [Cheatsheet](section/9/cheatsheet.md)
   > Intro to adaptive optimization methods: Adagrad, Adam, and AdamW.

[10. Benchmarking Optimizers: Challenges and Some Empirical Results](section/10/notes.md) | [Cheatsheet](section/10/cheatsheet.md)
   > How do we compare optimizers for deep learning? 

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