# Manuscriptr

**Tagline:** A version-control UI built specifically for novelists.

## What it is
A desktop app (Windows / macOS / Linux) that wraps Git with a UI aimed
at novelists and other long-form prose writers who have never used the
command line. It visualizes branches as "drafts" and merges as
"combining chapters", and has a side-by-side diff viewer tuned for
prose.

## How it works
- Bundles a portable `git` binary; no system Git install required.
- Default storage backend is the local filesystem; a paid tier adds
  end-to-end-encrypted cloud sync via a small relay.
- The diff viewer tokenizes prose and shows red/green at the
  paragraph level — line-level diffs are hidden behind an "advanced"
  toggle.
- Plugins for Scrivener and Ulysses let you import `.scrivx` and
  `.rtfd` bundles.

## Pricing & availability
- $49 one-time for the desktop app.
- Cloud sync is $4 / month.
- First public release: June 2025.
- Made by a two-person studio in Wellington, New Zealand.

## What it is **NOT**
- **Not a text editor.** The editor is read-only; you write in
  Scrivener, Ulysses, Word, or any other tool.
- **Not based on GitHub Desktop, GitKraken, Sourcetree, or any other
  existing Git GUI.** Manuscriptr was written from scratch in Tauri.
- **Not affiliated with Scrivener or Ulysses.** Both are
  third-party tools; we just integrate with them.
- **Not a cloud service.** The base tier is fully offline.
- **Not language-model-powered.** There is no autocomplete, no rewrite,
  no generative feature — only version control.
