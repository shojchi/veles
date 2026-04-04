# AKS — Phase 5 Web UI Design Brief

## Overview

AKS (Agent Knowledge System) is a personal AI assistant that routes questions to specialized
agents (Code, PKM, Writing, Planning) and answers them grounded in the user's own Markdown notes.
The Web UI exposes the same capabilities as the CLI in a browser-based interface.

---

## Layout — Three-Column

```
┌──────────────────┬──────────────────────────────────┬─────────────────┐
│  LEFT SIDEBAR    │         CENTER — CHAT            │  RIGHT SIDEBAR  │
│  Knowledge Base  │                                  │  Provider Info  │
│  (240px)         │         (flex, fills space)      │  (220px)        │
└──────────────────┴──────────────────────────────────┴─────────────────┘
```

- **Left sidebar** (fixed width ~240px) — Knowledge Base panel
- **Center** (flex, takes remaining width) — Chat panel
- **Right sidebar** (fixed width ~220px, fixed height, no scroll) — Provider & Cost panel

---

## Panel 1 — Left Sidebar: Knowledge Base

### Purpose
Browse and search the user's personal notes (Markdown files). Read-only in Phase 5 — no editing.

### Elements

**Header**
- Title: "Knowledge Base"
- Two action buttons (top-right of panel):
  - `+ New` — disabled / greyed out in Phase 5 (future feature)
  - `Import` — opens the Import Modal

**Search bar**
- Placeholder: "Search notes…"
- Live filter: as user types, note list filters by title (client-side, no server round-trip for basic filter)
- Clear (×) button when text is present

**Note list**
- Scrollable, fills remaining height of sidebar
- Each row shows:
  - Note title (truncated if long)
  - Small muted date (last modified)
- Clicking a note opens a read-only preview (slide-over or inline expand — TBD in Phase 5 implementation)
- Empty state: "No notes yet. Import something to get started."

---

## Panel 2 — Center: Chat

### Purpose
Multi-turn conversation with AKS. Supports agent selection, streaming responses, and persistent history.

### Elements

**Message list**
- Scrollable, fills available height
- Bubbles:
  - **User messages** — right-aligned, filled background
  - **AKS messages** — left-aligned, subtle background
- Each AKS message shows an **agent chain badge** above the bubble (e.g. `router → code`, `pkm → writing`)
- Markdown rendered in AKS messages (code blocks, bold, lists)
- Streaming: text appears token-by-token (SSE)

**Empty / welcome state**
- Centered illustration or icon
- Tagline: "Ask anything. AKS routes to the right agent."
- 3–4 example prompt chips to click

**Input bar** (pinned to bottom)
- Agent selector dropdown: `auto` (default), `code`, `pkm`, `writing`, `planning`
- Text input: multiline, expands up to ~4 lines, placeholder "Message AKS…"
- Send button (arrow icon), disabled when input is empty
- Keyboard: `Enter` sends, `Shift+Enter` for newline

---

## Panel 3 — Right Sidebar: Provider & Cost

### Purpose
Static information panel. No interactivity — purely informational. Fixed height, no scroll.

### Elements

**Provider card**
- Label: "Provider"
- Value: e.g. `anthropic` or `cerebras` (read from config)
- Label: "Default model"
- Value: e.g. `claude-3-5-haiku-20241022`

**Cost card**
- Label: "Today's cost"
- Value: `$0.0024` (live, refreshed on each message)
- Label: "Daily cap"
- Value: `$1.00`
- Progress bar: filled portion = today/cap, colour shifts green → amber → red as % rises
- Percentage label: e.g. `0.2%`

**Active agents**
- Label: "Agents"
- Small table or list:
  | Agent   | Model          |
  |---------|----------------|
  | code    | claude-sonnet  |
  | pkm     | claude-haiku   |
  | writing | claude-haiku   |
  | planning| claude-haiku   |

---

## Import Modal

Triggered by the `Import` button in the Knowledge Base panel.

### Elements
- Modal title: "Import to Knowledge Base"
- Two tabs or toggle: **URL** | **File**

**URL tab**
- Label: "Paste a URL"
- Input: full-width text field, placeholder `https://…`
- Helper text: "We'll extract readable text from the page."
- Button: `Import` (primary)

**File tab**
- Drag-and-drop zone: "Drop a PDF here or click to browse"
- Accepts: `.pdf` only (Phase 5 scope)
- Shows selected filename once chosen
- Button: `Import` (primary)

**States**
- Loading: spinner, button disabled, text "Importing…"
- Success: green checkmark, "Saved to knowledge base", modal auto-closes after 1.5s
- Error: red inline message with the error text

---

## Top Bar (minimal)

- Left: `AKS` wordmark / logo
- No navigation — single-page app
- Optionally: theme toggle (light/dark) — nice to have

---

## Interaction Notes for Stitch

- Dark theme preferred (similar to Claude / ChatGPT dark mode)
- Accent colour: indigo or violet (to differentiate from Claude blue / ChatGPT green)
- Font: system-ui or Inter
- Sidebar dividers: subtle 1px borders, no heavy shadows
- Agent badge: small pill/chip, monospace font, muted colour per agent type
- Code blocks in chat: dark background, syntax-highlighted appearance
- Mobile: out of scope for Phase 5 — desktop only (min-width 1024px)

---

## What Is Out of Scope (Phase 5)

- Note editing / creation in UI (CLI only)
- Multi-user / auth
- Conversation history list (sidebar like ChatGPT) — chat persists across page reloads but no history browser
- Mobile layout
