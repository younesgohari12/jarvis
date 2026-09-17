# JARVIS v0.11.0 Intelligence v18 — Cyberdeck UI Rebuild

This package keeps the Intelligence v18 brain/runtime intact and replaces the primary desktop conversation experience with a new futuristic command deck.

## Rebuilt from scratch

- Main chat layout and visual hierarchy
- Holographic conversation surface
- Layered user/JARVIS message cards with depth/shadow treatment
- Perspective neural backdrop with low-cost parallax particles and grid animation
- New command composer, file attach control, dry-run control and send control
- Reworked top command bar, performance/personality selectors and live state chip
- Reworked cognitive-core rail and navigation
- Restyled response controls and conversation-copy controls

## Kept compatible

- Persian / English text handling and RTL/LTR behavior
- Ctrl+C / Ctrl+V and Persian-layout clipboard fallback
- File attachments
- Conversation copy counts
- Regenerate, edit, stop, teach and feedback controls
- Live monitor, settings, lab, history, local knowledge and action viewer
- Existing agent/runtime/weights/datasets/training logic

## Runtime dependency

The UI uses `customtkinter==6.0.0`, already present in `requirements-runtime.txt`. No Electron, Chromium, web server or online UI dependency was added.

## Validation

- GUI modules compile successfully.
- UI/context/clipboard regression tests: 25/25 passed.
- Complete project suite: 686/686 tests passed when test modules were executed in batches.
- Intelligence/model files were not modified by this UI rebuild.
