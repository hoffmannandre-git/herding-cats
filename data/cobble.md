# Cobble

**Tagline:** The macropad that programs itself by watching you.

## What it is
A 12-key macropad with a small e-ink display. It runs an on-device
behavior model that watches the first 50 keypresses of any new app and
generates a keymap from observed usage. No cloud, no config files,
no driver. The firmware is roughly 240 KB of Rust.

## How it works
- Watches keystroke timing and app focus events.
- After 50 samples, fits a tiny HMM to model which keys lead to which
  intents.
- Suggests a layout on the e-ink screen. Press "yes" to apply.
- A 2032 coin cell lasts about 14 months of normal use.

## Pricing & availability
- $89 for the 12-key model, $129 for the 18-key with rotary encoder.
- First batch sold out in 11 minutes in October 2025.
- Currently shipping from a small workshop in Tallinn, Estonia.

## What it is **NOT**
- **Not open source.** The firmware is a closed-source `.uf2`.
- **Not cloud-synced.** Layouts never leave the device.
- **Not a Stream Deck alternative.** It has no plugins, no icons, no
  touch surface, and no Windows / macOS driver. It only sends raw HID
- **Not a product from Elgato.** Elgato makes the Stream Deck line;
  Cobble is unrelated.
- **Not a Raspberry Pi project.** There is no Pi inside.
