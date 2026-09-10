# Astrolabe

**Tagline:** A tiny offline star map for one specific night.

## What it is
A mobile app that, given a date and a latitude, generates a single
hand-drawn star map for that one night and nothing else. No search, no
sky calendar, no notifications. The map is rendered once, on-device,
as a 1500×1500 SVG that you can save or share.

## How it works
- The HYG star database (a public catalog of ~14k named stars)
  is bundled in the app — about 1.8 MB compressed.
- The renderer is a hand-written projection in Kotlin / Swift.
- Stars are sized by apparent magnitude, and a hand-tuned palette
  distinguishes planets, bright stars, deep-sky objects, and the moon.
- No network connection is ever used; the app is fully offline.

## Pricing & availability
- $1.99 one-time, no subscription.
- Released in November 2024.
- Available on iOS and Android, no tablet-specific layout.
- Built by a single developer, Ada Jovanović, in Belgrade, Serbia.

## What it is **NOT**
- **Not a planetarium.** There is no real-time tracking, no
  augmented-reality overlay, no "hold to the sky" feature.
- **Not a competitor to Stellarium, Sky Map, or SkySafari.** Astrolabe
  is intentionally minimal; those are full planetarium apps.
- **Not free.** It costs $1.99 (one-time).
- **Not a science tool.** It is not designed for astronomical
  observation planning, ephemerides, or telescope control.
- **Not a web app.** There is no browser version.
