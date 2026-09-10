# Sleepwell

**Tagline:** A bedtime story generator for adults with insomnia.

## What it is
A subscription audiobook app that generates a fresh 25-minute bedtime
story every night at 22:00, narrated by one of three AI voices. The
stories are procedurally scored: each night continues from the night
before, with a small cast of recurring characters that grow over months.

## How it works
- Uses a fine-tuned 7B model running on a private GPU cluster in
  Frankfurt.
- The narrative graph has roughly 40 hand-authored "anchors" (e.g.
  *a stranger arrives at a lighthouse*) and the LLM bridges between them.
- The narrator pipeline is three independent voices with different
  timbres, paced at ~155 words per minute.
- Users can pick "rain", "fireplace", or "thunderstorm" ambience.

## Pricing & availability
- $6.99 / month or $59 / year.
- Launched in March 2025.
- Available on iOS, Android, and as a web player.
- The team is based in Lisbon, Portugal.

## What it is **NOT**
- **Not a meditation app.** No breathing exercises, no body scans,
  no Headspace-style content. Only stories.
- **Not a podcast.** There is no host, no interviews, no news.
- **Not produced by Calm or Headspace.** Sleepwell is unrelated to
  either company, and the comparison points are not celebrities:
  Calm's "Sleep Stories" are pre-recorded voiceovers; Sleepwell is
  fully AI-generated from scratch.
- **Not available on Alexa or Google Home.** It only runs on phones
  and the web.
- **Not free.** There is no free tier.
