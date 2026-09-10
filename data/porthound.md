# Porthound

**Tagline:** A peer-to-peer Git hosting daemon that fits in one binary.

## What it is
A single-binary Git server written in Zig. Total disk footprint under
9 MB. Replaces a typical self-hosted-Git-on-a-VPS stack with
one static executable and an embedded database.

## How it works
- Listens on a configurable port and serves the smart HTTP Git protocol.
- Supports `git push`, `git pull`, and web browsing of repositories.
- Has no user model beyond SSH public keys — every push is attributed
  to whoever holds the key.
- Optional Tailscale integration for zero-config discovery of other
  Porthound instances on the same tailnet.
- Ships as a single binary for linux/amd64, linux/arm64, and darwin/arm64.

## Pricing & availability
- Free and open-source under the BSD-2-Clause license.
- The author (a person named Janelle Brubeck) maintains it on weekends.
- There is no company behind it and no paid tier.

## What it is **NOT**
- **Not a SaaS.** There is no hosted version, no porthound.io, no
  login flow. You run it on your own box.
- **Not a fork of Gitea.** Zero lines of that project's code are reused.
- **Not related to GitHub, GitLab, Bitbucket, or SourceHut.** It is a
  hobby project, not affiliated with any of them.
- **Not based on libgit2.** It speaks the wire protocol directly.
- **Not a Docker image.** A single static binary is the only
  distribution.
- **Not backed by an SQL server.** It uses an embedded database, not
  Postgres, MySQL, or any external daemon.
