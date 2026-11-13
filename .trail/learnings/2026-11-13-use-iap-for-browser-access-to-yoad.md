---
title: Use IAP for browser access to the GPU demo
author: Yoad
date: 2026-11-13
branch: main
id: l-96741f27
scope: repo
status: active
src: agent
touches:
  - deploy/README.md
---

## Decided
- Protect the Cloud Run demo with direct IAP and service-scoped user email grants. Cloud Run's IAM-only invoker check does not start a browser Google sign-in flow, so whitelisting run.invoker identities alone returns Forbidden to normal browser navigation.
- Keep Cloud Run's unauthenticated invoker disabled; grant the IAP service agent run.invoker and named users roles/iap.httpsResourceAccessor on the service. This allows the run.app URL to redirect to Google sign-in without a local proxy.
## Watch out
- Gmail users may count as external to the project's Google organization; verify the OAuth client supports external users and have each reviewer sign in with the exact whitelisted email. Never switch the service to allUsers just to fix a browser 403.
## In the code:
- deploy/README.md documents the Cloud Run auth policy and service URL; service biased-whisper-l4 in project project-71583fab-5cd0-4bba-a6f region europe-west1.
