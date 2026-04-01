---
title: Redis Caching Strategy
---

Use SETEX for TTL-based caching. Always prefix keys with service name to avoid collisions. Invalidate on write, not on read. Avoid caching large objects over 1MB.
