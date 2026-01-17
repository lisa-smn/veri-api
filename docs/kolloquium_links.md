# Links für Kolloquium-Präsentation

## 1) GitHub-Link

**Remote-URL (origin):**
```
https://github.com/lisa-smn/veri-api.git
```

**Saubere HTTPS-URL (ohne .git):**
```
https://github.com/lisa-smn/veri-api
```

---

## 2) Docker-Links

**Docker-Compose Datei:**
```
https://github.com/lisa-smn/veri-api/blob/main/docker-compose.yml
```

**Dockerfile:**
```
https://github.com/lisa-smn/veri-api/blob/main/Dockerfile
```

**Hinweis:** Keine eigenen Registry-Images vorhanden. Docker-Compose verwendet Standard-Images:
- `postgres:16` (Docker Hub: https://hub.docker.com/_/postgres)
- `neo4j:5` (Docker Hub: https://hub.docker.com/_/neo4j)
- API wird lokal gebaut (`build: .`)

**Docker-Kommandos (aus docker-compose.yml belegt):**
- Start: `docker-compose up` (oder `docker compose up`)
- Mit Build: `docker-compose up --build`

---

## 3) Folienfertige Ausgabe

### Zeile 1: GitHub
```
GitHub: https://github.com/lisa-smn/veri-api
```

### Zeile 2: Docker
```
Docker: https://github.com/lisa-smn/veri-api/blob/main/docker-compose.yml
```

### Optional: Docker Run-Kommando
```
Run: docker-compose up
```

---

## Alternative Format (mit kurzen Linktexten)

**Für Folien:**
- **GitHub:** [github.com/lisa-smn/veri-api](https://github.com/lisa-smn/veri-api)
- **Docker:** [docker-compose.yml](https://github.com/lisa-smn/veri-api/blob/main/docker-compose.yml)

**Mit Run-Kommando:**
- **GitHub:** [github.com/lisa-smn/veri-api](https://github.com/lisa-smn/veri-api)
- **Docker:** [docker-compose.yml](https://github.com/lisa-smn/veri-api/blob/main/docker-compose.yml) | Run: `docker-compose up`





