# Chat Widget Integration Guide

## Architecture

```
GitHub Pages (public repo)              Azure Static Web Apps (private repo)
markandrewmarquez.com                   your-widget-app.azurestaticapps.net
┌──────────────────────┐                ┌──────────────────────┐
│  index.html          │  ──loads──►    │  widget.js           │
│  portfolio/index.html│  cross-origin  │  chat-widget.css     │
│  + embed snippet     │                │  (auto-deploy from   │
│    (2 script tags)   │                │   private GitHub repo)│
└──────────────────────┘                └──────────┬───────────┘
                                                   │
                                          API calls│
                                                   ▼
                                        Azure Container Apps
                                        your-app.azurecontainerapps.io
                                        ┌──────────────────────┐
                                        │  FastAPI backend     │
                                        │  /chat/stream        │
                                        │  /chat/init          │
                                        └──────────────────────┘
```

Widget code stays **private** (Azure Static Web Apps from private repo).
Your main site repo stays **public** — only a 2-line embed snippet is visible.

---

## Step 1: Deploy widget to Azure Static Web Apps

Push `widget.js` and `chat-widget.css` to your **private** GitHub repo.
Azure Static Web Apps auto-deploys on push.

Your private repo structure:
```
private-widget-repo/
├── widget.js
├── chat-widget.css
└── index.html          ← optional standalone demo page
```

After deployment, your files are at:
- `https://your-widget-app.azurestaticapps.net/widget.js`
- `https://your-widget-app.azurestaticapps.net/chat-widget.css`

---

## Step 2: Add embed snippet to GitHub Pages

On each HTML page where you want the chat bubble, add these 2 lines
right before `</body>`, after your Google Fonts `<link>`:

```html
  <link href="https://fonts.googleapis.com/css?family=Montserrat:500,600|Raleway:400,400i,600" rel="stylesheet">

  <!-- AI Sales Lead Bot Chat Widget -->
  <script>window.CHAT_BACKEND_URL = 'https://your-app.azurecontainerapps.io';</script>
  <script src="https://your-widget-app.azurestaticapps.net/widget.js" type="module"></script>

</body>
</html>
```

The CSS loads automatically — `widget.js` detects its own origin via
`import.meta.url` and resolves `chat-widget.css` from the same domain.

---

## Step 3: Configure CORS on the backend

### Backend (.env)
```
CORS_ORIGINS=https://markandrewmarquez.com,https://your-widget-app.azurestaticapps.net,http://localhost:8000,http://127.0.0.1:5500
```

Both origins need to be allowed because:
- **GitHub Pages** (`markandrewmarquez.com`) — where the page lives
- **Azure Static Web Apps** — where `widget.js` makes `fetch()` calls from

Azure Static Web Apps and GitHub Pages don't need any CORS config
on their end — they just serve static files.

---

## Step 4: Replace URL placeholders

| Placeholder | Replace with |
|---|---|
| `your-widget-app.azurestaticapps.net` | Your Azure Static Web Apps domain |
| `your-app.azurecontainerapps.io` | Your Azure Container Apps domain |
| `markandrewmarquez.com` | Already set (your GitHub Pages custom domain) |

---

## Local development

For local testing, comment out the backend URL (defaults to localhost:8000)
and load widget.js from a local path:

```html
  <!-- <script>window.CHAT_BACKEND_URL = '...';</script> -->
  <script src="widget.js" type="module"></script>
```

Or use VS Code Live Server — add `http://127.0.0.1:5500` to CORS_ORIGINS.

---

## Optional: CTA link that opens the chat

```html
<a href="#" onclick="window.__chatWidget && window.__chatWidget.toggle(); return false;">
  Chat with our AI Advisor
</a>
```
