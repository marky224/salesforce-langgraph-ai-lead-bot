# Chat Widget Integration Guide

## Architecture

```
GitHub Pages (public repo)              Azure Static Web Apps
markandrewmarquez.com                   zealous-moss-0360b7210.7.azurestaticapps.net
┌──────────────────────┐                ┌──────────────────────┐
│  index.html          │  ──loads──►    │  widget.js           │
│  portfolio/index.html│  cross-origin  │  chat-widget.css     │
│  + embed snippet     │                │  tars-avatar.svg     │
│    (2 script tags)   │                │  staticwebapp.config │
└──────────────────────┘                └──────────┬───────────┘
                                                   │
                                          API calls│
                                                   ▼
                                        Azure Container Apps
                                        salesforce-langgraph-ai-lead-bot
                                          .purplesky-0949fcd0
                                          .centralus.azurecontainerapps.io
                                        ┌──────────────────────┐
                                        │  FastAPI backend     │
                                        │  /chat/stream        │
                                        │  /chat/init          │
                                        └──────────────────────┘
```

Widget code is deployed to Azure Static Web Apps via the SWA CLI (no GitHub
repo required). Your main site repo stays public — only a 2-line embed
snippet is visible.

---

## Step 1: Deploy widget to Azure Static Web Apps

Install the SWA CLI and deploy the `frontend/` directory:

```powershell
npm install -g @azure/static-web-apps-cli

$SWA_TOKEN = az staticwebapp secrets list `
  --name ai-lead-bot-widget `
  --resource-group rg-ai-lead-bot `
  --query "properties.apiKey" `
  --output tsv

swa deploy ./frontend --deployment-token $SWA_TOKEN --env production
```

After deployment, your files are at:
- `https://zealous-moss-0360b7210.7.azurestaticapps.net/widget.js`
- `https://zealous-moss-0360b7210.7.azurestaticapps.net/chat-widget.css`
- `https://zealous-moss-0360b7210.7.azurestaticapps.net/tars-avatar.svg`

---

## Step 2: Add embed snippet to GitHub Pages

On each HTML page where you want the chat bubble, add these 2 lines
right before `</body>`:

```html
  <!-- AI Sales Lead Bot Chat Widget -->
  <script>window.CHAT_BACKEND_URL = 'https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io';</script>
  <script src="https://zealous-moss-0360b7210.7.azurestaticapps.net/widget.js" type="module"></script>

</body>
</html>
```

The CSS loads automatically — `widget.js` detects its own origin via
`import.meta.url` and resolves `chat-widget.css` from the same domain.

---

## Step 3: Configure CORS on the backend

### Backend environment variable
```
CORS_ORIGINS=http://localhost:3000,https://markandrewmarquez.com,https://zealous-moss-0360b7210.7.azurestaticapps.net
```

Both origins need to be allowed because:
- **GitHub Pages** (`markandrewmarquez.com`) — where the page lives
- **Azure Static Web Apps** — where `widget.js` makes `fetch()` calls from

### Cross-origin script loading
The `frontend/staticwebapp.config.json` file adds CORS headers so that
`<script type="module">` can load `widget.js` cross-origin from GitHub Pages:

```json
{
  "globalHeaders": {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  }
}
```

Azure Static Web Apps and GitHub Pages don't need any additional CORS config
on their end — they just serve static files.

---

## Step 4: URL Reference

| Resource | URL |
|---|---|
| Backend API | `https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io` |
| Backend Swagger UI | `https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io/docs` |
| Frontend Widget | `https://zealous-moss-0360b7210.7.azurestaticapps.net` |
| Portfolio Site | `https://markandrewmarquez.com` |

---

## Local Development

For local testing, the widget defaults to `localhost:8000` when
`window.CHAT_BACKEND_URL` is not set:

```bash
# Terminal 1 — Backend
cd backend
uvicorn app.server:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
python -m http.server 3000

# Open http://localhost:3000
```

Make sure `http://localhost:3000` is in `CORS_ORIGINS` in your `.env`.

---

## Optional: CTA link that opens the chat

```html
<a href="#" onclick="window.__chatWidget && window.__chatWidget.toggle(); return false;">
  Chat with our AI Advisor
</a>
```
