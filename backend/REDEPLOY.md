# Deploying to Azure Container Apps

Quick-reference for redeploying the backend after code changes.

---

## Prerequisites

- Azure CLI installed and logged in (`az login`)
- You are in the `backend/` directory
- The `containerapp` CLI extension is installed (`az extension add --name containerapp --upgrade`)

---

## Full Deploy + Test Workflow

Follow all steps in order every time you deploy.

### Step 1: Run tests locally

```powershell
cd backend
pytest tests/ -v
```

All tests must pass before deploying.

### Step 2: Build and deploy

```powershell
# Build the image remotely (bump the version tag each time: v3, v4, v5...) -- current: v6
az acr build --registry aileadbotacr --resource-group rg-ai-lead-bot --image ai-lead-bot:v6 .

# Update the container app to use the new image
az containerapp update --name salesforce-langgraph-ai-lead-bot --resource-group rg-ai-lead-bot --image aileadbotacr.azurecr.io/ai-lead-bot:v6
```

### Step 3: Wait for the container to start

```powershell
# Health check (may take 10-15s on cold start)
curl https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io/health

# Salesforce connectivity
curl https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io/health/salesforce
```

### Step 4: Hard reset the browser

**This is critical — the browser caches the old widget code and conversation state.**

1. Open an **incognito / private window** in your browser (Ctrl+Shift+N in Chrome)
2. Go to `https://markandrewmarquez.com`
3. Do a **hard refresh**: `Ctrl + Shift + R`
4. Open DevTools (F12) → Application tab → Clear Site Data (or manually clear Local Storage)
5. Close the incognito window and open a fresh one for each test run

### Step 5: Test the conversation

Run through a full conversation and verify:

- [ ] TARS greets you on bubble click
- [ ] Discovery: TARS asks about pain points and current tools (no "what impact" questions)
- [ ] Qualification: TARS asks about budget, timeline, company size, and decision-maker (no repeated questions)
- [ ] Lead capture: TARS asks for name, email, and company
- [ ] Confirmation: TARS summarizes and ends with "Does that all look correct?"
- [ ] No transcript summary leaks after the confirmation
- [ ] Lead appears in Salesforce with correct data

---

## Update Environment Variables (if needed)

```powershell
az containerapp update `
  --name salesforce-langgraph-ai-lead-bot `
  --resource-group rg-ai-lead-bot `
  --set-env-vars `
    LLM_PROVIDER=xai `
    CORS_ORIGINS="https://markandrewmarquez.com,https://zealous-moss-0360b7210.7.azurestaticapps.net"
```

For secrets (API keys, passwords):

```powershell
az containerapp secret set `
  --name salesforce-langgraph-ai-lead-bot `
  --resource-group rg-ai-lead-bot `
  --secrets `
    xai-key="your-key-here"
```

---

## Roll Back

If a deployment breaks something, revert to the previous image tag:

```powershell
az containerapp update --name salesforce-langgraph-ai-lead-bot --resource-group rg-ai-lead-bot --image aileadbotacr.azurecr.io/ai-lead-bot:v2
```

---

## View Logs

```powershell
az containerapp logs show `
  --name salesforce-langgraph-ai-lead-bot `
  --resource-group rg-ai-lead-bot `
  --type console `
  --follow
```

---

## Azure Resources Reference

| Resource | Name |
|---|---|
| Resource Group | `rg-ai-lead-bot` |
| Container Registry | `aileadbotacr` |
| Container App | `salesforce-langgraph-ai-lead-bot` |
| Container Apps Environment | `ai-lead-bot-env` |
| Region | `centralus` |
| App URL | `https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io` |
