# Azure Deployment Guide

Deploy the AI Sales Lead Bot to Azure Container Apps (backend) and Azure Static
Web Apps (frontend widget).

**Live URLs:**
- Backend API: `https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io`
- Frontend Widget: `https://zealous-moss-0360b7210.7.azurestaticapps.net`
- Portfolio Site: `https://markandrewmarquez.com`

---

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- An Azure subscription
- [Node.js](https://nodejs.org/) (for the SWA CLI)
- The `containerapp` CLI extension: `az extension add --name containerapp --upgrade`

---

## Part 1: Backend — Azure Container Apps

### 1. Set variables

```powershell
$RESOURCE_GROUP = "rg-ai-lead-bot"
$LOCATION = "centralus"
$ENVIRONMENT = "ai-lead-bot-env"
$APP_NAME = "salesforce-langgraph-ai-lead-bot"
```

### 2. Register providers and create resource group

```powershell
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry

az group create --name $RESOURCE_GROUP --location $LOCATION
```

### 3. Create Container Apps environment

```powershell
az containerapp env create `
  --name $ENVIRONMENT `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION
```

### 4. Create Azure Container Registry

```powershell
az acr create `
  --name aileadbotacr `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku Basic `
  --admin-enabled true
```

### 5. Build the Docker image in ACR

```powershell
cd backend

az acr build `
  --registry aileadbotacr `
  --resource-group $RESOURCE_GROUP `
  --image ai-lead-bot:v1 `
  --file Dockerfile .
```

### 6. Deploy to Container Apps

```powershell
az containerapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --environment $ENVIRONMENT `
  --image aileadbotacr.azurecr.io/ai-lead-bot:v1 `
  --registry-server aileadbotacr.azurecr.io `
  --registry-username aileadbotacr `
  --registry-password (az acr credential show --name aileadbotacr --query "passwords[0].value" --output tsv) `
  --ingress external `
  --target-port 8000
```

### 7. Set scale-to-zero

```powershell
az containerapp update `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --min-replicas 0 `
  --max-replicas 1
```

### 8. Set secrets

```powershell
az containerapp secret set `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --secrets `
    xai-key="<your-XAI_API_KEY>" `
    sf-client-id="<your-SF_CLIENT_ID>" `
    sf-client-secret="<your-SF_CLIENT_SECRET>" `
    sf-password="<your-SF_PASSWORD>" `
    sf-token="<your-SF_SECURITY_TOKEN>"
```

> **Tip:** You can automate this from your `.env` file using PowerShell:
> ```powershell
> $env_vars = @{}
> Get-Content .env | ForEach-Object {
>     if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
>         $env_vars[$matches[1].Trim()] = $matches[2].Trim()
>     }
> }
> az containerapp secret set `
>   --name $APP_NAME --resource-group $RESOURCE_GROUP `
>   --secrets `
>     xai-key="$($env_vars['XAI_API_KEY'])" `
>     sf-client-id="$($env_vars['SF_CLIENT_ID'])" `
>     sf-client-secret="$($env_vars['SF_CLIENT_SECRET'])" `
>     sf-password="$($env_vars['SF_PASSWORD'])" `
>     sf-token="$($env_vars['SF_SECURITY_TOKEN'])"
> ```

### 9. Set environment variables

```powershell
az containerapp update `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --set-env-vars `
    LLM_PROVIDER=xai `
    LLM_MODEL=grok-4-1-fast-reasoning `
    LLM_TEMPERATURE=0.7 `
    XAI_API_KEY=secretref:xai-key `
    SF_INSTANCE_URL=https://orgfarm-d025e11150-dev-ed.develop.my.salesforce.com `
    SF_CLIENT_ID=secretref:sf-client-id `
    SF_CLIENT_SECRET=secretref:sf-client-secret `
    SF_USERNAME=me.1b89f9c8baec@agentforce.com `
    SF_PASSWORD=secretref:sf-password `
    SF_SECURITY_TOKEN=secretref:sf-token `
    CORS_ORIGINS="http://localhost:3000,https://markandrewmarquez.com,https://zealous-moss-0360b7210.7.azurestaticapps.net" `
    LOG_LEVEL=INFO
```

### 10. Verify

```powershell
# Health check
Invoke-RestMethod -Uri "https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io/health"

# Salesforce connectivity
Invoke-RestMethod -Uri "https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io/health/salesforce"
```

### Redeploying after code changes

When you update backend code, rebuild and deploy a new image version:

```powershell
cd backend

az acr build `
  --registry aileadbotacr `
  --resource-group $RESOURCE_GROUP `
  --image ai-lead-bot:v2 `
  --file Dockerfile .

az containerapp update `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --image aileadbotacr.azurecr.io/ai-lead-bot:v2
```

Increment the tag (`v2`, `v3`, etc.) each time.

---

## Part 2: Frontend Widget — Azure Static Web Apps

### 1. Install the SWA CLI

```powershell
npm install -g @azure/static-web-apps-cli
```

### 2. Create the Static Web App

```powershell
az staticwebapp create `
  --name ai-lead-bot-widget `
  --resource-group $RESOURCE_GROUP `
  --location centralus
```

### 3. Deploy widget files

```powershell
$SWA_TOKEN = az staticwebapp secrets list `
  --name ai-lead-bot-widget `
  --resource-group $RESOURCE_GROUP `
  --query "properties.apiKey" `
  --output tsv

swa deploy ./frontend --deployment-token $SWA_TOKEN --env production
```

### 4. Get the widget URL

```powershell
az staticwebapp show `
  --name ai-lead-bot-widget `
  --resource-group $RESOURCE_GROUP `
  --query "defaultHostname" `
  --output tsv
```

### Redeploying after frontend changes

```powershell
swa deploy ./frontend --deployment-token $SWA_TOKEN --env production
```

If `$SWA_TOKEN` expired (new terminal session), re-fetch it with the
`az staticwebapp secrets list` command from step 3.

---

## Part 3: GitHub Pages Embed

Add these two lines before `</body>` on any page where you want the chat bubble:

```html
<script>window.CHAT_BACKEND_URL = 'https://salesforce-langgraph-ai-lead-bot.purplesky-0949fcd0.centralus.azurecontainerapps.io';</script>
<script src="https://zealous-moss-0360b7210.7.azurestaticapps.net/widget.js" type="module"></script>
```

---

## Free Tier Limits

| Resource | Free Tier Limit | Notes |
|---|---|---|
| Container Apps | 2M requests/month | Scale to zero when idle |
| | 180K vCPU-seconds/month | ~50 hours of active compute |
| | 360K GiB-seconds/month | ~100 hours at 1 GiB |
| Static Web Apps | 100 GB bandwidth/month | More than enough for widget files |
| | 2 custom domains | markandrewmarquez.com + optional subdomain |

For a portfolio project with moderate traffic, the free tier is sufficient.

---

## Monitoring

### View logs

```powershell
az containerapp logs show `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --follow
```

### View revision status

```powershell
az containerapp show `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "properties.latestRevisionName"
```

Or use the Azure Portal → Container Apps → your app → Monitoring → Logs.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Cold start slow (10-15s) | Expected with `min-replicas: 0`. Set `--min-replicas 1` if latency matters (costs more). |
| Container crashes on start | Check logs: `az containerapp logs show`. Usually a missing env var or import error. |
| CORS errors in browser | Verify `CORS_ORIGINS` includes both your GitHub Pages domain and Azure Static Web Apps domain. |
| SSE streaming not working | Ensure `--timeout-keep-alive 120` is set in the Dockerfile CMD. Azure's load balancer may close idle connections. |
| Health check failing | The `/health` endpoint must respond within 5s. If the LLM init is slow, increase `--start-period`. |
| `invalid_client_id` from Salesforce | Secrets contain placeholder text instead of real credentials. Re-run `az containerapp secret set` with values from your `.env` file. |
| Widget not loading cross-origin | Ensure `frontend/staticwebapp.config.json` exists with `Access-Control-Allow-Origin: *` and is deployed to Azure Static Web Apps. |
