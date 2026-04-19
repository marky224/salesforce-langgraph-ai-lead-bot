# Azure Container Apps Deployment

Deploy the FastAPI backend to Azure Container Apps (free tier: 2M requests/month,
180K vCPU-seconds).

---

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- An Azure subscription (Mark's existing one)
- Docker installed locally (for building the image)
- The `containerapp` CLI extension: `az extension add --name containerapp --upgrade`

---

## Option A: Deploy via Azure CLI (Manual)

### 1. Set variables

```bash
RESOURCE_GROUP="rg-ai-lead-bot"
LOCATION="centralus"
ENVIRONMENT="ai-lead-bot-env"
APP_NAME="salesforce-langgraph-ai-lead-bot"
```

### 2. Create resource group

```bash
az group create --name $RESOURCE_GROUP --location $LOCATION
```

### 3. Create Container Apps environment

```bash
az containerapp env create \
  --name $ENVIRONMENT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

### 4. Build and deploy in one command

Azure Container Apps can build from source — no container registry needed:

```bash
cd backend

az containerapp up \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --source . \
  --ingress external \
  --target-port 8000 \
  --min-replicas 0 \
  --max-replicas 1
```

> `--min-replicas 0` enables scale-to-zero (saves cost on free tier).
> The app cold-starts in ~10-15 seconds on first request.

### 5. Set environment variables (secrets)

```bash
az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars \
    LLM_PROVIDER=anthropic \
    ANTHROPIC_API_KEY=secretref:anthropic-key \
    SF_INSTANCE_URL=https://your-domain.develop.my.salesforce.com \
    SF_CLIENT_ID=secretref:sf-client-id \
    SF_CLIENT_SECRET=secretref:sf-client-secret \
    SF_USERNAME=your-user@salesforce.com \
    SF_PASSWORD=secretref:sf-password \
    SF_SECURITY_TOKEN=secretref:sf-token \
    CORS_ORIGINS="https://markandrewmarquez.com,https://your-widget.azurestaticapps.net" \
    LOG_LEVEL=INFO
```

Set the secrets separately:

```bash
az containerapp secret set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --secrets \
    anthropic-key="sk-ant-..." \
    sf-client-id="your-client-id" \
    sf-client-secret="your-client-secret" \
    sf-password="your-password" \
    sf-token="your-token"
```

### 6. Get the app URL

```bash
az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
```

This returns something like: `salesforce-langgraph-ai-lead-bot.centralus.azurecontainerapps.io`

### 7. Verify

```bash
curl https://salesforce-langgraph-ai-lead-bot.centralus.azurecontainerapps.io/health
```

---

## Option B: Deploy via GitHub Actions (Automated)

Create `.github/workflows/deploy.yml` in your repo:

```yaml
name: Deploy to Azure Container Apps

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'

env:
  RESOURCE_GROUP: rg-ai-lead-bot
  APP_NAME: salesforce-langgraph-ai-lead-bot

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Azure Login
        uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Deploy to Container Apps
        uses: azure/container-apps-deploy-action@v1
        with:
          resourceGroup: ${{ env.RESOURCE_GROUP }}
          containerAppName: ${{ env.APP_NAME }}
          appSourcePath: backend
          dockerfilePath: backend/Dockerfile
```

### GitHub Actions setup

1. Create a service principal:
   ```bash
   az ad sp create-for-rbac \
     --name "github-deploy-ai-lead-bot" \
     --role contributor \
     --scopes /subscriptions/<subscription-id>/resourceGroups/rg-ai-lead-bot \
     --json-auth
   ```

2. Copy the JSON output and add it as a GitHub repository secret named
   `AZURE_CREDENTIALS`.

3. Push to `main` — the workflow auto-deploys on changes to `backend/`.

---

## Azure Static Web Apps (Frontend Widget)

Deploy the widget files to Azure Static Web Apps from your private repo:

### 1. Create the Static Web App

```bash
az staticwebapp create \
  --name ai-lead-bot-widget \
  --resource-group $RESOURCE_GROUP \
  --source https://github.com/your-username/your-private-widget-repo \
  --location $LOCATION \
  --branch main \
  --app-location "/" \
  --output-location "" \
  --login-with-github
```

### 2. Get the widget URL

```bash
az staticwebapp show \
  --name ai-lead-bot-widget \
  --resource-group $RESOURCE_GROUP \
  --query "defaultHostname" \
  --output tsv
```

Returns: `your-widget-app.azurestaticapps.net`

### 3. Optional: Add custom domain

```bash
az staticwebapp hostname set \
  --name ai-lead-bot-widget \
  --resource-group $RESOURCE_GROUP \
  --hostname chat.markandrewmarquez.com
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

```bash
az containerapp logs show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --follow
```

### View metrics

```bash
az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
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
| SSE streaming not working | Ensure `--timeout-keep-alive 120` is set. Azure's load balancer may close idle connections. |
| Health check failing | The `/health` endpoint must respond within 5s. If the LLM init is slow, increase `--start-period`. |
