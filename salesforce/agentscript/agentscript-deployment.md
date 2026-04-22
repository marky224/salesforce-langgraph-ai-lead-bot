# Agent Script Deployment Guide — CLI-First Workflow

This guide replaces the click-heavy Salesforce UI approach with a
code-first workflow using Agent Script and the Salesforce CLI.
You'll author the agent as a `.agent` script file locally, publish
it to your org via CLI, preview it from the terminal, and activate
it — all without navigating through Setup screens.

**Prerequisites:**
- Your 4 Apex classes are already deployed and tested (12/12 pass)
- Your 4 Agent Actions are already registered in Agentforce Assets
- Your Permission Set is configured (object + field + Apex class access)
- Your Record-Triggered Flow exists (will be reactivated at the end)

---

## Step 1 — Install / Update Salesforce CLI + Agent Script Extension

### 1.1: Update Salesforce CLI

Agent Script support requires **CLI v2.113.6+**. The `agent` commands
were added in late 2025 and refined through early 2026.

```powershell
# Check your current version
sf --version

# Update to latest
sf update

# Verify the agent plugin is available
sf agent --help
```

You should see commands like `agent generate authoring-bundle`,
`agent publish authoring-bundle`, `agent preview`, `agent activate`, etc.

If `sf agent` isn't recognized, install the plugin:

```powershell
sf plugins install @salesforce/plugin-agent
```

### 1.2: Install VS Code Extensions (Optional but Recommended)

If you want syntax highlighting and validation in VS Code alongside PyCharm:

1. Install the **Salesforce Agentforce DX Extension** from the VS Code Marketplace
   - Search: "Agentforce DX" or "Agent Script"
   - Publisher: Salesforce
2. Install the **Agent Script Language** extension for syntax highlighting
   - Search: "Agent Script Language Client"

These give you colored syntax highlighting, real-time error indicators,
linting, and an Outline symbol tree for `.agent` files.

> **Note:** PyCharm doesn't have Agent Script support yet. For this
> specific workflow, VS Code is the better editor. You can keep PyCharm
> for your Python backend work and use VS Code just for the Agent Script
> files.

---

## Step 2 — Set Up the Agent Script Project

### 2.1: Create the Project Directory

You have two options:

**Option A — Add to your existing SFDX project** (recommended if you want
everything in one `salesforce/apex/` directory):

```powershell
# From your project root
cd salesforce\apex

# Create the authoring bundles directory
mkdir -p force-app\main\default\aiAuthoringBundles\Lead_Qualification_Follow_Up_Agent
```

Then copy the `.agent` and `.aiAuthoringBundle-meta.xml` files into that
directory.

**Option B — Use the standalone project** (provided in this guide):

```powershell
# From your project root
cd salesforce\agentscript
```

The directory structure is:

```
salesforce/agentscript/
├── sfdx-project.json
└── force-app/
    └── main/
        └── default/
            └── aiAuthoringBundles/
                └── Lead_Qualification_Follow_Up_Agent/
                    ├── Lead_Qualification_Follow_Up_Agent.agent
                    └── Lead_Qualification_Follow_Up_Agent.aiAuthoringBundle-meta.xml
```

### 2.2: Authenticate Your Org

```powershell
# Authenticate (opens browser — use your Agentforce Dev Edition credentials)
sf org login web --alias agentforce-dev --instance-url https://orgfarm-d025e11150-dev-ed.develop.my.salesforce.com

# Verify connection
sf org display --target-org agentforce-dev
```

---

## Step 3 — Deactivate the Existing Agent and Flow

Before publishing the new Agent Script version, deactivate the current
agent and flow. These CLI commands save you from clicking through Setup:

```powershell
# Deactivate the agent
sf agent deactivate --api-name Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev

# Deactivate the Flow (this one you'll need to do in Setup, or via
# Metadata API — the CLI doesn't have a direct flow deactivate command)
```

For the Flow, you'll need to either:
- Go to **Setup → Flows → Lead Created - Agentforce Follow-Up → Deactivate**
- Or deploy a Flow metadata file with `status: Draft` (more complex)

> **Note:** If the `sf agent deactivate` command isn't available in your
> CLI version, deactivate the agent from Setup → Agentforce Agents →
> Open in Builder → Deactivate. This is the one UI step you may need.

---

## Step 4 — Validate the Agent Script

Before publishing, validate that the script compiles:

```powershell
cd salesforce\agentscript    # or salesforce\apex if using Option A

# Validate the authoring bundle
sf agent validate authoring-bundle --target-org agentforce-dev
```

This compiles the `.agent` file and reports any syntax errors. Fix any
issues before proceeding.

Common validation errors:
- **Indentation mismatch** — Agent Script is whitespace-sensitive (like
  Python). Use consistent spaces, never mix tabs and spaces.
- **Unknown action target** — The Apex class name in `target: "apex://ClassName"`
  must match a deployed class. Your 4 classes are already deployed.
- **Duplicate developer_name** — If an agent with this `developer_name`
  already exists, the publish will create a new version (which is fine).

---

## Step 5 — Publish the Agent Script to Your Org

Publishing compiles the script and creates/updates the agent metadata:

```powershell
# Publish the authoring bundle
sf agent publish authoring-bundle --target-org agentforce-dev
```

If you're not prompted to select a bundle, specify it:

```powershell
sf agent publish authoring-bundle --name Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev
```

**What happens during publish:**
1. The `.agent` file is compiled and validated
2. Bot and GenAi* metadata is created (or a new version if the agent exists)
3. The metadata is retrieved back to your DX project
4. The authoring bundle metadata is deployed to your org

After publishing, you can see the agent (or new version) in Agentforce
Builder in your org.

> **Important:** Publishing creates the agent structure but does NOT
> activate it. You still need to activate in a separate step.

---

## Step 6 — Preview the Agent from CLI

Test the agent before activating it. The CLI preview supports two modes:

### Simulated Mode (No Org Data)

Good for testing the script structure and LLM reasoning without hitting
real Salesforce data:

```powershell
sf agent preview --authoring-bundle Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev
```

This opens an interactive chat in your terminal. Type the test message:

```
Process the web chat lead with ID 00Qg5000003P393EAC. The lead score is 85.
Analyze the lead, create follow-up tasks, draft a follow-up email, and
create an Opportunity if the lead qualifies.
```

In simulated mode, actions are mocked — the LLM simulates what each
action would do based on the action descriptions. Useful for verifying
the agent calls all 4 actions in the right order.

### Live Mode (Uses Real Org Data)

Tests with your actual Apex classes and Salesforce data:

```powershell
sf agent preview --authoring-bundle Lead_Qualification_Follow_Up_Agent --mode live --target-org agentforce-dev
```

Or preview a published (not local) agent:

```powershell
sf agent preview --api-name Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev
```

**What to verify in the preview:**
- ✅ Agent extracts Lead ID and score from the message
- ✅ Calls Update Lead → sets Rating to "Hot" (score 85)
- ✅ Calls Create Tasks → creates 3 tasks
- ✅ Calls Draft Email → composes personalized email
- ✅ Calls Create Opportunity → creates Opp (score 85 ≥ 80)
- ✅ Provides a summary of all actions taken
- ✅ Does NOT ask "would you like me to continue?"

Type `exit` or `Ctrl+C` to end the preview session.

---

## Step 7 — Activate the Agent

Once the preview confirms the agent works correctly:

```powershell
# Activate the agent
sf agent activate --api-name Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev
```

If this command isn't available, activate via Setup:
1. **Setup** → **Agentforce Agents** → click the agent name
2. Click **Open in Builder** → click **Activate**
3. If configuration warnings appear about channels/Data Library, click
   **Ignore & Activate** (same as before — these don't apply to
   Flow-triggered agents)

---

## Step 8 — Reactivate the Flow

The Record-Triggered Flow doesn't change — it sends the same message
to the agent. Just reactivate it:

1. **Setup** → Quick Find → **Flows**
2. Click **Lead Created - Agentforce Follow-Up**
3. Click **Activate**

The Flow's agent request message stays the same:
```
Process the web chat lead with ID {!$Record.Id}. The lead score is
{!$Record.Lead_Score__c}. Analyze the lead, create follow-up tasks,
draft a follow-up email, and create an Opportunity if the lead qualifies.
```

---

## Step 9 — End-to-End Verification

Run a fresh chat conversation to create a new Lead and verify the full
pipeline:

### 9.1: Run the Chatbot

```powershell
# Terminal 1 — Backend
cd backend
uvicorn app.server:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
python -m http.server 3000
```

Use the test conversation script to create a high-value lead (score 80+,
budget $50K-$100K).

### 9.2: Verify Results

Wait 30–60 seconds for the async Flow to fire, then check the Lead
record in Salesforce:

- [ ] **Rating** = "Hot"
- [ ] **Status** = "Working - Contacted"
- [ ] **Description** has "--- Agent Analysis ---" section appended
- [ ] **Activity History** has 3 follow-up Tasks + 1 email draft Task
- [ ] **Opportunity** created with Stage = "Qualification"

### 9.3: Check Agent Logs

```powershell
# Open Agentforce Studio to check agent analytics
sf org open authoring-bundle --target-org agentforce-dev
```

Or go to **Setup → Agent Analytics** to review the session trace and
verify all 4 actions were called in a single turn.

---

## Making Changes to the Agent

The beauty of Agent Script: edit the `.agent` file locally, validate,
publish, done. No clicking through Setup menus.

### Edit → Validate → Publish Workflow

```powershell
# 1. Edit the .agent file in VS Code (or any editor)
code salesforce/agentscript/force-app/main/default/aiAuthoringBundles/Lead_Qualification_Follow_Up_Agent/Lead_Qualification_Follow_Up_Agent.agent

# 2. Validate
sf agent validate authoring-bundle --target-org agentforce-dev

# 3. Deactivate current version (required before publishing updates)
sf agent deactivate --api-name Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev

# 4. Publish new version
sf agent publish authoring-bundle --target-org agentforce-dev

# 5. Preview to verify changes
sf agent preview --authoring-bundle Lead_Qualification_Follow_Up_Agent --mode live --target-org agentforce-dev

# 6. Activate
sf agent activate --api-name Lead_Qualification_Follow_Up_Agent --target-org agentforce-dev
```

### Retrieve Changes Made in the UI

If you make changes in the Agentforce Builder UI and want to sync them
back to your local project:

```powershell
# Retrieve all authoring bundles
sf project retrieve start --metadata AiAuthoringBundle --target-org agentforce-dev

# Or retrieve a specific bundle with all its versions
sf project retrieve start --metadata "AiAuthoringBundle:Lead_Qualification_Follow_Up_Agent*" --target-org agentforce-dev
```

---

## Troubleshooting

### "sf agent" commands not found

Update your CLI: `sf update`. If still missing, install the plugin:
`sf plugins install @salesforce/plugin-agent`

### Validation error: "Unknown action target"

The Apex class referenced in `target: "apex://ClassName"` must be deployed
to the org. Verify with:

```powershell
sf apex list log --target-org agentforce-dev
# Or check via SOQL
sf data query --query "SELECT Name FROM ApexClass WHERE Name IN ('UpdateLeadFields','CreateFollowUpTasks','DraftFollowUpEmail','CreateLeadOpportunity')" --target-org agentforce-dev
```

### Publish fails with "agent already exists"

This is expected — the publish creates a new VERSION of the existing agent.
Make sure the existing agent is deactivated first.

### Agent still only runs 1 action in preview

Check the reasoning instructions. The LLM instructions must explicitly
tell the agent to call all 4 tools in sequence. Also verify the
`available when` condition on `create_opportunity` — if the lead_score
variable isn't set yet when the agent first loads, the tool may be hidden.
You may need to remove the `available when` clause and let the
instructions handle the conditional logic.

### Agent Script not available in your org

Agent Script requires the new Agentforce Builder, which is available in
Developer Edition orgs as of TDX 2026 (April 2026). If your org was
created before March 2026, you may need to create a new Developer Edition
org at https://developer.salesforce.com/signup to get the latest features.

---

## File Reference

| File | Purpose |
|---|---|
| `Lead_Qualification_Follow_Up_Agent.agent` | Agent Script — the full agent definition |
| `Lead_Qualification_Follow_Up_Agent.aiAuthoringBundle-meta.xml` | Metadata wrapper for the authoring bundle |
| `sfdx-project.json` | SFDX project configuration |
| `agentscript-deployment.md` | This deployment guide |
