# Record-Triggered Flow: Lead Created → Agentforce

Specification for the Salesforce Flow that fires when a new Lead is created
by the AI chat bot, triggering the Agentforce agent for follow-up processing.

---

## Flow Overview

| Property | Value |
|---|---|
| Flow Label | `AI Chat Lead - Post-Creation Automation` |
| Flow API Name | `AI_Chat_Lead_Post_Creation` |
| Type | Record-Triggered Flow |
| Object | Lead |
| Trigger | After a record is created |
| Run Mode | System Context — without Sharing |
| Entry Conditions | `LeadSource EQUALS 'Web Chat' AND Lead_Score__c IS NOT NULL` |

---

## What This Flow Does

When the LangGraph backend creates a Lead with `LeadSource = "Web Chat"` and
a `Lead_Score__c` value, this flow automatically:

1. Invokes the Agentforce agent to analyze the lead and conversation transcript
2. The Agentforce agent then handles all downstream actions:
   - Assigns priority (High / Medium / Low)
   - Creates follow-up Tasks with due dates
   - Drafts a personalized follow-up email
   - Optionally creates an Opportunity for high-value leads
   - Updates Lead fields (Status, etc.)

---

## Entry Conditions (Detail)

All conditions must be true (AND):

| Field | Operator | Value |
|---|---|---|
| `LeadSource` | Equals | `Web Chat` |
| `Lead_Score__c` | Is Not Null | |

These conditions ensure the flow only fires for leads created by the chat bot
(not manually created leads or leads from other sources).

---

## Flow Elements

### Element 1: Get Related Task (Transcript)

| Property | Value |
|---|---|
| Type | Get Records |
| Object | Task |
| Filter | `WhoId EQUALS {!$Record.Id}` AND `Subject STARTS WITH 'AI Chat Transcript'` |
| Sort | CreatedDate, Descending |
| How Many | Only the first record |
| Store in | `varTranscriptTask` |

### Element 2: Decision — Transcript Found?

| Property | Value |
|---|---|
| Type | Decision |
| Outcome: Has Transcript | `varTranscriptTask IS NOT NULL` |
| Default Outcome | No Transcript (proceed anyway with limited context) |

### Element 3: Invoke Agentforce Agent

| Property | Value |
|---|---|
| Type | Action (Invoke Agent) |
| Agent | `Lead Qualification Follow-Up Agent` |
| Input: Lead ID | `{!$Record.Id}` |
| Input: Lead Score | `{!$Record.Lead_Score__c}` |
| Input: Budget Range | `{!$Record.Budget_Range__c}` |
| Input: Timeline | `{!$Record.Timeline__c}` |
| Input: Pain Points | `{!$Record.Pain_Points__c}` |
| Input: Company Size | `{!$Record.Company_Size__c}` |
| Input: Transcript | `{!varTranscriptTask.Description}` |
| Input: Lead Name | `{!$Record.FirstName} {!$Record.LastName}` |
| Input: Company | `{!$Record.Company}` |
| Input: Email | `{!$Record.Email}` |

### Element 4: Update Lead Status

| Property | Value |
|---|---|
| Type | Update Records |
| Record | `{!$Record}` |
| Field: Status | `Working - Contacted` |

> Note: If Agentforce is not yet available in your org, replace Element 3
> with an Apex Invocable Action that performs the follow-up logic, or use
> a Subflow with sequential automation elements.

---

## Fallback: Flow-Only Automation (No Agentforce)

If Agentforce is not available, replace the agent invocation with these
direct Flow elements:

### Alt Element 3a: Decision — Lead Priority

| Outcome | Condition |
|---|---|
| High Priority | `Lead_Score__c >= 70` |
| Medium Priority | `Lead_Score__c >= 40` |
| Low Priority | Default |

### Alt Element 3b: Create Follow-Up Tasks

**High Priority path** — create 3 Tasks:

| Task | Subject | Due Date | Priority |
|---|---|---|---|
| 1 | "Call {FirstName} to discuss requirements" | Today + 1 day | High |
| 2 | "Send relevant case study to {Email}" | Today + 2 days | Normal |
| 3 | "Schedule demo for {Company}" | Today + 3 days | High |

**Medium Priority path** — create 2 Tasks:

| Task | Subject | Due Date | Priority |
|---|---|---|---|
| 1 | "Follow up with {FirstName} at {Company}" | Today + 2 days | Normal |
| 2 | "Send overview materials to {Email}" | Today + 3 days | Normal |

**Low Priority path** — create 1 Task:

| Task | Subject | Due Date | Priority |
|---|---|---|---|
| 1 | "Nurture: Send info to {FirstName} at {Company}" | Today + 5 days | Low |

All Tasks: `WhoId = {!$Record.Id}`, `Status = "Not Started"`,
`OwnerId = {!$Record.OwnerId}` (or a queue ID for round-robin).

### Alt Element 3c: Create Opportunity (High Value Only)

Condition: `Lead_Score__c >= 80 AND Budget_Range__c IN ('$50K-$100K', '$100K+')`

| Field | Value |
|---|---|
| Name | `{Company} - Web Chat Lead` |
| StageName | `Qualification` |
| CloseDate | Based on Timeline__c (Immediate: +30 days, 1-3 months: +90 days, etc.) |
| Amount | Midpoint of budget range ($75K for $50K-$100K, $150K for $100K+) |
| LeadSource | `Web Chat` |
