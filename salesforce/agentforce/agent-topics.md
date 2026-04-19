# Agentforce Agent Topics & Actions

Topic and action definitions for the **Lead Qualification Follow-Up Agent**.
Each topic is a distinct capability the agent can invoke. Configure these
in the Agentforce agent's Topics section in Salesforce Setup.

---

## Topic 1: Lead Analysis

| Property | Value |
|---|---|
| Topic Label | Lead Analysis |
| Topic Description | Analyze a newly created web chat Lead by reading the Lead record and associated conversation transcript. Identify buying signals, objections, pain points, and urgency level. Determine priority classification. |
| Scope | This topic reads Lead fields and the related Task containing the transcript. It does not create or modify any records. |

### Instructions for this Topic

```
Read the Lead record and find the associated Task where Subject starts
with "AI Chat Transcript". Parse the transcript for:
- Buying signals: urgency words, specific requirements, mentions of budget
- Objections: concerns about cost, timing, past failures, complexity
- Pain points: what problems they described and their business impact
- Goals: what outcomes they want to achieve
- Decision context: are they the decision maker, who else is involved

Classify the lead priority:
- High: score >= 70, OR score 60-69 with strong urgency signals
- Medium: score 40-69 with neutral signals
- Low: score < 40, OR score 40-49 with passive/exploratory signals

Output a structured analysis with: priority, key_pain_points (list),
buying_signals (list), objections (list), recommended_actions (list),
and estimated_deal_value.
```

### Actions

| Action | Type | Description |
|---|---|---|
| Read Lead Record | Record Query | Query the Lead record by ID, return all fields including custom fields |
| Read Transcript Task | Record Query | Query Task where WhoId = Lead ID AND Subject STARTS WITH 'AI Chat Transcript', return Description field |

---

## Topic 2: Follow-Up Task Creation

| Property | Value |
|---|---|
| Topic Label | Follow-Up Task Creation |
| Topic Description | Create prioritized follow-up Tasks assigned to the Lead owner based on the lead analysis. Task count and urgency scale with lead priority. Task descriptions reference specific details from the conversation. |
| Scope | Creates 1-3 Task records linked to the Lead. |

### Instructions for this Topic

```
Create follow-up Tasks based on the lead priority determined by the
Lead Analysis topic.

High Priority (3 tasks):
- Immediate call task (due tomorrow, High priority)
- Send relevant materials task (due in 2 days, Normal priority)  
- Schedule demo/meeting task (due in 3 days, High priority)

Medium Priority (2 tasks):
- Follow-up contact task (due in 2 days, Normal priority)
- Send overview materials task (due in 3 days, Normal priority)

Low Priority (1 task):
- Nurture/send info task (due in 5 days, Low priority)

CRITICAL: Every Task description MUST reference specific details from
the conversation. Include the prospect's name, company, and at least
one specific pain point or goal they mentioned. Generic task descriptions
provide no value to the sales rep picking up this lead.

All Tasks:
- WhoId = Lead ID
- OwnerId = Lead OwnerId (or default queue)
- Status = "Not Started"
- Type = "Other" (or "Call" for phone tasks)
```

### Actions

| Action | Type | Description |
|---|---|---|
| Create Task | Record Create | Create a Task record with Subject, Description, WhoId, OwnerId, Priority, Status, ActivityDate, Type |

---

## Topic 3: Email Drafting

| Property | Value |
|---|---|
| Topic Label | Email Drafting |
| Topic Description | Draft a personalized follow-up email referencing specific pain points from the conversation transcript. Store as a Task with Type = Email linked to the Lead. |
| Scope | Creates one Task record of Type "Email" with the draft email in the Description field. |

### Instructions for this Topic

```
Draft a follow-up email for the prospect. The email must:

1. Subject line: Professional, specific, no exclamation marks.
   Good: "Following up on your CRM automation inquiry"
   Bad: "Great chatting with you!"

2. Opening: Reference something specific they said in the conversation.
   Good: "You mentioned your team spends about 10 hours a week on manual
   data entry — that's a problem we've helped several companies solve."
   Bad: "Thanks for chatting with us!"

3. Body: Connect their pain point to a relevant solution or resource.
   Keep it to 2-3 short paragraphs. Mention their timeline context
   naturally ("Since you're looking to make a change in the next
   1-3 months...").

4. Call to action: One clear next step.
   Good: "Would you have 20 minutes this week for a quick call? I can
   walk you through how [specific solution] addresses [their pain point]."
   Bad: "Let me know if you have any questions!"

5. Sign-off: Professional, warm.
   "Best regards," or "Looking forward to connecting,"

Total length: 150-250 words.

Store the email as a Task:
- Subject: "Draft follow-up email: [Email Subject Line]"
- Description: Full email text (Subject + Body)
- Type: "Email"
- Status: "Not Started" (sales rep reviews before sending)
- Priority: matches the lead priority
- WhoId: Lead ID
- ActivityDate: today
```

### Actions

| Action | Type | Description |
|---|---|---|
| Create Email Task | Record Create | Create a Task with Type = "Email", containing the draft email in Description |

---

## Topic 4: Opportunity Creation

| Property | Value |
|---|---|
| Topic Label | Opportunity Creation |
| Topic Description | Create an Opportunity record for high-value leads that meet both score and budget thresholds. Only triggers when Lead_Score__c >= 80 AND Budget_Range__c is $50K or above. |
| Scope | Conditionally creates one Opportunity record. |

### Instructions for this Topic

```
ONLY create an Opportunity if BOTH conditions are met:
- Lead_Score__c >= 80
- Budget_Range__c is "$50K-$100K" or "$100K+"

If conditions are not met, do nothing and skip this topic.

Opportunity fields:
- Name: "[Company] — Web Chat Inquiry"
- StageName: "Qualification"
- LeadSource: "Web Chat"
- CloseDate: Calculate from Timeline__c:
  * "Immediate" → today + 30 days
  * "1-3 months" → today + 90 days
  * "3-6 months" → today + 180 days
  * "6+ months" → today + 365 days
  * "Just exploring" or unknown → today + 365 days
- Amount: Midpoint of budget range:
  * "$50K-$100K" → 75000
  * "$100K+" → 150000
- Description: One-paragraph summary including the prospect's primary
  pain point, their current solution, and what they're looking for.
  Pull this from the conversation transcript.

After creating the Opportunity, update the Lead:
- Status: "Qualified"
```

### Actions

| Action | Type | Description |
|---|---|---|
| Create Opportunity | Record Create | Create an Opportunity with calculated fields |
| Update Lead Status | Record Update | Set Lead Status to "Qualified" |

---

## Topic Summary

| # | Topic | Records Created | Trigger Condition |
|---|---|---|---|
| 1 | Lead Analysis | None (read-only) | Always runs first |
| 2 | Follow-Up Task Creation | 1-3 Tasks | Always (count scales with priority) |
| 3 | Email Drafting | 1 Task (Type: Email) | Always |
| 4 | Opportunity Creation | 0-1 Opportunity | Only if score >= 80 AND budget >= $50K |
