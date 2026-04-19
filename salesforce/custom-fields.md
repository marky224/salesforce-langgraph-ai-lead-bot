# Custom Salesforce Fields

Custom fields required on the **Lead** object for the AI Sales Lead Bot.
Create these before deploying the backend — the Salesforce tool expects
these fields to exist when writing lead data.

---

## How to Create Custom Fields

1. **Setup** → search "Object Manager" → click **Object Manager**
2. Click **Lead**
3. Click **Fields & Relationships** (left sidebar)
4. Click **New** (top right)
5. Follow the field-specific instructions below
6. On the field-level security step, make the field **visible** to all profiles
   that need it (at minimum: System Administrator and your integration user's profile)
7. On the page layout step, add the field to your default Lead page layout

---

## Fields to Create on the Lead Object

### 1. Lead Score

| Setting | Value |
|---|---|
| Data Type | Number |
| Field Label | Lead Score |
| API Name | `Lead_Score__c` (auto-generated) |
| Length | 3 |
| Decimal Places | 0 |
| Description | AI-computed qualification score (0–100) |
| Help Text | Score from 0–100 based on budget, timeline, company size, decision-maker status, and pain points |
| Default Value | (leave blank) |

### 2. Budget Range

| Setting | Value |
|---|---|
| Data Type | Picklist |
| Field Label | Budget Range |
| API Name | `Budget_Range__c` |
| Values (one per line) | `Under $10K` |
| | `$10K-$50K` |
| | `$50K-$100K` |
| | `$100K+` |
| Sort | Use values in the order entered |
| Restrict to admin-defined values | Yes (checked) |

### 3. Timeline

| Setting | Value |
|---|---|
| Data Type | Picklist |
| Field Label | Timeline |
| API Name | `Timeline__c` |
| Values (one per line) | `Immediate` |
| | `1-3 months` |
| | `3-6 months` |
| | `6+ months` |
| | `Just exploring` |
| Sort | Use values in the order entered |
| Restrict to admin-defined values | Yes (checked) |

### 4. Pain Points

| Setting | Value |
|---|---|
| Data Type | Long Text Area |
| Field Label | Pain Points |
| API Name | `Pain_Points__c` |
| Length | 32000 |
| Visible Lines | 5 |
| Description | Semicolon-separated pain points captured from chat conversation |

### 5. Company Size

| Setting | Value |
|---|---|
| Data Type | Picklist |
| Field Label | Company Size |
| API Name | `Company_Size__c` |
| Values (one per line) | `1-10` |
| | `11-50` |
| | `51-200` |
| | `201-1000` |
| | `1000+` |
| Sort | Use values in the order entered |
| Restrict to admin-defined values | Yes (checked) |

### 6. Chat Transcript ID

| Setting | Value |
|---|---|
| Data Type | Text |
| Field Label | Chat Transcript ID |
| API Name | `Chat_Transcript_ID__c` |
| Length | 18 |
| Description | Salesforce Task ID of the linked chat transcript record |
| Help Text | Auto-populated by the AI chat bot when a transcript Task is created |
| External ID | No |
| Unique | No |

---

## Field Summary

| # | Field Label | API Name | Type | Purpose |
|---|---|---|---|---|
| 1 | Lead Score | `Lead_Score__c` | Number(3,0) | 0–100 qualification score |
| 2 | Budget Range | `Budget_Range__c` | Picklist | Budget bracket from conversation |
| 3 | Timeline | `Timeline__c` | Picklist | Purchase readiness timeline |
| 4 | Pain Points | `Pain_Points__c` | Long Text Area | Captured challenges |
| 5 | Company Size | `Company_Size__c` | Picklist | Employee count bracket |
| 6 | Chat Transcript ID | `Chat_Transcript_ID__c` | Text(18) | Link to transcript Task |

---

## Standard Fields Used (no creation needed)

These standard Lead fields are populated by the bot — they already exist:

| Field | API Name | How the bot uses it |
|---|---|---|
| First Name | `FirstName` | From lead capture conversation |
| Last Name | `LastName` | From lead capture conversation |
| Email | `Email` | From lead capture conversation |
| Company | `Company` | From lead capture conversation |
| Phone | `Phone` | From lead capture conversation (optional) |
| Title | `Title` | From lead capture conversation (optional) |
| Lead Source | `LeadSource` | Always set to "Web Chat" |
| Status | `Status` | Always set to "New" |
| Description | `Description` | AI-generated transcript summary |

---

## Standard Task Fields Used (no creation needed)

The transcript Task uses only standard fields:

| Field | API Name | Value |
|---|---|---|
| Subject | `Subject` | "AI Chat Transcript - {date}" |
| Related To (Who) | `WhoId` | Lead record ID |
| Description | `Description` | Full conversation transcript |
| Status | `Status` | "Completed" |
| Priority | `Priority` | "Normal" |
| Due Date | `ActivityDate` | Today's date |
| Type | `Type` | "Other" |

---

## Verification

After creating all fields, verify via the API:

```bash
# From your backend directory
python -c "
from simple_salesforce import Salesforce
sf = Salesforce(username='...', password='...', security_token='...', domain='login')
fields = sf.Lead.describe()['fields']
custom = [f['name'] for f in fields if f['name'].endswith('__c')]
print('Custom fields:', custom)
"
```

Expected output should include all six `__c` fields listed above.
