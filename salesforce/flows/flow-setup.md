# Flow Creation — Step-by-Step

How to build the Record-Triggered Flow in Salesforce that fires when the
AI chat bot creates a Lead.

---

## Step 1: Open Flow Builder

1. **Setup** → search "Flows" → click **Flows**
2. Click **New Flow**
3. Select **Record-Triggered Flow** → **Create**

---

## Step 2: Configure the Trigger

The Start element opens automatically:

1. **Object**: Select **Lead**
2. **Trigger the Flow When**: A record is **created**
3. **Entry Conditions**:
   - Condition Requirements: **All Conditions Are Met (AND)**
   - Row 1: `LeadSource` | Equals | `Web Chat`
   - Row 2: `Lead_Score__c` | Is Null | `False`
4. **Optimize the Flow For**: **Actions and Related Records**
5. **When to Run**: Select **After the record is saved**
   (we need the Lead ID to exist before querying related Tasks)
6. Click **Done**

---

## Step 3: Add "Get Records" — Fetch the Transcript Task

1. Click the **+** below the Start element → **Get Records**
2. Configure:
   - **Label**: `Get Transcript Task`
   - **API Name**: `Get_Transcript_Task`
   - **Object**: Task
   - **Filter Conditions**:
     - `WhoId` | Equals | `{!$Record.Id}`
     - `Subject` | Starts With | `AI Chat Transcript`
   - **Sort**: CreatedDate, Descending
   - **How Many Records**: Only the first record
   - **Store**: Automatically store all fields
3. Click **Done**

---

## Step 4: Add Decision — Was a Transcript Found?

1. Click **+** → **Decision**
2. Configure:
   - **Label**: `Transcript Found?`
   - **Outcome 1 Label**: `Has Transcript`
   - **Condition**: `{!Get_Transcript_Task}` | Is Null | `False`
   - **Default Outcome Label**: `No Transcript`
3. Click **Done**

---

## Step 5: Add Decision — Lead Priority

On the **Has Transcript** path (and also connect **No Transcript** here):

1. Click **+** → **Decision**
2. Configure:
   - **Label**: `Lead Priority`
   - **Outcome 1**: `High Priority` — `{!$Record.Lead_Score__c}` >= `70`
   - **Outcome 2**: `Medium Priority` — `{!$Record.Lead_Score__c}` >= `40`
   - **Default Outcome**: `Low Priority`
3. Click **Done**

---

## Step 6: Add "Create Records" — Follow-Up Tasks

### High Priority Path

1. Click **+** on the High Priority path → **Create Records**
2. **Label**: `Create High Priority Tasks`
3. **How Many**: Multiple
4. **Record 1**:
   - Object: Task
   - Subject: `Call {!$Record.FirstName} to discuss requirements`
   - WhoId: `{!$Record.Id}`
   - OwnerId: `{!$Record.OwnerId}`
   - Status: `Not Started`
   - Priority: `High`
   - ActivityDate: `{!$Flow.CurrentDate}` + 1 day (use a formula resource)
5. **Record 2**:
   - Subject: `Send case study to {!$Record.Email}`
   - Priority: `Normal`
   - ActivityDate: `{!$Flow.CurrentDate}` + 2 days
6. **Record 3**:
   - Subject: `Schedule demo for {!$Record.Company}`
   - Priority: `High`
   - ActivityDate: `{!$Flow.CurrentDate}` + 3 days

> **Tip**: Create a Formula resource for due dates:
> - Name: `DueDatePlus1` / Type: Date / Formula: `{!$Flow.CurrentDate} + 1`
> - Repeat for +2, +3, +5

### Medium Priority Path

1. Click **+** → **Create Records**
2. Create 2 Tasks with Normal priority, due +2 and +3 days

### Low Priority Path

1. Click **+** → **Create Records**
2. Create 1 Task with Low priority, due +5 days

---

## Step 7: Add "Update Records" — Set Lead Status

After all three priority paths, add an Update element (connect all paths to it):

1. Click **+** → **Update Records**
2. **Label**: `Update Lead Status`
3. **Record**: Use the Lead record that triggered the flow
4. **Field**: `Status` → `Working - Contacted`
5. Click **Done**

---

## Step 8: Add Opportunity Creation (High Priority Only)

On the **High Priority** path, before the status update:

1. Click **+** → **Decision**
2. **Label**: `Create Opportunity?`
3. **Outcome**: `Yes` — conditions (AND):
   - `{!$Record.Lead_Score__c}` >= `80`
   - `{!$Record.Budget_Range__c}` In `$50K-$100K, $100K+`
4. On the **Yes** path → **Create Records**:
   - Object: Opportunity
   - Name: `{!$Record.Company} - Web Chat Lead`
   - StageName: `Qualification`
   - CloseDate: `{!$Flow.CurrentDate}` + 90
   - Amount: `75000` (adjust based on budget range with a formula)
   - LeadSource: `Web Chat`

---

## Step 9: Save and Activate

1. Click **Save**
2. **Flow Label**: `AI Chat Lead - Post-Creation Automation`
3. **Flow API Name**: `AI_Chat_Lead_Post_Creation`
4. Click **Save**
5. Click **Activate**

---

## Step 10: Test

1. Start your backend locally
2. Complete a full chat conversation through the widget
3. Verify in Salesforce:
   - A new Lead exists with `LeadSource = Web Chat` and a `Lead_Score__c` value
   - Follow-up Tasks were auto-created with correct due dates and priorities
   - Lead Status changed to `Working - Contacted`
   - If score >= 80 and budget >= $50K: an Opportunity was created

---

## Debugging Tips

- **Setup** → **Flows** → find your flow → **Run History** shows each execution
  with input/output values and any errors
- If the flow doesn't fire, check:
  - Is the flow **Active**?
  - Does the Lead have `LeadSource = 'Web Chat'`?
  - Is `Lead_Score__c` populated (not null)?
- Enable **Debug Log** on the integration user to see flow execution in real time
