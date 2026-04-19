# Salesforce Connected App & OAuth Setup

Step-by-step guide to creating a Connected App in your Salesforce Developer
Edition org for server-to-server authentication from the LangGraph backend.

---

## Prerequisites

- A **Salesforce Developer Edition** org (free at [developer.salesforce.com](https://developer.salesforce.com/signup))
- System Administrator access to the org
- The custom fields from `custom-fields.md` already created (do that first)

---

## Step 1: Create a Connected App

1. Go to **Setup** → search "App Manager" → click **App Manager**
2. Click **New Connected App** (top right)
3. Fill in the basics:

   | Field | Value |
   |---|---|
   | Connected App Name | `AI Lead Bot` |
   | API Name | `AI_Lead_Bot` (auto-fills) |
   | Contact Email | Your email |

4. Under **API (Enable OAuth Settings)**, check **Enable OAuth Settings**
5. Set **Callback URL** to `https://login.salesforce.com/services/oauth2/callback`
6. Under **Selected OAuth Scopes**, add:
   - `Full access (full)`
   - `Manage user data via APIs (api)`
   - `Perform requests at any time (refresh_token, offline_access)`
7. Check **Enable Client Credentials Flow**
8. Uncheck **Require Proof Key for Code Exchange (PKCE)**
9. Click **Save**, then **Continue**

> After saving, Salesforce takes 2–10 minutes to activate the Connected App.

---

## Step 2: Retrieve Consumer Key & Secret

1. Go back to **App Manager**
2. Find **AI Lead Bot** → dropdown arrow → **View**
3. Under **API (Enable OAuth Settings)**, click **Manage Consumer Details**
4. Salesforce sends a verification code to your email — enter it
5. Copy:
   - **Consumer Key** → `SF_CLIENT_ID`
   - **Consumer Secret** → `SF_CLIENT_SECRET`

---

## Step 3: Configure Connected App Policies

1. From the Connected App detail page, click **Manage** → **Edit Policies**
2. Set:

   | Setting | Value |
   |---|---|
   | Permitted Users | Admin approved users are pre-authorized |
   | IP Relaxation | Relax IP restrictions |
   | Refresh Token Policy | Refresh token is valid until revoked |

3. Click **Save**

---

## Step 4: Assign a Run-As User

The Client Credentials Flow needs a designated Run As user.

1. From the Connected App **Manage** page, scroll to **Client Credentials Flow**
2. Click **Edit** next to "Run As"
3. Select your **System Administrator** user (or a dedicated integration user)
4. Click **Save**

For production, consider creating a dedicated integration user with a custom
profile limited to Lead and Task CRUD + API access.

---

## Step 5: Assign the Connected App to a Profile

1. From the Connected App **Manage** page → **Profiles** → **Manage Profiles**
2. Check **System Administrator** (and the integration user's profile if different)
3. Click **Save**

---

## Step 6: Get Your Security Token

1. Log in as the integration user → avatar → **Settings**
2. **My Personal Information** → **Reset My Security Token**
3. Check email for the new token → `SF_SECURITY_TOKEN`

> If "Reset My Security Token" isn't visible, IP restrictions are set to
> trusted ranges only. You can leave `SF_SECURITY_TOKEN` empty and rely
> on IP relaxation from Step 3.

---

## Step 7: Find Your Instance URL

Visible in the browser address bar when logged in:
- `https://your-domain.develop.my.salesforce.com` (Developer Edition)

This is your `SF_INSTANCE_URL`.

---

## Step 8: Set Environment Variables

```env
SF_INSTANCE_URL=https://your-domain.develop.my.salesforce.com
SF_CLIENT_ID=<Consumer Key>
SF_CLIENT_SECRET=<Consumer Secret>
SF_USERNAME=<Integration user email>
SF_PASSWORD=<Integration user password>
SF_SECURITY_TOKEN=<Token, or empty if IP relaxed>
```

---

## Step 9: Verify

```bash
curl http://localhost:8000/health/salesforce
```

Expected:
```json
{
  "connected": true,
  "instance_url": "your-domain.develop.my.salesforce.com",
  "api_calls_remaining": 14950,
  "api_calls_max": 15000
}
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `INVALID_CLIENT_ID` | Consumer Key wrong or app not yet activated (wait 10 min) |
| `INVALID_CLIENT` | Client Credentials Flow not enabled or Run As user not set |
| `INVALID_GRANT` | Password or security token is wrong |
| `API_DISABLED_FOR_ORG` | Your edition lacks API access (Developer Edition has it) |

---

## Security Notes

- Never commit `.env` to Git — use Azure Container Apps environment variables for production
- Client Credentials Flow is preferred over username-password for production
- Rotate the Consumer Secret periodically
- Integration user should have minimum permissions: Lead (CRUD), Task (CRUD)
