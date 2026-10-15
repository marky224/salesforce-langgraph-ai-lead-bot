// @ts-check
/**
 * widget.js — AI Sales Lead Bot for markandrewmarquez.com
 *
 * Drop this file + chat-widget.css into your site's file structure
 * and add the script tag before </body> on any page you want the
 * chat bubble to appear.
 */

// ---------------------------------------------------------------------------
// Types (JSDoc — checked by `npm run typecheck` in widget-typecheck/)
// ---------------------------------------------------------------------------

/**
 * Backend `ConversationStage` enum `.value` strings (backend/app/models/schemas.py).
 * @typedef {'greeting' | 'discovery' | 'qualification' | 'objection_handling'
 *   | 'lead_capture' | 'confirmation' | 'complete'} ConversationStage
 */

/**
 * Contact fields collected so far (backend `LeadData`); any may be missing or null.
 * @typedef {Object} LeadData
 * @property {string | null} [first_name]
 * @property {string | null} [last_name]
 * @property {string | null} [email]
 * @property {string | null} [company]
 * @property {string | null} [phone]
 * @property {string | null} [title]
 */

/**
 * One `data:` line from POST /chat/stream (server.py `event_generator`).
 * Token lines carry only `token`; an error line only `error`; the final
 * line carries `done` plus the rest.
 * @typedef {Object} SseEvent
 * @property {string} [token]
 * @property {string} [error]
 * @property {boolean} [done]
 * @property {string} [thread_id]
 * @property {LeadData} [lead_data]
 * @property {ConversationStage} [stage]
 * @property {boolean} [is_complete]
 * @property {string | null} [lead_id]
 */

/**
 * Reply from POST /chat/init (the fields the widget reads).
 * @typedef {Object} ChatInitReply
 * @property {string} thread_id
 * @property {string} greeting
 */

/**
 * @typedef {Object} Persona
 * @property {string} name
 * @property {string} avatar
 */

/**
 * @typedef {Object} WidgetConfig
 * @property {string} backendUrl
 * @property {Persona & { tagline: string }} assistant
 * @property {Persona} user
 * @property {string} cssPath
 */

/**
 * nlux streaming-adapter observer (the methods the adapter calls).
 * @typedef {Object} StreamObserver
 * @property {(chunk: string) => void} next
 * @property {(error: Error) => void} error
 * @property {() => void} complete
 */

/**
 * @typedef {Object} ChatWidgetApi
 * @property {() => void} init
 * @property {() => void} toggle
 * @property {() => string | null} getThreadId
 */

/**
 * `window` plus the globals the host page sets and the widget exposes.
 * @typedef {Window & typeof globalThis & {
 *   CHAT_BACKEND_URL?: string,
 *   CHAT_CSS_PATH?: string,
 *   __chatWidget: ChatWidgetApi,
 * }} WidgetWindow
 */

const widgetWindow = /** @type {WidgetWindow} */ (window);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** @type {WidgetConfig} */
const CONFIG = {
  backendUrl: widgetWindow.CHAT_BACKEND_URL || 'http://localhost:8000',

  assistant: {
    name: 'TARS',
    tagline: 'AI Solutions Advisor',
    avatar: 'https://zealous-moss-0360b7210.7.azurestaticapps.net/tars-avatar.svg',
  },

  user: {
    name: 'You',
    avatar: 'https://api.dicebear.com/9.x/initials/svg?seed=You&backgroundColor=475569',
  },

  // CSS path — auto-resolved relative to this script's origin so it
  // loads from Azure Static Web Apps even when embedded on GitHub Pages
  get cssPath() {
    if (widgetWindow.CHAT_CSS_PATH) return widgetWindow.CHAT_CSS_PATH;
    try {
      const scriptUrl = new URL(import.meta.url);
      return `${scriptUrl.origin}/chat-widget.css`;
    } catch {
      return '/chat-widget.css';
    }
  },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** @type {string | null} */
let threadId = null;
let chatInstance = null;
let isOpen = false;
let isInitialised = false;

// ---------------------------------------------------------------------------
// Custom SSE Streaming Adapter
// ---------------------------------------------------------------------------

function createCustomAdapter() {
  return {
    /**
     * @param {string} message
     * @param {StreamObserver} observer
     */
    streamText: async (message, observer) => {
      try {
        const response = await fetch(`${CONFIG.backendUrl}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: message,
            thread_id: threadId,
          }),
        });

        if (!response.ok) {
          observer.error(new Error(`Server returned ${response.status}`));
          return;
        }

        // Thrown into the catch below, so it is logged and reported like any stream error.
        if (!response.body) {
          throw new Error('Chat stream response has no body');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            /** @type {SseEvent} */
            let data;
            try {
              data = JSON.parse(line.slice(6));
            } catch (parseErr) {
              console.debug('Skipping SSE line:', line);
              continue;
            }
            if (!data || typeof data !== 'object') {
              console.debug('Skipping SSE line:', line);
              continue;
            }

            if (data.token) {
              observer.next(data.token);
            }

            if (data.error) {
              observer.error(new Error(data.error));
              return;
            }

            if (data.done) {
              if (data.thread_id) {
                threadId = data.thread_id;
              }
              // UI side-effects must not kill the stream, but must not be silent either.
              if (data.lead_data) {
                try {
                  updateLeadFieldsPanel(data.lead_data, data.stage);
                } catch (uiErr) {
                  console.error('Lead fields panel update failed:', uiErr);
                }
              }
              if (data.is_complete) {
                try {
                  handleConversationComplete(data.lead_id);
                } catch (uiErr) {
                  console.error('Conversation complete handler failed:', uiErr);
                }
              }
            }
          }
        }

        observer.complete();
      } catch (err) {
        console.error('Chat stream error:', err);
        observer.error(/** @type {Error} */ (err));
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Chat initialisation
// ---------------------------------------------------------------------------

async function initConversation() {
  try {
    const response = await fetch(`${CONFIG.backendUrl}/chat/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Init failed: ${response.status}`);
    }

    /** @type {ChatInitReply} */
    const data = await response.json();
    threadId = data.thread_id;
    return data.greeting;
  } catch (err) {
    console.error('Failed to initialise conversation:', err);
    return (
      "Hey there! I'm TARS, an AI solutions advisor working with Mark. " +
      "Whether you need help with IT automation, data engineering, " +
      "cybersecurity, or AI-powered workflows — I'd love to hear " +
      "what you're working on."
    );
  }
}

// ---------------------------------------------------------------------------
// Chat widget mount
// ---------------------------------------------------------------------------

async function mountChat() {
  const { createAiChat } = await import(
    // @ts-expect-error -- CDN ESM URL; no local type declarations for @nlux/core.
    'https://cdn.jsdelivr.net/npm/@nlux/core@2.17.1/+esm'
  );

  const greeting = await initConversation();
  const adapter = createCustomAdapter();

  chatInstance = createAiChat()
    .withAdapter(adapter)
    .withPersonaOptions({
      assistant: {
        name: CONFIG.assistant.name,
        tagline: CONFIG.assistant.tagline,
        avatar: CONFIG.assistant.avatar,
      },
      user: {
        name: CONFIG.user.name,
        avatar: CONFIG.user.avatar,
      },
    })
    .withInitialConversation([
      {
        role: 'assistant',
        message: greeting,
      },
    ])
    .withConversationOptions({
      layout: 'bubbles',
      scrollWhenGenerating: true,
    })
    .withComposerOptions({
      placeholder: 'Ask me about IT solutions...',
      autoFocus: true,
    })
    .withDisplayOptions({
      colorScheme: 'dark',
      width: '100%',
      height: '100%',
    });

  const container = document.getElementById('mm-chat-container');
  chatInstance.mount(container);
  isInitialised = true;
}

// ---------------------------------------------------------------------------
// Floating bubble UI
// ---------------------------------------------------------------------------

function toggleChat() {
  const panel = document.getElementById('mm-chat-panel');
  const bubble = document.getElementById('mm-chat-bubble');
  if (!panel || !bubble) {
    console.error('Chat widget: #mm-chat-panel or #mm-chat-bubble not found; cannot toggle chat.');
    return;
  }
  const bubbleIcon = /** @type {HTMLElement | null} */ (bubble.querySelector('.bubble-icon'));
  const closeIcon = /** @type {HTMLElement | null} */ (bubble.querySelector('.close-icon'));
  if (!bubbleIcon || !closeIcon) {
    console.error('Chat widget: bubble icons (.bubble-icon / .close-icon) not found; cannot toggle chat.');
    return;
  }

  isOpen = !isOpen;

  if (isOpen) {
    panel.classList.add('open');
    bubble.classList.add('active');
    bubbleIcon.style.display = 'none';
    closeIcon.style.display = 'block';

    if (!isInitialised) {
      mountChat();
    }
  } else {
    panel.classList.remove('open');
    bubble.classList.remove('active');
    bubbleIcon.style.display = 'block';
    closeIcon.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// Lead fields panel — shows what contact info has been collected so far
// ---------------------------------------------------------------------------

/** @type {Set<ConversationStage>} */
const LEAD_CAPTURE_STAGES = new Set([
  'lead_capture',
  'confirmation',
  'complete',
]);

/**
 * @param {LeadData} leadData
 * @param {ConversationStage} [stage]
 */
function updateLeadFieldsPanel(leadData, stage) {
  const panel = document.getElementById('mm-lead-fields');
  if (!panel) return;

  // Show the panel once we've reached lead_capture (or later), OR whenever
  // any contact field has been captured — whichever comes first.
  const hasAnyValue =
    leadData &&
    (leadData.first_name ||
      leadData.last_name ||
      leadData.title ||
      leadData.company ||
      leadData.email ||
      leadData.phone);

  const shouldShow =
    (stage !== undefined && LEAD_CAPTURE_STAGES.has(stage)) || hasAnyValue;

  if (!shouldShow) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';

  const fullName = [leadData.first_name, leadData.last_name]
    .filter(Boolean)
    .join(' ');

  /**
   * @param {string} id
   * @param {string | null | undefined} value
   */
  const set = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (value) {
      el.textContent = value;
      el.classList.remove('empty');
    } else {
      el.textContent = '';
      el.classList.add('empty');
    }
  };

  set('mm-lf-name', fullName);
  set('mm-lf-title', leadData.title);
  set('mm-lf-company', leadData.company);
  set('mm-lf-email', leadData.email);
  set('mm-lf-phone', leadData.phone);
}

// ---------------------------------------------------------------------------
// Conversation completion handler
// ---------------------------------------------------------------------------

/** @param {string | null | undefined} leadId */
function handleConversationComplete(leadId) {
  console.info('Conversation complete. Lead ID:', leadId);

  setTimeout(() => {
    const banner = document.getElementById('mm-completion-banner');
    if (banner) {
      banner.style.display = 'flex';
    }
  }, 2000);
}

// ---------------------------------------------------------------------------
// Inject DOM elements
// ---------------------------------------------------------------------------

function injectWidget() {
  // Load nlux Nova theme CSS
  const themeLink = document.createElement('link');
  themeLink.rel = 'stylesheet';
  themeLink.href = 'https://cdn.jsdelivr.net/npm/@nlux/themes@2.17.1/nova.css';
  document.head.appendChild(themeLink);

  // Load widget styles
  const styleLink = document.createElement('link');
  styleLink.rel = 'stylesheet';
  styleLink.href = CONFIG.cssPath;
  document.head.appendChild(styleLink);

  // Create the chat panel
  const panel = document.createElement('div');
  panel.id = 'mm-chat-panel';
  panel.innerHTML = `
    <div class="mm-chat-header">
      <div class="mm-chat-header-info">
        <img src="${CONFIG.assistant.avatar}" alt="${CONFIG.assistant.name}" class="mm-header-avatar" />
        <div>
          <div class="mm-header-name">${CONFIG.assistant.name}</div>
          <div class="mm-header-tagline">${CONFIG.assistant.tagline}</div>
        </div>
      </div>
      <button class="mm-chat-minimize" onclick="window.__chatWidget.toggle()" aria-label="Minimize chat">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="5" y1="12" x2="19" y2="12"></line>
        </svg>
      </button>
    </div>
    <div id="mm-chat-container"></div>
    <div id="mm-lead-fields" style="display:none" aria-label="Contact info collected so far">
      <div class="mm-lf-heading">Contact info</div>
      <div class="mm-lf-row"><span class="mm-lf-label">Name:</span><span id="mm-lf-name" class="mm-lf-value empty"></span></div>
      <div class="mm-lf-row"><span class="mm-lf-label">Title:</span><span id="mm-lf-title" class="mm-lf-value empty"></span></div>
      <div class="mm-lf-row"><span class="mm-lf-label">Company:</span><span id="mm-lf-company" class="mm-lf-value empty"></span></div>
      <div class="mm-lf-row"><span class="mm-lf-label">Email:</span><span id="mm-lf-email" class="mm-lf-value empty"></span></div>
      <div class="mm-lf-row"><span class="mm-lf-label">Phone:</span><span id="mm-lf-phone" class="mm-lf-value empty"></span></div>
    </div>
    <div id="mm-completion-banner" style="display:none">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
      <span>Thanks! Someone from the team will reach out soon.</span>
    </div>
  `;
  document.body.appendChild(panel);

  // Create the floating bubble
  const bubble = document.createElement('button');
  bubble.id = 'mm-chat-bubble';
  bubble.setAttribute('aria-label', 'Open chat');
  bubble.onclick = () => widgetWindow.__chatWidget.toggle();
  bubble.innerHTML = `
    <span class="bubble-icon">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
    </span>
    <span class="close-icon" style="display:none">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
    </span>
  `;
  document.body.appendChild(bubble);

  // Pulse after delay
  setTimeout(() => {
    bubble.classList.add('pulse');
    setTimeout(() => bubble.classList.remove('pulse'), 3000);
  }, 5000);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

widgetWindow.__chatWidget = {
  init: injectWidget,
  toggle: toggleChat,
  getThreadId: () => threadId,
};

// Auto-initialise
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', injectWidget);
} else {
  injectWidget();
}
