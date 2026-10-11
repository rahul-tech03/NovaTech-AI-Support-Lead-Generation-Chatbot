/**
 * NovaTech Solutions — AI Support & Lead Qualification Chatbot Client
 * Communicates with FastAPI backend /api/chat and /api/leads endpoints.
 */

// Configuration
const API_BASE_URL = window.location.origin.startsWith('http') 
  ? window.location.origin 
  : 'http://127.0.0.1:8000';

// State Management
let currentSessionId = getOrCreateSessionId();

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const typingAgent = document.getElementById('typing-agent');
const newChatBtn = document.getElementById('new-chat-btn');
const viewLeadsBtn = document.getElementById('view-leads-btn');
const leadsModal = document.getElementById('leads-modal');
const closeLeadsBtn = document.getElementById('close-leads-btn');
const leadsList = document.getElementById('leads-list');
const welcomeCard = document.getElementById('welcome-card');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  userInput.focus();
});

function getOrCreateSessionId() {
  let sid = sessionStorage.getItem('novatech_chat_session_id');
  if (!sid) {
    sid = 'session_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
    sessionStorage.setItem('novatech_chat_session_id', sid);
  }
  return sid;
}

function setupEventListeners() {
  // Form submission
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;
    await handleUserMessage(message);
  });

  // Quick Prompt Chips
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', async () => {
      const query = chip.getAttribute('data-query');
      if (query) {
        userInput.value = query;
        await handleUserMessage(query);
      }
    });
  });

  // Reset Chat
  newChatBtn.addEventListener('click', () => {
    sessionStorage.removeItem('novatech_chat_session_id');
    currentSessionId = getOrCreateSessionId();
    chatMessages.innerHTML = '';
    if (welcomeCard) {
      chatMessages.appendChild(welcomeCard);
    }
    userInput.value = '';
    userInput.focus();
  });

  // Leads Modal Controls
  viewLeadsBtn.addEventListener('click', openLeadsModal);
  closeLeadsBtn.addEventListener('click', closeLeadsModal);
  leadsModal.addEventListener('click', (e) => {
    if (e.target === leadsModal) closeLeadsModal();
  });
}

/**
 * Sends user message to the AI agent and renders response.
 */
async function handleUserMessage(text) {
  userInput.value = '';
  userInput.disabled = true;
  sendBtn.disabled = true;

  // Render User Message
  appendMessage({
    sender: 'user',
    text: text,
    time: formatCurrentTime()
  });

  // Show typing indicator
  showTyping(true);

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    showTyping(false);

    // Render Bot Message
    appendMessage({
      sender: 'bot',
      text: data.reply,
      agent: data.agent,
      lead_saved: data.lead_saved,
      lead_info: data.lead_info,
      time: formatCurrentTime()
    });

  } catch (error) {
    console.error('Chat error:', error);
    showTyping(false);
    appendMessage({
      sender: 'bot',
      text: `⚠️ Unable to connect to the agent server at ${API_BASE_URL}. Please ensure the FastAPI server is running with 'python chatbot_inference.py server'.`,
      agent: 'system',
      time: formatCurrentTime()
    });
  } finally {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

/**
 * Appends a message bubble to the chat stream.
 */
function appendMessage({ sender, text, agent, lead_saved, lead_info, time }) {
  const row = document.createElement('div');
  row.className = `message-row ${sender}`;

  let innerHTML = '';

  if (sender === 'bot') {
    const agentLabel = (agent === 'lead_agent') 
      ? 'Lead Qualification Specialist' 
      : 'QnA Support Agent';
    const badgeClass = (agent === 'lead_agent') ? 'lead' : 'qna';

    innerHTML += `
      <div class="agent-badge-row">
        <span class="agent-badge ${badgeClass}">${agentLabel}</span>
      </div>
    `;
  }

  // Format message text (handle newlines and basic bolding)
  const formattedText = escapeHtml(text)
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  innerHTML += `
    <div class="message-bubble">
      ${formattedText}
      ${lead_saved ? renderLeadSuccessCard(lead_info) : ''}
    </div>
    <span class="message-time">${time}</span>
  `;

  row.innerHTML = innerHTML;
  chatMessages.appendChild(row);
  scrollToBottom();
}

function renderLeadSuccessCard(info) {
  const leadId = (info && info.lead_id) ? info.lead_id : 'Atlas_Registered';
  const name = (info && info.name) ? info.name : 'Prospect';
  return `
    <div class="lead-captured-card">
      <div class="icon">🎯</div>
      <div>
        <strong>Lead Captured Successfully!</strong>
        <span>Lead for <em>${escapeHtml(name)}</em> has been securely saved in MongoDB Atlas. Lead ID: <span class="lead-id-code">${escapeHtml(leadId)}</span></span>
      </div>
    </div>
  `;
}

function showTyping(show, agentName = 'NovaTech Agent') {
  if (show) {
    typingAgent.textContent = `${agentName} is thinking`;
    typingIndicator.style.display = 'flex';
  } else {
    typingIndicator.style.display = 'none';
  }
  scrollToBottom();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(string) {
  if (!string) return '';
  return String(string)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Fetch and display captured leads from MongoDB Atlas
 */
async function openLeadsModal() {
  leadsModal.style.display = 'flex';
  leadsList.innerHTML = '<p class="loading-leads">Querying MongoDB Atlas (novatech_db.leads)...</p>';

  try {
    const response = await fetch(`${API_BASE_URL}/api/leads`);
    if (!response.ok) {
      throw new Error(`Failed to fetch leads: ${response.status}`);
    }
    const data = await response.json();
    const leads = data.leads || [];

    if (leads.length === 0) {
      leadsList.innerHTML = '<p class="loading-leads">No leads recorded yet. Try telling the chatbot: "I am interested in getting an AI chatbot for my business" to qualify a lead!</p>';
      return;
    }

    leadsList.innerHTML = leads.map(l => `
      <div class="lead-card">
        <div class="lead-card-header">
          <span class="lead-name">👤 ${escapeHtml(l.name || 'Unnamed')}</span>
          <span class="lead-date">${l.created_at ? new Date(l.created_at).toLocaleString() : ''}</span>
        </div>
        <div class="lead-meta">
          <div>📧 <strong>Email:</strong> ${escapeHtml(l.email || 'N/A')}</div>
          <div>📞 <strong>Contact:</strong> ${escapeHtml(l.contact || 'N/A')}</div>
          ${l.company_name ? `<div>🏢 <strong>Company:</strong> ${escapeHtml(l.company_name)}</div>` : ''}
          ${l.requirements ? `<div>📝 <strong>Needs:</strong> ${escapeHtml(l.requirements)}</div>` : ''}
        </div>
        <div class="lead-service-tag">
          🎯 ${escapeHtml(l.service_of_interest || 'General Inquiry')}
        </div>
      </div>
    `).join('');

  } catch (err) {
    leadsList.innerHTML = `
      <div style="color: #f87171; font-size: 0.85rem; padding: 12px; border: 1px solid rgba(248,113,113,0.3); border-radius: 8px;">
        ⚠️ Error querying leads: ${escapeHtml(err.message)}.<br><br>
        Make sure the backend server is running and your current IP address is whitelisted on MongoDB Atlas Network Access.
      </div>
    `;
  }
}

function closeLeadsModal() {
  leadsModal.style.display = 'none';
}
