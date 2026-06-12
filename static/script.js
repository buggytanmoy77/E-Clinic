const sendBtn = document.getElementById('sendBtn');
const chatInput = document.getElementById('chatInput');
const chatMessages = document.getElementById('chatMessages');

function appendMessage(text, role) {
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;
  msg.appendChild(bubble);
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msg;
}

function sendMessage() {
  const userInput = chatInput.value.trim();
  if (!userInput) return;

  appendMessage(userInput, 'user');
  chatInput.value = '';

  // Typing indicator
  const typingMsg = appendMessage('Thinking…', 'bot typing');

  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: userInput })
  })
    .then(response => response.json())
    .then(data => {
      typingMsg.remove();
      appendMessage(data.response, 'bot');
    })
    .catch(() => {
      typingMsg.remove();
      appendMessage('Sorry, something went wrong. Please try again.', 'bot');
    });
}

sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
});