document.addEventListener('DOMContentLoaded', function () {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const clearChatButton = document.getElementById('clear-chat');

    // Auto-resize textarea as user types
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';

        // Enable/disable send button based on input
        if (this.value.trim() === '') {
            sendButton.disabled = true;
        } else {
            sendButton.disabled = false;
        }
    });

    // Submit on Enter key (but allow Shift+Enter for new line)
    userInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim() !== '') {
                sendMessage();
            }
        }
    });

    // Clear chat history
    clearChatButton.addEventListener('click', function () {
        if (confirm('Are you sure you want to clear the conversation?')) {
            // Keep only the first welcome message
            const firstMessage = chatMessages.firstElementChild;
            chatMessages.innerHTML = '';
            chatMessages.appendChild(firstMessage);
        }
    });

    // Initialize the send button state
    sendButton.disabled = true;
});

function getCurrentTime() {
    const now = new Date();
    let hours = now.getHours();
    const minutes = now.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';

    hours = hours % 12;
    hours = hours ? hours : 12; // Hour '0' should be '12'

    const formattedMinutes = minutes < 10 ? '0' + minutes : minutes;
    return `${hours}:${formattedMinutes} ${ampm}`;
}

function createMessageElement(text, isUser) {
    const messageContainer = document.createElement('div');
    messageContainer.className = `message-container ${isUser ? 'user-container' : 'bot-container'}`;

    const avatar = document.createElement('div');
    avatar.className = `avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`;
    avatar.innerHTML = `<i class="fas fa-${isUser ? 'user' : 'robot'}"></i>`;

    const messageElement = document.createElement('div');
    messageElement.className = `message ${isUser ? 'user' : 'bot'}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    // For bot messages, process markdown-like formatting
    if (!isUser) {
        messageContent.innerHTML = formatMessage(text);
    } else {
        messageContent.textContent = text;
    }

    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = getCurrentTime();

    messageElement.appendChild(messageContent);
    messageElement.appendChild(messageTime);

    messageContainer.appendChild(avatar);
    messageContainer.appendChild(messageElement);

    return messageContainer;
}

function formatMessage(text) {
    // Convert URLs to links
    text = text.replace(/(https?:\/\/[^\s)\]]+)(?=[)\].,]*\s|$)/g, '<a href="$1" target="_blank">$1</a>');

    // Convert *bold* text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Convert _italic_ text
    text = text.replace(/\_(.*?)\_/g, '<em>$1</em>');

    // Convert ```code blocks```
    text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // Convert `inline code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Convert bullet points
    text = text.replace(/^- (.*)/gm, '<li>$1</li>').replace(/<li>(.*)<\/li>/g, '<ul><li>$1</li></ul>');

    // Convert numbered lists
    //text = text.replace(/^\d+\. (.*)/gm, '<li>$1</li>').replace(/<li>(.*)<\/li>/g, '<ol><li>$1</li></ol>');

    // Convert numbered lists
    text = text.replace(/^\d+\.\s?(.*)/gm, '<li>$1</li>'); // Convert each line to <li>
    text = text.replace(/(<li>.*<\/li>)+/g, '<ol>$&</ol>'); // Wrap consecutive <li> in <ol>

    // Convert paragraphs
    const paragraphs = text.split('\n\n');
    if (paragraphs.length > 1) {
        text = paragraphs.map(p => `<p>${p}</p>`).join('');
    }

    return text;
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.classList.add('visible');
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.classList.remove('visible');
}

function sendMessage() {
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const sendButton = document.getElementById('send-button');

    const userText = userInput.value.trim();
    if (userText === '') return;

    // Disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;

    // Add user message to chat
    const userMessage = createMessageElement(userText, true);
    chatMessages.appendChild(userMessage);
    scrollToBottom();

    // Clear input and reset height
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show typing indicator
    showTypingIndicator();

    // Send user message to backend
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userText }),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide typing indicator
            hideTypingIndicator();

            // Add bot response to chat
            const botMessage = createMessageElement(data.response, false);
            chatMessages.appendChild(botMessage);
            scrollToBottom();

            // Re-enable input
            userInput.disabled = false;
            userInput.focus();
        })
        .catch(error => {
            hideTypingIndicator();

            // Add error message
            const errorMessage = createMessageElement(
                "I'm sorry, I couldn't process your request. Please try again later.",
                false
            );
            chatMessages.appendChild(errorMessage);
            scrollToBottom();

            console.error('Error:', error);

            // Re-enable input
            userInput.disabled = false;
            userInput.focus();
        });
}