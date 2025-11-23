const uploadView = document.getElementById('upload-view');
const chatView = document.getElementById('chat-view');
const pdfUpload = document.getElementById('pdf-upload');
const uploadStatus = document.getElementById('upload-status');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');
const chatContainer = document.getElementById('chat-container');

// This is the URL of your local Python backend.
const API_URL = "http://127.0.0.1:8000";

pdfUpload.addEventListener('change', handlePdfUpload);
sendButton.addEventListener('click', handleSendMessage);
questionInput.addEventListener('keyup', (event) => {
    if (event.key === 'Enter') {
        handleSendMessage();
    }
});

function handlePdfUpload(event) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
        uploadStatus.textContent = `Processing "${file.name}"...`;
        
        // --- BACKEND INTEGRATION POINT 1 (REAL) ---
        // We use FormData to send the file to the backend.
        const formData = new FormData();
        formData.append('file', file);
        
        fetch(`${API_URL}/upload`, { 
            method: 'POST', 
            body: formData 
        })
        .then(response => {
            if (!response.ok) {
                // Handle HTTP errors
                return response.json().then(err => { 
                    throw new Error(err.detail || 'File upload failed'); 
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Upload successful:', data);
            switchToChatView(file.name);
        })
        .catch(error => {
            console.error('Upload failed:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
        });

    } else {
        uploadStatus.textContent = 'Please select a valid PDF file.';
    }
}

function switchToChatView(fileName) {
    uploadView.classList.add('hidden');
    chatView.classList.remove('hidden');
    questionInput.disabled = false;
    sendButton.disabled = false;
    
    addMessage('assistant', `The file "${fileName}" has been processed. You can now ask questions about its content.`);
}

function handleSendMessage() {
    const question = questionInput.value.trim();
    if (question === '') return;

    addMessage('user', question);
    questionInput.value = '';
    showThinkingIndicator();

    // --- BACKEND INTEGRATION POINT 2 (REAL) ---
    // We send the question as JSON to the /ask endpoint.
    fetch(`${API_URL}/ask`, { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question }) 
    })
    .then(response => {
         if (!response.ok) {
            // Handle HTTP errors
            return response.json().then(err => { 
                throw new Error(err.detail || 'Failed to get answer'); 
            });
        }
        return response.json();
    })
    .then(data => {
        removeThinkingIndicator();
        addMessage('assistant', data.answer);
    })
    .catch(error => {
        removeThinkingIndicator();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
        console.error('Q&A failed:', error);
    });
}

function addMessage(sender, text) {
    const messageWrapper = document.createElement('div');
    messageWrapper.className = `flex mb-4 ${sender === 'user' ? 'justify-end' : 'justify-start'}`;
    
    const messageBubble = document.createElement('div');
    messageBubble.className = `rounded-lg px-4 py-3 max-w-lg ${sender === 'user' ? 'bg-indigo-600 text-white' : 'bg-slate-200 text-slate-800'}`;
    messageBubble.textContent = text;
    
    messageWrapper.appendChild(messageBubble);
    chatContainer.appendChild(messageWrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showThinkingIndicator() {
    const indicatorWrapper = document.createElement('div');
    indicatorWrapper.id = 'thinking-indicator';
    indicatorWrapper.className = 'flex mb-4 justify-start';
    
    const indicatorBubble = document.createElement('div');
    indicatorBubble.className = 'rounded-lg px-4 py-3 max-w-lg bg-slate-200 text-slate-800 flex items-center';
    
    const spinner = document.createElement('div');
    spinner.className = 'animate-spin rounded-full h-4 w-4 border-b-2 border-slate-600 mr-3';

    const thinkingText = document.createElement('span');
    thinkingText.textContent = 'Thinking...';
    
    indicatorBubble.appendChild(spinner);
    indicatorBubble.appendChild(thinkingText);
    indicatorWrapper.appendChild(indicatorBubble);
    chatContainer.appendChild(indicatorWrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeThinkingIndicator() {
    const indicator = document.getElementById('thinking-indicator');
    if (indicator) {
        indicator.remove();
    }
}
