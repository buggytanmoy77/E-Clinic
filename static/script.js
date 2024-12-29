document.querySelector('.send-button').addEventListener('click',() => {
    const userInput = document.querySelector('.user-input').value; 
    if (userInput.trim() !== '') {
        const chatBox = document.querySelector('.chatbot-output'); 
        // User Message
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = userInput;
        chatBox.appendChild(userMessage);

        // Query Handling
        fetch('/chat',{
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({query: userInput})
        })
        .then(response => response.json())
        .then(data =>{
            // Bot Message
            const botMessage = document.createElement('div');
            botMessage.className = 'bot-message';
            botMessage.textContent = data.response;
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error =>{
            console.error('Error:', error);
        });
        // Clears the input field to ''
        document.querySelector('.user-input').value = '';
    }
});
