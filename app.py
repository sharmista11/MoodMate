<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoodMate Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f4f4f4;
        }
        .chat-container {
            width: 50%;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-box {
            height: 300px;
            overflow-y: scroll;
            border: 1px solid #ccc;
            padding: 10px;
            background: #fff;
            text-align: left;
        }
        .chat-box p {
            padding: 5px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .user-msg {
            background-color: #e3f2fd;
        }
        .bot-msg {
            background-color: #c8e6c9;
        }
        input {
            padding: 10px;
            width: 70%;
        }
        button {
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2>MoodMate Chatbot</h2>
        <div class="chat-box" id="chatBox"></div>
        <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
    <script>
        function sendMessage() {
            let userInput = document.getElementById("userInput").value;
            let chatBox = document.getElementById("chatBox");
            if (userInput.trim() === "") {
                alert("Please enter a message!");
                return;
            }
    
            chatBox.insertAdjacentHTML('beforeend', `<p class="user-msg"><strong>You:</strong> ${userInput}</p>`);
            document.getElementById("userInput").value = "";
    
            fetch("https://moodmate-12.onrender.com", { 
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userInput })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(data);  // Log the response for debugging
                if (data && data.response) {
                    chatBox.insertAdjacentHTML('beforeend', `<p class="bot-msg"><strong>Bot:</strong> ${data.response}</p>`);
                } else {
                    chatBox.insertAdjacentHTML('beforeend', `<p class="bot-msg"><strong>Bot:</strong> Sorry, I didn't understand that.</p>`);
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(error => {
                console.error("Error:", error);
                chatBox.insertAdjacentHTML('beforeend', `<p class="bot-msg"><strong>Bot:</strong> There was an error processing your message. Please try again.</p>`);
                chatBox.insertAdjacentHTML('beforeend', `<p class="bot-msg"><strong>Error Details:</strong> ${error.message}</p>`);
            });
        }
    
        function handleKeyPress(event) {
            if (event.key === "Enter") {
                sendMessage();
            }
        }
    </script>
</body>
</html>
