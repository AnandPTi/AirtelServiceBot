import React, { useState } from 'react';
import './App.css';
import Card from './Card';

function App() {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isMessageSent, setIsMessageSent] = useState(false);

  const handleSendMessage = async () => {
    if (!message) return;

    setChatHistory((prevHistory) => [
      ...prevHistory,
      { sender: 'You', text: message },
    ]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();

      setChatHistory((prevHistory) => [
        ...prevHistory,
        { sender: 'Bot', text: data.response },
      ]);
    } catch (error) {
      console.error('Error sending message:', error);
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { sender: 'Bot', text: 'Sorry, something went wrong.' },
      ]);
    }


    setMessage('');
    setIsMessageSent(true);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>TeraBot Airtel</h1>
        <div className="status">
          <span className="dot"></span>
        </div>
      </header>

      {!isMessageSent && (
        <div className="card-grid">
          <Card
            title="Get quick answers to common questions"
            description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort."
          />
          <Card
            title="Participate in everyday conversations"
            description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person."
          />
          <Card
            title="Easily request a Sim Swap"
            description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat."
          />
          <Card
            title="Start a Conversation"
            description="Begin chatting and let the assistant guide you through any query you have."
          />
        </div>
      )}

      <div className="chatbox">
        <div className="messages">
          {chatHistory.map((chat, index) => (
            <div key={index} className={`message-box ${chat.sender.toLowerCase()}`}>
              <p>
                <strong>{chat.sender}:</strong> {chat.text}
              </p>
            </div>
          ))}
        </div>

        <footer className="footer">
          <input
            type="text"
            placeholder="Enter message here..."
            className="message-input"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress} 
          />
          <button className="send-button" onClick={handleSendMessage}>
            ➤
          </button>
        </footer>
      </div>
    </div>
  );
}

export default App;


// import React, { useState } from 'react';
// import './App.css';
// import Card from './Card';

// function App() {
//   const [message, setMessage] = useState('');
//   const [chatHistory, setChatHistory] = useState([]);
//   const [isMessageSent, setIsMessageSent] = useState(false);

//   const handleSendMessage = async () => {
//     if (!message) return;

//     // Adding the user's message to the chat history
//     setChatHistory(prevHistory => [
//       ...prevHistory,
//       { sender: 'You', text: message },
//     ]);

//     try {
//       const response = await fetch('http://localhost:8000/chat', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ message }),
//       });

//       const data = await response.json();

//       // Adding the bot's response to the chat history
//       setChatHistory(prevHistory => [
//         ...prevHistory,
//         { sender: 'Bot', text: data.response },
//       ]);
//     } catch (error) {
//       console.error('Error sending message:', error);
//       setChatHistory(prevHistory => [
//         ...prevHistory,
//         { sender: 'Bot', text: 'Sorry, something went wrong.' },
//       ]);
//     }

//     // Clear the input after sending the message
//     setMessage('');
//     setIsMessageSent(true);
//   };

//   return (
//     <div className="app-container">
//       <header className="header">
//         <div className="header">
//           <h1>TeraBot Airtel</h1>
//           <div className="status">
//             <span className="dot"></span>
//           </div>
//         </div>
//       </header>

//       {!isMessageSent && (
//         <div className="card-grid">
//           <Card title="Get quick answers to common questions" description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort." />
//           <Card title="Participate in everyday conversations" description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person." />
//           <Card title="Easily request a Sim Swap" description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat." />
//           <Card title="Start a Conversation" description="Begin chatting and let the assistant guide you through any query you have." />
//         </div>
//       )}

//       <div className="chatbox">
//         <div className="messages">
//           {chatHistory.map((chat, index) => (
//             <div key={index} className={`message-box ${chat.sender.toLowerCase()}`}>
//               <p><strong>{chat.sender}:</strong> {chat.text}</p>
//             </div>
//           ))}
//         </div>

//         <footer className="footer">
//           <input
//             type="text"
//             placeholder="Enter message here..."
//             className="message-input"
//             value={message}
//             onChange={(e) => setMessage(e.target.value)}
//           />
//           <button className="send-button" onClick={handleSendMessage}>➤</button>
//         </footer>
//       </div>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from 'react';
// import './App.css';
// import Card from './Card';

// function App() {
//   const [message, setMessage] = useState('');
//   const [isMessageSent, setIsMessageSent] = useState(false);
//   const [chatHistory, setChatHistory] = useState([]); // Store chat history

//   const handleSendMessage = async () => {
//     if (!message) return;

//     // Add user message to chat history
//     setChatHistory((prevChat) => [...prevChat, { sender: 'You', text: message }]);

//     try {
//       const response = await fetch('http://localhost:8000/chat', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ message }),
//       });
//       const data = await response.json();
      
//       // Add bot response to chat history
//       setChatHistory((prevChat) => [...prevChat, { sender: 'Bot', text: data.response }]);
//       setIsMessageSent(true); 
//     } catch (error) {
//       console.error('Error sending message:', error);
//     }

//     // Clear the message input box
//     setMessage('');
//   };

//   return (
//     <div className="app-container">
//       <header className="header">
//         <h1>TeraBot Airtel</h1>
//         <div className="status">
//           <span className="dot"></span>
//         </div>
//       </header>

//       {!isMessageSent && (
//         <div className="card-grid">
//           <Card title="Get quick answers to common questions" description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort." />
//           <Card title="Participate in everyday conversations" description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person." />
//           <Card title="Easily request a Sim Swap" description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat." />
//           <Card title="Start a Conversation" description="Begin chatting and let the assistant guide you through any query you have." />
//         </div>
//       )}

//       <div className="chatbox">
//         <div className="messages">
//           {chatHistory.map((chat, index) => (
//             <div key={index} className={`message-box ${chat.sender.toLowerCase()}`}>
//               <p><strong>{chat.sender}:</strong> {chat.text}</p>
//             </div>
//           ))}
//         </div>

//         <footer className="footer">
//           <input 
//             type="text" 
//             placeholder="Enter message here..." 
//             className="message-input"
//             value={message}
//             onChange={(e) => setMessage(e.target.value)}
//           />
//           <button className="send-button" onClick={handleSendMessage}>➤</button>
//         </footer>
//       </div>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from 'react';
// import './App.css';
// import Card from './Card';

// function App() {
//   const [response, setResponse] = useState('');
//   const [message, setMessage] = useState('');  // For keeping the message in the box
//   const [isMessageSent, setIsMessageSent] = useState(false);  // Track if a message has been sent

//   const handleSendMessage = async () => {
//     if (!message) return;  // If there's no message, don't send it

//     try {
//       const response = await fetch('http://localhost:8000/chat', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ message }),
//       });
//       const data = await response.json();
//       setResponse(data.response);
//       setIsMessageSent(true);  // Hide cards after message is sent
//     } catch (error) {
//       console.error('Error sending message:', error);
//     }
//   };

//   return (
//     <div className="app-container">
//       <header className="header">
//         <div className="header">
//           <h1>TeraBot Airtel</h1>
//           <div className="status">
//             <span className="dot"></span>
//           </div>
//         </div>
//       </header>

//       {!isMessageSent && (  // Conditionally render cards if message is not sent
//         <div className="card-grid">
//           <Card title="Get quick answers to common questions" description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort." />
//           <Card title="Participate in everyday conversations" description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person." />
//           <Card title="Easily request a Sim Swap" description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat." />
//           <Card title="Start a Conversation" description="Begin chatting and let the assistant guide you through any query you have." />
//         </div>
//       )}

//       <div className="chatbox">
//         <div className="messages">
//           {response && <p>{response}</p>}
//         </div>
//         <footer className="footer">
//           <input 
//             type="text" 
//             placeholder="Enter message here..." 
//             className="message-input"
//             value={message}  // Keep the message in the box
//             onChange={(e) => setMessage(e.target.value)}  // Update message on typing
//           />
//           <button className="send-button" onClick={handleSendMessage}>➤</button>
//         </footer>
//       </div>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from "react";
// import { TextField, Button, Paper, Typography, Box, List, ListItem, ListItemText, Divider } from "@mui/material";
// import SendIcon from '@mui/icons-material/Send';
// import { grey } from "@mui/material/colors";
// import "./App.css";

// function App() {
//   const [message, setMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (message.trim() === "") return;

//     const newChatHistory = [...chatHistory, { sender: "user", text: message }];
//     setChatHistory(newChatHistory);

//     try {
//       const response = await fetch("http://localhost:8000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: message,
//         }),
//       });

//       const data = await response.json();
//       setChatHistory([...newChatHistory, { sender: "bot", text: data.response }]);
//       setMessage("");
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <Box className="app-container" display="flex" flexDirection="column" justifyContent="space-between" height="100vh" bgcolor="#1d2e27" p={3}>
//       <Paper elevation={6} style={{ backgroundColor: "#1B1B1B", padding: "1.5rem", borderRadius: "12px" }}>
//         <Typography variant="h4" align="center" gutterBottom color={grey[100]}>
//           TeraBot Airtel
//         </Typography>
//         <Divider style={{ backgroundColor: grey[600], marginBottom: "1rem" }} />
//         <Box
//           className="chat-box"
//           style={{
//             height: "500px",
//             overflowY: "auto",
//             padding: "1rem",
//             backgroundColor: "#162522",
//             borderRadius: "12px",
//             border: `1px solid ${grey[700]}`,
//           }}
//         >
//           <style>
//             {`
//               ::-webkit-scrollbar {
//                 width: 8px;
//               }
//               ::-webkit-scrollbar-track {
//                 background: transparent;
//               }
//               ::-webkit-scrollbar-thumb {
//                 background-color: ${grey[700]};
//                 border-radius: 20px;
//               }
//             `}
//           </style>
//           <List>
//             {chatHistory.map((chat, index) => (
//               <ListItem
//                 key={index}
//                 style={{
//                   justifyContent: chat.sender === "user" ? "flex-end" : "flex-start",
//                 }}
//               >
//                 <ListItemText
//                   primary={
//                     <Typography
//                       variant="body1"
//                       style={{
//                         color: "white",
//                         backgroundColor: chat.sender === "user" ? "#00af87" : "#2b4237",
//                         padding: "0.5rem 1rem",
//                         borderRadius: "12px",
//                         maxWidth: "80%",
//                       }}
//                     >
//                       {chat.text}
//                     </Typography>
//                   }
//                 />
//               </ListItem>
//             ))}
//           </List>
//         </Box>
//         <Divider style={{ backgroundColor: grey[600], marginTop: "1rem" }} />
//         <Box mt={2}>
//           <form onSubmit={handleSubmit}>
//             <Box display="flex" gap={2} style={{ width: "100%" }}>
//               <TextField
//                 variant="outlined"
//                 fullWidth
//                 label="Enter message here..."
//                 value={message}
//                 onChange={(e) => setMessage(e.target.value)}
//                 InputLabelProps={{
//                   style: { color: grey[400] },
//                 }}
//                 InputProps={{
//                   style: {
//                     color: grey[100],
//                     backgroundColor: "#333333",
//                     borderRadius: "20px",
//                   },
//                 }}
//               />
//               <Button
//                 variant="contained"
//                 color="primary"
//                 type="submit"
//                 style={{ height: "100%", borderRadius: "20px" }}
//               >
//                 <SendIcon />
//               </Button>
//             </Box>
//           </form>
//         </Box>
//       </Paper>
//     </Box>
//   );
// }

// export default App;


// import React, { useState } from 'react';
// import './App.css';

// function App() {
//   const [messages, setMessages] = useState([]);
//   const [message, setMessage] = useState('');

//   const handleSend = async () => {
//     if (message.trim()) {
//       // Add user message to the chat
//       setMessages([...messages, { sender: 'user', text: message }]);
//       setMessage(''); // Keep the message in the input box

//       try {
//         // Send message to the backend (chatbot server)
//         const response = await fetch('http://localhost:8000/chat', {
//           method: 'POST',
//           headers: {
//             'Content-Type': 'application/json',
//           },
//           body: JSON.stringify({ message, user_id: 'default_user' }),
//         });
//         const data = await response.json();
//         // Add chatbot response to the chat
//         setMessages([...messages, { sender: 'user', text: message }, { sender: 'bot', text: data.response }]);
//       } catch (error) {
//         console.error('Error sending message:', error);
//       }
//     }
//   };

//   return (
//     <div className="chat-container">
//       <header className="header">
//         <h1>TeraBot Airtel</h1>
//       </header>

//       <div className="chat-box">
//         {messages.map((msg, index) => (
//           <div key={index} className={`message ${msg.sender}`}>
//             <span>{msg.text}</span>
//           </div>
//         ))}
//       </div>

//       <footer className="footer">
//         <input
//           type="text"
//           value={message}
//           onChange={(e) => setMessage(e.target.value)}
//           placeholder="Enter message here..."
//           className="message-input"
//         />
//         <button className="send-button" onClick={handleSend}>
//           ➤
//         </button>
//       </footer>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from 'react';
// import './App.css';
// import Card from './Card';

// function App() {
//   const [response, setResponse] = useState('');
//   const [message, setMessage] = useState('');  // For keeping the message in the box
//   const [isMessageSent, setIsMessageSent] = useState(false);  // Track if a message has been sent

//   const handleSendMessage = async () => {
//     if (!message) return;  // If there's no message, don't send it

//     try {
//       const response = await fetch('http://localhost:8000/chat', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ message }),
//       });
//       const data = await response.json();
//       setResponse(data.response);
//       setIsMessageSent(true);  // Hide cards after message is sent
//     } catch (error) {
//       console.error('Error sending message:', error);
//     }
//   };

//   return (
//     <div className="app-container">
//       <header className="header">
//         <div className="header">
//           <h1>GenAI Support Agent</h1>
//           <div className="status">
//             <span className="dot"></span>
//           </div>
//         </div>
//       </header>

//       {!isMessageSent && (  // Conditionally render cards if message is not sent
//         <div className="card-grid">
//           <Card title="Get quick answers to common questions" description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort." />
//           <Card title="Participate in everyday conversations" description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person." />
//           <Card title="Easily request a Sim Swap" description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat." />
//           <Card title="Start a Conversation" description="Begin chatting and let the assistant guide you through any query you have." />
//         </div>
//       )}

//       <div className="chatbox">
//         <div className="messages">
//           {response && <p>{response}</p>}
//         </div>
//         <footer className="footer">
//           <input 
//             type="text" 
//             placeholder="Enter message here..." 
//             className="message-input"
//             value={message}  // Keep the message in the box
//             onChange={(e) => setMessage(e.target.value)}  // Update message on typing
//           />
//           <button className="send-button" onClick={handleSendMessage}>➤</button>
//         </footer>
//       </div>
//     </div>
//   );
// }

// export default App;



// import React, { useState } from 'react';
// import './App.css';
// import Card from './Card';

// function App() {
//   const [message, setMessage] = useState('');
//   const [response, setResponse] = useState('');
//   const [conversation, setConversation] = useState([]);

//   // Function to handle message input change
//   const handleMessageChange = (e) => {
//     setMessage(e.target.value);
//   };

//   // Function to handle sending message
//   const handleSendMessage = async () => {
//     if (!message.trim()) return;

//     // Add the user message to the conversation
//     const newConversation = [...conversation, { role: 'user', content: message }];
//     setConversation(newConversation);

//     // Send message to the backend
//     try {
//       const res = await fetch('http://localhost:8000/chat', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           message: message,  // Send the user message
//           user_id: 'default_user',  // Example user_id
//         }),
//       });
//       const data = await res.json();
//       const botResponse = data.response;

//       // Update the conversation with the bot's response
//       setConversation([...newConversation, { role: 'bot', content: botResponse }]);
//       setMessage(''); // Clear the input after sending the message
//     } catch (error) {
//       console.error('Error sending message:', error);
//     }
//   };

//   return (
//     <div className="app-container">
//       <header className="header">
//         <div className="header">
//           <h1>GenAI Support Agent</h1>
//           <div className="status">
//             <span className="dot"></span>
//           </div>
//         </div>
//       </header>

//       <div className="card-grid">
//         <Card title="Get quick answers to common questions" description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort." />
//         <Card title="Participate in everyday conversations" description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person." />
//         <Card title="Easily request a Sim Swap" description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat." />
//         <Card title="Start a Conversation" description="Begin chatting and let the assistant guide you through any query you have." />
//       </div>

//       <div className="conversation-container">
//         {conversation.map((msg, index) => (
//           <div key={index} className={msg.role === 'user' ? 'user-message' : 'bot-message'}>
//             {msg.content}
//           </div>
//         ))}
//       </div>

//       <footer className="footer">
//         <input
//           type="text"
//           placeholder="Enter message here..."
//           value={message}
//           onChange={handleMessageChange}
//           className="message-input"
//         />
//         <button className="send-button" onClick={handleSendMessage}>➤</button>
//       </footer>
//     </div>
//   );
// }

// export default App;



// import React from 'react';
// import './App.css';
// import Card from './Card';

// function App() {
//   return (
//     <div className="app-container">
//       <header className="header">
//       <div className="header">
//   <h1>GenAI Support Agent</h1>
//   <div className="status">
//     <span className="dot"></span>

//   </div>
// </div>
//       </header>

//       <div className="card-grid">
//         <Card title="Get quick answers to common questions" description="Instantly get answers to common questions with fast, accurate responses, saving you time and effort." />
//         <Card title="Participate in everyday conversations" description="Engage effortlessly in everyday conversations, enjoying smooth interactions like talking to a real person." />
//         <Card title="Easily request a Sim Swap" description="The chatbot allows you to easily request a sim swap and fill up a form without leaving the chat." />
//         <Card title="Start a Conversation" description="Begin chatting and let the assistant guide you through any query you have." />
//       </div>

//       <footer className="footer">
//         <input type="text" placeholder="Enter message here..." className="message-input" />
//         <button className="send-button">➤</button>
//       </footer>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from "react";
// import {
//   Container,
//   Box,
//   TextField,
//   Button,
//   Paper,
//   Typography,
//   List,
//   ListItem,
//   ListItemText,
//   Divider,
//   createTheme,
//   ThemeProvider,
//   AppBar,
//   Toolbar,
//   IconButton,
//   Link,
// } from "@mui/material";
// import { grey, blue, purple } from "@mui/material/colors";
// import SendIcon from '@mui/icons-material/Send'; 
// import GitHubIcon from '@mui/icons-material/GitHub';
// import MailIcon from '@mui/icons-material/Mail';
// import TelegramIcon from '@mui/icons-material/Telegram';
// import "./App.css";

// // Custom dark theme
// const darkTheme = createTheme({
//   palette: {
//     mode: "dark",
//     background: {
//       default: "#121212", // More dark background
//       paper: "#1B1B1B", // Darker paper background
//     },
//     primary: {
//       main: blue[500],
//     },
//     secondary: {
//       main: purple[500],
//     },
//     text: {
//       primary: grey[100],
//       secondary: grey[400],
//     },
//   },
//   typography: {
//     h4: {
//       fontWeight: 700,
//       color: grey[100], // White heading
//     },
//     body1: {
//       color: grey[100],
//     },
//   },
// });

// // Header component
// const Header = () => (
//   <AppBar position="static" color="primary">
//     <Toolbar>
//       <Typography variant="h6" style={{ flexGrow: 1 }}>
//         TeraBot Airtel
//       </Typography>
//       <IconButton color="inherit" component={Link} href="https://github.com/your-profile" target="_blank">
//         <GitHubIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="mailto:your-email@example.com">
//         <MailIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="https://t.me/your-telegram">
//         <TelegramIcon />
//       </IconButton>
//     </Toolbar>
//   </AppBar>
// );

// // Footer component
// // Footer component
// const Footer = () => (
//   <Box mt={5} textAlign="center" pb={3} className="footer">
//     <Typography variant="body2">
//       © 2024 StackWalls. All Rights Reserved.
//     </Typography>
//   </Box>
// );


// function App() {
//   const [message, setMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (message.trim() === "") return;

//     const newChatHistory = [...chatHistory, { sender: "user", text: message }];
//     setChatHistory(newChatHistory);

//     try {
//       const response = await fetch("http://localhost:8000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: message,
//         }),
//       });

//       const data = await response.json();

//       setChatHistory([
//         ...newChatHistory,
//         { sender: "bot", text: data.response },
//       ]);

//       setMessage("");
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <ThemeProvider theme={darkTheme}>
//       <Header />
//       <Container maxWidth="lg" style={{ marginTop: "2rem" }}>
//         <Paper elevation={6} style={{ backgroundColor: "#1B1B1B", padding: "1.5rem" }}>
//           <Typography variant="h4" align="center" gutterBottom>
//             Airtel Bot
//           </Typography>
//           <Typography variant="body2" align="center" color="textSecondary">
//             Powered by TeraBot
//           </Typography>
//           <Divider style={{ backgroundColor: grey[600], marginBottom: "1rem" }} />
//           <Box
//             className="chatbox"
//             style={{
//               height: "500px",
//               overflowY: "auto",
//               padding: "1rem",
//               backgroundColor: "#121212",
//               borderRadius: "8px",
//               border: `1px solid ${grey[700]}`,
//               scrollbarWidth: "thin", // for Firefox
//               scrollbarColor: `${grey[700]} transparent`, // for Firefox
//             }}
//           >
//             <style>
//               {`
//                 ::-webkit-scrollbar {
//                   width: 8px;
//                 }
//                 ::-webkit-scrollbar-track {
//                   background: transparent;
//                 }
//                 ::-webkit-scrollbar-thumb {
//                   background-color: ${grey[700]}; 
//                   border-radius: 20px;
//                 }
//               `}
//             </style>
//             <List>
//               {chatHistory.map((chat, index) => (
//                 <ListItem
//                   key={index}
//                   className={chat.sender === "user" ? "chat-user" : "chat-bot"}
//                   style={{
//                     justifyContent: chat.sender === "user" ? "flex-end" : "flex-start",
//                   }}
//                 >
//                   <ListItemText
//                     primary={
//                       <Typography
//                       variant="body1"
//                       style={{
//                         color: "white",
//                         backgroundColor: chat.sender === "user" ? grey[800] : "#222222", // Lighter than black
//                         padding: "0.5rem 1rem",
//                         borderRadius: "12px",
//                         maxWidth: "80%",
//                       }}
//                     >
//                       {chat.sender === "user" ? "You" : "Bot"}: {chat.text}
//                     </Typography>
//                     }
//                   />
//                 </ListItem>
//               ))}
//             </List>
//           </Box>
//           <Divider style={{ backgroundColor: grey[700], marginTop: "1rem" }} />
//           <Box mt={2}>
//             <form onSubmit={handleSubmit}>
//               <Box display="flex" gap={2} style={{ width: "100%" }}>
//                 <TextField
//                   variant="outlined"
//                   fullWidth
//                   label="Type your message..."
//                   value={message}
//                   onChange={(e) => setMessage(e.target.value)}
//                   InputLabelProps={{
//                     style: { color: grey[400] },
//                   }}
//                   InputProps={{
//                     style: {
//                       color: grey[100],
//                       backgroundColor: "#333333",
//                       borderRadius: "20px",
//                     },
//                   }}
//                 />
//                 <Button
//                   variant="contained"
//                   color="primary"
//                   type="submit"
//                   style={{ height: "100%", borderRadius: "20px" }}
//                 >
//                   <SendIcon /> 
//                 </Button>
//               </Box>
//             </form>
//           </Box>
//         </Paper>
//       </Container>
//       <Footer />
//     </ThemeProvider>
//   );
// }

// export default App;





// //working properly modifying UI
// import React, { useState } from "react";
// import {
//   Container,
//   Box,
//   TextField,
//   Button,
//   Paper,
//   Typography,
//   List,
//   ListItem,
//   ListItemText,
//   Divider,
//   createTheme,
//   ThemeProvider,
//   AppBar,
//   Toolbar,
//   IconButton,
//   Link,
// } from "@mui/material";
// import { grey, blue, pink } from "@mui/material/colors";
// import SendIcon from '@mui/icons-material/Send'; // Importing Send icon
// import GitHubIcon from '@mui/icons-material/GitHub';
// import MailIcon from '@mui/icons-material/Mail';
// import TelegramIcon from '@mui/icons-material/Telegram';
// import "./App.css";

// // Custom dark theme
// const darkTheme = createTheme({
//   palette: {
//     mode: "dark",
//     background: {
//       default: "#2F4F2F", // Dark green background
//       paper: "#1E1E1E",
//     },
//     primary: {
//       main: blue[500],
//     },
//     secondary: {
//       main: pink[500],
//     },
//     text: {
//       primary: grey[100],
//       secondary: grey[400],
//     },
//   },
//   typography: {
//     h4: {
//       fontWeight: 700,
//       color: blue[500],
//     },
//     body1: {
//       color: grey[100],
//     },
//   },
// });

// // Header component
// const Header = () => (
//   <AppBar position="static" color="primary">
//     <Toolbar>
//       <Typography variant="h6" style={{ flexGrow: 1 }}>
//         Airtel HandShake
//       </Typography>
//       <IconButton color="inherit" component={Link} href="https://github.com/your-profile" target="_blank">
//         <GitHubIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="mailto:your-email@example.com">
//         <MailIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="https://t.me/your-telegram">
//         <TelegramIcon />
//       </IconButton>
//     </Toolbar>
//   </AppBar>
// );

// // Footer component
// const Footer = () => (
//   <Box mt={5} textAlign="center" pb={3}>
//     <Typography variant="body2" color="textSecondary">
//       © 2024 Your Name. All Rights Reserved.
//     </Typography>
//   </Box>
// );

// function App() {
//   const [message, setMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (message.trim() === "") return;

//     // Add user message to chat history
//     const newChatHistory = [...chatHistory, { sender: "user", text: message }];
//     setChatHistory(newChatHistory);

//     try {
//       // Send the user message to the backend
//       const response = await fetch("http://localhost:8000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: message,  // Message entered by the user
//         }),
//       });

//       const data = await response.json();

//       // Add bot response to chat history
//       setChatHistory([
//         ...newChatHistory,
//         { sender: "bot", text: data.response },  // Response from backend
//       ]);

//       setMessage(""); // Clear input field
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <ThemeProvider theme={darkTheme}>
//       <Header />
//       <Container maxWidth="lg" style={{ marginTop: "2rem" }}>
//         <Paper elevation={6} style={{ backgroundColor: "#2C2C2C", padding: "1.5rem" }}>
//           <Typography variant="h4" align="center" gutterBottom>
//             Airtel HandShake Chat
//           </Typography>
//           <Divider style={{ backgroundColor: grey[700], marginBottom: "1rem" }} />
//           <Box
//             className="chatbox"
//             style={{
//               height: "500px", // Increased height for chatbox
//               overflowY: "auto",
//               padding: "1rem",
//               backgroundColor: "#1E1E1E",
//               borderRadius: "8px",
//               border: `1px solid ${grey[700]}`,
//             }}
//           >
//             <List>
//               {chatHistory.map((chat, index) => (
//                 <ListItem
//                   key={index}
//                   className={chat.sender === "user" ? "chat-user" : "chat-bot"}
//                   style={{
//                     justifyContent: chat.sender === "user" ? "flex-end" : "flex-start",
//                   }}
//                 >
//                   <ListItemText
//                     primary={
//                       <Typography
//                         variant="body1"
//                         style={{
//                           color: "white",
//                           backgroundColor: chat.sender === "user" ? blue[800] : "#000000", // Background colors
//                           padding: "0.5rem 1rem",
//                           borderRadius: "12px",
//                           maxWidth: "80%",
//                         }}
//                       >
//                         {chat.sender === "user" ? "You" : "Bot"}: {chat.text}
//                       </Typography>
//                     }
//                   />
//                 </ListItem>
//               ))}
//             </List>
//           </Box>
//           <Divider style={{ backgroundColor: grey[700], marginTop: "1rem" }} />
//           <Box mt={2}>
//             <form onSubmit={handleSubmit}>
//               <Box display="flex" gap={2} style={{ width: "100%" }}>
//                 <TextField
//                   variant="outlined"
//                   fullWidth
//                   label="Type your message..."
//                   value={message}
//                   onChange={(e) => setMessage(e.target.value)}
//                   InputLabelProps={{
//                     style: { color: grey[400] },
//                   }}
//                   InputProps={{
//                     style: {
//                       color: grey[100],
//                       backgroundColor: "#333333",
//                       borderRadius: "20px", // Curved edges
//                     },
//                   }}
//                 />
//                 <Button
//                   variant="contained"
//                   color="primary"
//                   type="submit"
//                   style={{ height: "100%", borderRadius: "20px" }} // Curved edges for button
//                 >
//                   <SendIcon /> {/* Send icon */}
//                 </Button>
//               </Box>
//             </form>
//           </Box>
//         </Paper>
//       </Container>
//       <Footer />
//     </ThemeProvider>
//   );
// }

// export default App;




// import React, { useState } from "react";
// import {
//   Container,
//   Box,
//   TextField,
//   Button,
//   Paper,
//   Typography,
//   List,
//   ListItem,
//   ListItemText,
//   Divider,
//   createTheme,
//   ThemeProvider,
//   AppBar,
//   Toolbar,
//   IconButton,
//   Link,
// } from "@mui/material";
// import { grey, blue, pink } from "@mui/material/colors";
// import GitHubIcon from '@mui/icons-material/GitHub';
// import MailIcon from '@mui/icons-material/Mail';
// import TelegramIcon from '@mui/icons-material/Telegram';
// import "./App.css";

// // Custom dark theme
// const darkTheme = createTheme({
//   palette: {
//     mode: "dark",
//     background: {
//       default: "#2F4F2F", // Dark green background
//       paper: "#1E1E1E",
//     },
//     primary: {
//       main: blue[500],
//     },
//     secondary: {
//       main: pink[500],
//     },
//     text: {
//       primary: grey[100],
//       secondary: grey[400],
//     },
//   },
//   typography: {
//     h4: {
//       fontWeight: 700,
//       color: blue[500],
//     },
//     body1: {
//       color: grey[100],
//     },
//   },
// });

// // Header component
// const Header = () => (
//   <AppBar position="static" color="primary">
//     <Toolbar>
//       <Typography variant="h6" style={{ flexGrow: 1 }}>
//         Airtel HandShake
//       </Typography>
//       <IconButton color="inherit" component={Link} href="https://github.com/your-profile" target="_blank">
//         <GitHubIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="mailto:your-email@example.com">
//         <MailIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="https://t.me/your-telegram">
//         <TelegramIcon />
//       </IconButton>
//     </Toolbar>
//   </AppBar>
// );

// // Footer component
// const Footer = () => (
//   <Box mt={5} textAlign="center" pb={3}>
//     <Typography variant="body2" color="textSecondary">
//       © 2024 Your Name. All Rights Reserved.
//     </Typography>
//   </Box>
// );

// function App() {
//   const [message, setMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (message.trim() === "") return;

//     // Add user message to chat history
//     const newChatHistory = [...chatHistory, { sender: "user", text: message }];
//     setChatHistory(newChatHistory);

//     try {
//       // Send the user message to the backend
//       const response = await fetch("http://localhost:8000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: message,  // Message entered by the user
//         }),
//       });

//       const data = await response.json();

//       // Add bot response to chat history
//       setChatHistory([
//         ...newChatHistory,
//         { sender: "bot", text: data.response },  // Response from backend
//       ]);

//       setMessage(""); // Clear input field
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <ThemeProvider theme={darkTheme}>
//       <Header />
//       <Container maxWidth="lg" style={{ marginTop: "2rem" }}>
//         <Paper elevation={6} style={{ backgroundColor: "#2C2C2C", padding: "1.5rem" }}>
//           <Typography variant="h4" align="center" gutterBottom>
//             Airtel HandShake Chat
//           </Typography>
//           <Divider style={{ backgroundColor: grey[700], marginBottom: "1rem" }} />
//           <Box
//             className="chatbox"
//             style={{
//               height: "500px", // Increased height for chatbox
//               overflowY: "auto",
//               padding: "1rem",
//               backgroundColor: "#1E1E1E",
//               borderRadius: "8px",
//               border: `1px solid ${grey[700]}`,
//             }}
//           >
//             <List>
//               {chatHistory.map((chat, index) => (
//                 <ListItem
//                   key={index}
//                   className={chat.sender === "user" ? "chat-user" : "chat-bot"}
//                   style={{
//                     justifyContent: chat.sender === "user" ? "flex-end" : "flex-start",
//                   }}
//                 >
//                   <ListItemText
//                     primary={
//                       <Typography
//                         variant="body1"
//                         style={{
//                           color: chat.sender === "user" ? blue[300] : grey[400],
//                           backgroundColor: chat.sender === "user" ? blue[800] : "#333333",
//                           padding: "0.5rem 1rem",
//                           borderRadius: "12px",
//                           maxWidth: "80%",
//                         }}
//                       >
//                         {chat.sender === "user" ? "You" : "Bot"}: {chat.text}
//                       </Typography>
//                     }
//                   />
//                 </ListItem>
//               ))}
//             </List>
//           </Box>
//           <Divider style={{ backgroundColor: grey[700], marginTop: "1rem" }} />
//           <Box mt={2}>
//             <form onSubmit={handleSubmit}>
//               <Box display="flex" gap={2}>
//                 <TextField
//                   variant="outlined"
//                   fullWidth
//                   label="Type your message..."
//                   value={message}
//                   onChange={(e) => setMessage(e.target.value)}
//                   InputLabelProps={{
//                     style: { color: grey[400] },
//                   }}
//                   InputProps={{
//                     style: { color: grey[100], backgroundColor: "#333333" },
//                   }}
//                 />
//                 <Button
//                   variant="contained"
//                   color="primary"
//                   type="submit"
//                   style={{ height: "100%" }}
//                 >
//                   Send
//                 </Button>
//               </Box>
//             </form>
//           </Box>
//         </Paper>
//       </Container>
//       <Footer />
//     </ThemeProvider>
//   );
// }

// export default App;

// import React, { useState } from "react";
// import {
//   Container,
//   Box,
//   TextField,
//   Button,
//   Paper,
//   Typography,
//   List,
//   ListItem,
//   ListItemText,
//   Divider,
//   createTheme,
//   ThemeProvider,
//   AppBar,
//   Toolbar,
//   IconButton,
//   Link,
// } from "@mui/material";
// import { grey, blue, pink } from "@mui/material/colors";
// import GitHubIcon from '@mui/icons-material/GitHub';
// import MailIcon from '@mui/icons-material/Mail';
// import TelegramIcon from '@mui/icons-material/Telegram';
// import "./App.css";

// // Custom dark theme
// const darkTheme = createTheme({
//   palette: {
//     mode: "dark",
//     background: {
//       default: "#121212",
//       paper: "#1E1E1E",
//     },
//     primary: {
//       main: blue[500],
//     },
//     secondary: {
//       main: pink[500],
//     },
//     text: {
//       primary: grey[100],
//       secondary: grey[400],
//     },
//   },
//   typography: {
//     h4: {
//       fontWeight: 700,
//       color: blue[500],
//     },
//     body1: {
//       color: grey[100],
//     },
//   },
// });

// // Header component
// const Header = () => (
//   <AppBar position="static" color="primary">
//     <Toolbar>
//       <Typography variant="h6" style={{ flexGrow: 1 }}>
//         Chatbot Interface
//       </Typography>
//       <IconButton color="inherit" component={Link} href="https://github.com/your-profile" target="_blank">
//         <GitHubIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="mailto:your-email@example.com">
//         <MailIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="https://t.me/your-telegram">
//         <TelegramIcon />
//       </IconButton>
//     </Toolbar>
//   </AppBar>
// );

// // Footer component
// const Footer = () => (
//   <Box mt={5} textAlign="center" pb={3}>
//     <Typography variant="body2" color="textSecondary">
//       © 2024 Your Name. All Rights Reserved.
//     </Typography>
//   </Box>
// );

// function App() {
//   const [message, setMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (message.trim() === "") return;

//     // Add user message to chat history
//     const newChatHistory = [...chatHistory, { sender: "user", text: message }];
//     setChatHistory(newChatHistory);

//     try {
//       // Send the user message to the backend
//       const response = await fetch("http://localhost:8000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: message,  // Message entered by the user
//         }),
//       });

//       const data = await response.json();

//       // Add bot response to chat history
//       setChatHistory([
//         ...newChatHistory,
//         { sender: "bot", text: data.response },  // Response from backend
//       ]);

//       setMessage(""); // Clear input field
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <ThemeProvider theme={darkTheme}>
//       <Header />
//       <Container maxWidth="sm" style={{ marginTop: "2rem" }}>
//         <Paper elevation={6} style={{ backgroundColor: "#2C2C2C", padding: "1.5rem" }}>
//           <Typography variant="h4" align="center" gutterBottom>
//             Chatbot Interface
//           </Typography>
//           <Divider style={{ backgroundColor: grey[700], marginBottom: "1rem" }} />
//           <Box
//             className="chatbox"
//             style={{
//               height: "300px",
//               overflowY: "auto",
//               padding: "1rem",
//               backgroundColor: "#1E1E1E",
//               borderRadius: "8px",
//               border: `1px solid ${grey[700]}`,
//             }}
//           >
//             <List>
//               {chatHistory.map((chat, index) => (
//                 <ListItem
//                   key={index}
//                   className={chat.sender === "user" ? "chat-user" : "chat-bot"}
//                   style={{
//                     justifyContent: chat.sender === "user" ? "flex-end" : "flex-start",
//                   }}
//                 >
//                   <ListItemText
//                     primary={
//                       <Typography
//                         variant="body1"
//                         style={{
//                           color: chat.sender === "user" ? blue[300] : grey[400],
//                           backgroundColor: chat.sender === "user" ? blue[800] : "#333333",
//                           padding: "0.5rem 1rem",
//                           borderRadius: "12px",
//                           maxWidth: "80%",
//                         }}
//                       >
//                         {chat.sender === "user" ? "You" : "Bot"}: {chat.text}
//                       </Typography>
//                     }
//                   />
//                 </ListItem>
//               ))}
//             </List>
//           </Box>
//           <Divider style={{ backgroundColor: grey[700], marginTop: "1rem" }} />
//           <Box mt={2}>
//             <form onSubmit={handleSubmit}>
//               <Box display="flex" gap={2}>
//                 <TextField
//                   variant="outlined"
//                   fullWidth
//                   label="Type your message..."
//                   value={message}
//                   onChange={(e) => setMessage(e.target.value)}
//                   InputLabelProps={{
//                     style: { color: grey[400] },
//                   }}
//                   InputProps={{
//                     style: { color: grey[100], backgroundColor: "#333333" },
//                   }}
//                 />
//                 <Button
//                   variant="contained"
//                   color="primary"
//                   type="submit"
//                   style={{ height: "100%" }}
//                 >
//                   Send
//                 </Button>
//               </Box>
//             </form>
//           </Box>
//         </Paper>
//       </Container>
//       <Footer />
//     </ThemeProvider>
//   );
// }

// export default App;


// import React, { useState } from "react";
// import {
//   Container,
//   Box,
//   TextField,
//   Button,
//   Paper,
//   Typography,
//   List,
//   ListItem,
//   ListItemText,
//   Divider,
//   createTheme,
//   ThemeProvider,
//   AppBar,
//   Toolbar,
//   IconButton,
//   Link,
// } from "@mui/material";
// import { grey, blue, pink } from "@mui/material/colors";
// import GitHubIcon from '@mui/icons-material/GitHub';
// import MailIcon from '@mui/icons-material/Mail';
// import TelegramIcon from '@mui/icons-material/Telegram';
// import "./App.css";

// // Custom dark theme
// const darkTheme = createTheme({
//   palette: {
//     mode: "dark",
//     background: {
//       default: "#121212",
//       paper: "#1E1E1E",
//     },
//     primary: {
//       main: blue[500],
//     },
//     secondary: {
//       main: pink[500],
//     },
//     text: {
//       primary: grey[100],
//       secondary: grey[400],
//     },
//   },
//   typography: {
//     h4: {
//       fontWeight: 700,
//       color: blue[500],
//     },
//     body1: {
//       color: grey[100],
//     },
//   },
// });

// // Header component
// const Header = () => (
//   <AppBar position="static" color="primary">
//     <Toolbar>
//       <Typography variant="h6" style={{ flexGrow: 1 }}>
//         Chatbot Interface
//       </Typography>
//       <IconButton color="inherit" component={Link} href="https://github.com/your-profile" target="_blank">
//         <GitHubIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="mailto:your-email@example.com">
//         <MailIcon />
//       </IconButton>
//       <IconButton color="inherit" component={Link} href="https://t.me/your-telegram">
//         <TelegramIcon />
//       </IconButton>
//     </Toolbar>
//   </AppBar>
// );

// // Footer component
// const Footer = () => (
//   <Box mt={5} textAlign="center" pb={3}>
//     <Typography variant="body2" color="textSecondary">
//       © 2024 Your Name. All Rights Reserved.
//     </Typography>
//   </Box>
// );

// function App() {
//   const [message, setMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (message.trim() === "") return;

//     // Add user message to chat history
//     const newChatHistory = [...chatHistory, { sender: "user", text: message }];
//     setChatHistory(newChatHistory);

//     try {
//       // Send the request with message, name, and phone number
//       const response = await fetch("http://localhost:8000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: message,
//           full_name: "John Doe",
//           phone_number: "1234567890",
//         }),
//       });

//       const data = await response.json();

//       // Add bot response to chat history
//       setChatHistory([...newChatHistory, { sender: "bot", text: data.response }]);

//       setMessage(""); // Clear input field
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <ThemeProvider theme={darkTheme}>
//       <Header />
//       <Container maxWidth="sm" style={{ marginTop: "2rem" }}>
//         <Paper elevation={6} style={{ backgroundColor: "#2C2C2C", padding: "1.5rem" }}>
//           <Typography variant="h4" align="center" gutterBottom>
//             Chatbot Interface
//           </Typography>
//           <Divider style={{ backgroundColor: grey[700], marginBottom: "1rem" }} />
//           <Box
//             className="chatbox"
//             style={{
//               height: "300px",
//               overflowY: "auto",
//               padding: "1rem",
//               backgroundColor: "#1E1E1E",
//               borderRadius: "8px",
//               border: `1px solid ${grey[700]}`,
//             }}
//           >
//             <List>
//               {chatHistory.map((chat, index) => (
//                 <ListItem
//                   key={index}
//                   className={chat.sender === "user" ? "chat-user" : "chat-bot"}
//                   style={{
//                     justifyContent: chat.sender === "user" ? "flex-end" : "flex-start",
//                   }}
//                 >
//                   <ListItemText
//                     primary={
//                       <Typography
//                         variant="body1"
//                         style={{
//                           color: chat.sender === "user" ? blue[300] : grey[400],
//                           backgroundColor: chat.sender === "user" ? blue[800] : "#333333",
//                           padding: "0.5rem 1rem",
//                           borderRadius: "12px",
//                           maxWidth: "80%",
//                         }}
//                       >
//                         {chat.sender === "user" ? "You" : "Bot"}: {chat.text}
//                       </Typography>
//                     }
//                   />
//                 </ListItem>
//               ))}
//             </List>
//           </Box>
//           <Divider style={{ backgroundColor: grey[700], marginTop: "1rem" }} />
//           <Box mt={2}>
//             <form onSubmit={handleSubmit}>
//               <Box display="flex" gap={2}>
//                 <TextField
//                   variant="outlined"
//                   fullWidth
//                   label="Type your message..."
//                   value={message}
//                   onChange={(e) => setMessage(e.target.value)}
//                   InputLabelProps={{
//                     style: { color: grey[400] },
//                   }}
//                   InputProps={{
//                     style: { color: grey[100], backgroundColor: "#333333" },
//                   }}
//                 />
//                 <Button
//                   variant="contained"
//                   color="primary"
//                   type="submit"
//                   style={{ height: "100%" }}
//                 >
//                   Send
//                 </Button>
//               </Box>
//             </form>
//           </Box>
//         </Paper>
//       </Container>
//       <Footer />
//     </ThemeProvider>
//   );
// }

// export default App;
