# src/chat_widget.py
"""
Floating chat widget component - Alternative implementation
Use this for a more modern, floating chat experience like Messenger
"""

import streamlit as st
import streamlit.components.v1 as components

def render_floating_chat(chatbot, chat_history):
    """
    Render a floating chat widget that appears on the right side
    Similar to Facebook Messenger or Intercom
    """
    
    # Generate chat HTML
    messages_html = ""
    for msg in chat_history:
        if msg["role"] == "user":
            messages_html += f"""
            <div class="message user-message">
                <div class="message-content">{msg['content']}</div>
            </div>
            """
        else:
            messages_html += f"""
            <div class="message bot-message">
                <div class="bot-avatar">🤖</div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """
    
    # Get insights
    insights = chatbot.get_insights()
    
    # Complete HTML with JavaScript
    chat_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            
            /* Chat Toggle Button */
            #chat-toggle {{
                position: fixed;
                bottom: 24px;
                right: 24px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
                font-size: 28px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                z-index: 1000;
            }}
            
            #chat-toggle:hover {{
                transform: scale(1.1);
                box-shadow: 0 6px 30px rgba(102, 126, 234, 0.5);
            }}
            
            #chat-toggle.open {{
                background: #f44336;
            }}
            
            /* Badge for new messages */
            .badge {{
                position: absolute;
                top: -5px;
                right: -5px;
                background: #f44336;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
            }}
            
            /* Chat Container */
            #chat-container {{
                position: fixed;
                bottom: 24px;
                right: 24px;
                width: 380px;
                height: 600px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.15);
                display: none;
                flex-direction: column;
                overflow: hidden;
                z-index: 999;
                animation: slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            #chat-container.active {{
                display: flex;
            }}
            
            @keyframes slideUp {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            /* Chat Header */
            .chat-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            
            .chat-header-info {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            
            .chat-avatar {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.2);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            }}
            
            .chat-header-text h3 {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 2px;
            }}
            
            .chat-header-text p {{
                font-size: 12px;
                opacity: 0.9;
            }}
            
            .chat-minimize {{
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                font-size: 20px;
                padding: 5px;
                opacity: 0.8;
                transition: opacity 0.2s;
            }}
            
            .chat-minimize:hover {{
                opacity: 1;
            }}
            
            /* Insights Banner */
            .insights-banner {{
                background: #fff9e6;
                border-bottom: 1px solid #ffd966;
                padding: 12px 16px;
                font-size: 13px;
                color: #856404;
            }}
            
            .insights-banner strong {{
                display: block;
                margin-bottom: 4px;
            }}
            
            /* Quick Actions */
            .quick-actions {{
                padding: 12px 16px;
                background: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }}
            
            .quick-btn {{
                padding: 6px 12px;
                border-radius: 16px;
                border: 1px solid #667eea;
                background: white;
                color: #667eea;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            
            .quick-btn:hover {{
                background: #667eea;
                color: white;
            }}
            
            /* Messages Area */
            .chat-messages {{
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }}
            
            .chat-messages::-webkit-scrollbar {{
                width: 6px;
            }}
            
            .chat-messages::-webkit-scrollbar-thumb {{
                background: #cbd5e0;
                border-radius: 3px;
            }}
            
            /* Messages */
            .message {{
                margin-bottom: 16px;
                display: flex;
                animation: fadeIn 0.3s ease;
            }}
            
            @keyframes fadeIn {{
                from {{
                    opacity: 0;
                    transform: translateY(10px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            .bot-message {{
                flex-direction: row;
                gap: 8px;
            }}
            
            .user-message {{
                flex-direction: row-reverse;
            }}
            
            .bot-avatar {{
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }}
            
            .message-content {{
                padding: 12px 16px;
                border-radius: 16px;
                max-width: 75%;
                word-wrap: break-word;
                line-height: 1.5;
                font-size: 14px;
            }}
            
            .bot-message .message-content {{
                background: white;
                color: #2d3748;
                border-bottom-left-radius: 4px;
            }}
            
            .user-message .message-content {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-right-radius: 4px;
            }}
            
            /* Empty State */
            .empty-state {{
                text-align: center;
                padding: 40px 20px;
                color: #718096;
            }}
            
            .empty-state-icon {{
                font-size: 48px;
                margin-bottom: 16px;
            }}
            
            .empty-state h4 {{
                font-size: 16px;
                margin-bottom: 8px;
                color: #2d3748;
            }}
            
            .empty-state p {{
                font-size: 14px;
                margin-bottom: 20px;
            }}
            
            .example-questions {{
                text-align: left;
                background: white;
                padding: 16px;
                border-radius: 12px;
                margin-top: 16px;
            }}
            
            .example-questions div {{
                padding: 8px;
                margin: 4px 0;
                cursor: pointer;
                border-radius: 6px;
                transition: background 0.2s;
            }}
            
            .example-questions div:hover {{
                background: #f7fafc;
            }}
            
            /* Input Area */
            .chat-input {{
                padding: 16px;
                background: white;
                border-top: 1px solid #e9ecef;
            }}
            
            .input-wrapper {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            
            #message-input {{
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #e2e8f0;
                border-radius: 24px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s;
            }}
            
            #message-input:focus {{
                border-color: #667eea;
            }}
            
            #send-btn {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.2s;
            }}
            
            #send-btn:hover {{
                transform: scale(1.1);
            }}
            
            #send-btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            
            /* Typing Indicator */
            .typing-indicator {{
                display: none;
                align-items: center;
                gap: 8px;
                padding: 12px;
            }}
            
            .typing-indicator.active {{
                display: flex;
            }}
            
            .typing-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #cbd5e0;
                animation: typing 1.4s infinite;
            }}
            
            .typing-dot:nth-child(2) {{
                animation-delay: 0.2s;
            }}
            
            .typing-dot:nth-child(3) {{
                animation-delay: 0.4s;
            }}
            
            @keyframes typing {{
                0%, 60%, 100% {{
                    transform: translateY(0);
                }}
                30% {{
                    transform: translateY(-10px);
                }}
            }}
            
            /* Mobile Responsive */
            @media (max-width: 768px) {{
                #chat-container {{
                    width: 100%;
                    height: 100%;
                    bottom: 0;
                    right: 0;
                    border-radius: 0;
                }}
                
                #chat-toggle {{
                    bottom: 16px;
                    right: 16px;
                    width: 56px;
                    height: 56px;
                }}
            }}
        </style>
    </head>
    <body>
        <!-- Chat Toggle Button -->
        <button id="chat-toggle" onclick="toggleChat()">
            <span id="chat-icon">💬</span>
        </button>
        
        <!-- Chat Container -->
        <div id="chat-container">
            <!-- Header -->
            <div class="chat-header">
                <div class="chat-header-info">
                    <div class="chat-avatar">🤖</div>
                    <div class="chat-header-text">
                        <h3>Dashboard Assistant</h3>
                        <p>Always here to help</p>
                    </div>
                </div>
                <button class="chat-minimize" onclick="toggleChat()">✕</button>
            </div>
            
            <!-- Insights Banner -->
            <div class="insights-banner">
                <strong>💡 Quick Insights:</strong>
                {insights[:100]}...
            </div>
            
            <!-- Quick Actions -->
            <div class="quick-actions">
                <button class="quick-btn" onclick="sendQuickMessage('What are the total sales?')">📊 Sales</button>
                <button class="quick-btn" onclick="sendQuickMessage('Compare regions')">🌍 Regions</button>
                <button class="quick-btn" onclick="sendQuickMessage('Show top 5 products')">🏆 Top 5</button>
                <button class="quick-btn" onclick="sendQuickMessage('What\\'s the trend?')">📈 Trend</button>
            </div>
            
            <!-- Messages -->
            <div class="chat-messages" id="messages">
                {messages_html if messages_html else '''
                <div class="empty-state">
                    <div class="empty-state-icon">👋</div>
                    <h4>Hi! I'm your dashboard assistant</h4>
                    <p>Ask me anything about your sales data</p>
                    <div class="example-questions">
                        <div onclick="sendQuickMessage('What are the total sales?')">💰 What are the total sales?</div>
                        <div onclick="sendQuickMessage('Compare regions')">🌍 Compare regions</div>
                        <div onclick="sendQuickMessage('Show top 5 products')">🏆 Show top 5 products</div>
                        <div onclick="sendQuickMessage('What\\'s the sales trend?')">📈 What's the sales trend?</div>
                    </div>
                </div>
                '''}
            </div>
            
            <!-- Typing Indicator -->
            <div class="typing-indicator" id="typing">
                <div class="bot-avatar">🤖</div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            
            <!-- Input -->
            <div class="chat-input">
                <div class="input-wrapper">
                    <input 
                        type="text" 
                        id="message-input" 
                        placeholder="Type your question..."
                        onkeypress="handleKeyPress(event)"
                    />
                    <button id="send-btn" onclick="sendMessage()">📤</button>
                </div>
            </div>
        </div>
        
        <script>
            function toggleChat() {{
                const container = document.getElementById('chat-container');
                const toggle = document.getElementById('chat-toggle');
                const icon = document.getElementById('chat-icon');
                
                container.classList.toggle('active');
                toggle.classList.toggle('open');
                
                if (container.classList.contains('active')) {{
                    icon.textContent = '✕';
                    document.getElementById('message-input').focus();
                }} else {{
                    icon.textContent = '💬';
                }}
            }}
            
            function handleKeyPress(event) {{
                if (event.key === 'Enter') {{
                    sendMessage();
                }}
            }}
            
            function sendQuickMessage(message) {{
                document.getElementById('message-input').value = message;
                sendMessage();
            }}
            
            function sendMessage() {{
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                
                if (!message) return;
                
                // This would integrate with Streamlit
                // For now, we'll show a placeholder
                addMessage('user', message);
                input.value = '';
                
                // Show typing indicator
                showTyping();
                
                // Simulate bot response
                setTimeout(() => {{
                    hideTyping();
                    addMessage('bot', 'This message would come from the chatbot. Integrate with Streamlit callbacks for real responses.');
                }}, 1000);
            }}
            
            function addMessage(type, content) {{
                const messagesDiv = document.getElementById('messages');
                
                // Remove empty state if exists
                const emptyState = messagesDiv.querySelector('.empty-state');
                if (emptyState) {{
                    emptyState.remove();
                }}
                
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{type}}-message`;
                
                if (type === 'bot') {{
                    messageDiv.innerHTML = `
                        <div class="bot-avatar">🤖</div>
                        <div class="message-content">${{content}}</div>
                    `;
                }} else {{
                    messageDiv.innerHTML = `
                        <div class="message-content">${{content}}</div>
                    `;
                }}
                
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}
            
            function showTyping() {{
                document.getElementById('typing').classList.add('active');
                const messagesDiv = document.getElementById('messages');
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}
            
            function hideTyping() {{
                document.getElementById('typing').classList.remove('active');
            }}
        </script>
    </body>
    </html>
    """
    
    # Render in Streamlit
    components.html(chat_html, height=0, scrolling=False)


def inject_chat_styles():
    """Inject only the floating button (for main app.py integration)"""
    st.markdown("""
    <style>
        /* Floating chat button that opens sidebar */
        .floating-chat-btn {
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            font-size: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1000;
            animation: pulse 2s infinite;
        }
        
        .floating-chat-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 30px rgba(102, 126, 234, 0.5);
        }
        
        @keyframes pulse {
            0%, 100% {
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            }
            50% {
                box-shadow: 0 4px 30px rgba(102, 126, 234, 0.6);
            }
        }
        
        /* Notification badge */
        .chat-badge {
            position: fixed;
            bottom: 70px;
            right: 70px;
            background: #f44336;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
            z-index: 1001;
        }
    </style>
    """, unsafe_allow_html=True)