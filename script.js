/**
 * RagAda - Premium Review Analysis Assistant
 * Ultra-professional UI with advanced animations and transitions
 * Enhanced for a truly mind-blowing experience
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const messagesContainer = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const themeToggle = document.querySelector('.theme-toggle');
    const exampleQueries = document.querySelectorAll('.example-queries li');
    
    // State
    let darkMode = false;
    let isProcessingQuery = false;
    
    // Initialize
    initTheme();
    initTextareaResize();
    initEventListeners();
    initRippleEffects();
    showWelcomeMessage();

    // Function to initialize theme based on user preference
    function initTheme() {
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('ragada-theme');
        if (savedTheme === 'dark') {
            enableDarkMode();
        }
    }
    
    // Function to initialize textarea auto-resize
    function initTextareaResize() {
        userInput.addEventListener('input', () => {
            // Reset height to auto to get the correct scrollHeight
            userInput.style.height = 'auto';
            
            // Set new height based on content (with a max-height limit)
            const newHeight = Math.min(userInput.scrollHeight, 150);
            userInput.style.height = `${newHeight}px`;
        });
    }
    
    // Function to set up all event listeners
    function initEventListeners() {
        // Send message on button click
        sendButton.addEventListener('click', handleSendMessage);
        
        // Send message on Enter key (without shift)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
        
        // Toggle theme
        themeToggle.addEventListener('click', toggleTheme);
        
        // Example query click
        exampleQueries.forEach(query => {
            query.addEventListener('click', () => {
                const queryText = query.getAttribute('data-query');
                if (queryText) {
                    userInput.value = queryText;
                    // Trigger input event to resize textarea
                    userInput.dispatchEvent(new Event('input'));
                    // Highlight the selected query
                    highlightSelectedQuery(query);
                    // Focus on input
                    userInput.focus();
                }
            });
        });
    }
    
    // Function to highlight selected example query
    function highlightSelectedQuery(selectedQuery) {
        // Remove highlight from all queries
        exampleQueries.forEach(query => {
            query.style.borderLeft = 'none';
            query.style.transform = '';
            query.style.boxShadow = '';
        });
        
        // Add highlight to selected query with enhanced animation
        selectedQuery.style.transition = 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)';
        selectedQuery.style.borderLeft = '4px solid var(--primary-color)';
        selectedQuery.style.transform = 'translateY(-3px) scale(1.02)';
        selectedQuery.style.boxShadow = '0 8px 16px var(--shadow-color)';
        
        // Add subtle pulse animation
        const pulse = () => {
            selectedQuery.animate([
                { boxShadow: '0 8px 16px var(--shadow-color)' },
                { boxShadow: '0 8px 24px var(--shadow-color)' },
                { boxShadow: '0 8px 16px var(--shadow-color)' }
            ], {
                duration: 1000,
                easing: 'ease-in-out'
            });
        };
        
        // Run pulse animation once
        pulse();
        
        // Remove highlight after a delay with smooth transition back
        setTimeout(() => {
            selectedQuery.style.transition = 'all 0.5s ease-out';
            selectedQuery.style.borderLeft = 'none';
            selectedQuery.style.transform = '';
            selectedQuery.style.boxShadow = '';
        }, 2000);
    }
    
    // Function to toggle between light/dark themes
    function toggleTheme() {
        if (darkMode) {
            disableDarkMode();
        } else {
            enableDarkMode();
        }
    }
    
    // Function to enable dark mode
    function enableDarkMode() {
        document.body.classList.add('dark-theme');
        darkMode = true;
        localStorage.setItem('ragada-theme', 'dark');
    }
    
    // Function to disable dark mode
    function disableDarkMode() {
        document.body.classList.remove('dark-theme');
        darkMode = false;
        localStorage.setItem('ragada-theme', 'light');
    }
    
    // Function to handle message sending
    function handleSendMessage() {
        const message = userInput.value.trim();
        if (message && !isProcessingQuery) {
            // Add user message to chat
            addMessageToChat('user', message);
            
            // Clear input and reset height
            userInput.value = '';
            userInput.style.height = 'auto';
            
            // Add loading indicator
            addLoadingIndicator();
            
            // Set processing state
            isProcessingQuery = true;
            
            // Process the query
            processQuery(message);
        }
    }
    
    // Function to process the query (sends to backend)
    async function processQuery(query) {
        try {
            // Make API call to backend
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Remove loading indicator
            removeLoadingIndicator();
            
            // Add assistant response to chat
            addMessageToChat('assistant', data.response);
            
        } catch (error) {
            console.error('Error processing query:', error);
            
            // Remove loading indicator
            removeLoadingIndicator();
            
            // Add error message to chat
            addMessageToChat('assistant', 'Sorry, there was an error processing your request. Please try again.');
        }
        
        // Reset processing state
        isProcessingQuery = false;
        
        // Scroll to bottom
        scrollToBottom();
    }
    
    // Function to add message to chat
    function addMessageToChat(type, content) {
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        
        // Create message content
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content', 'markdown');
        messageContent.style.opacity = '0';
        messageContent.style.transform = 'translateY(20px) scale(0.95)';
        
        // Format the content with Markdown if it's an assistant response
        if (type === 'assistant') {
            messageContent.innerHTML = formatMarkdown(content);
            
            // Add subtle highlight effect to code blocks
            setTimeout(() => {
                const codeBlocks = messageContent.querySelectorAll('pre code');
                codeBlocks.forEach(block => {
                    block.style.transition = 'all 0.3s ease';
                    block.addEventListener('mouseover', () => {
                        block.style.boxShadow = '0 5px 15px var(--shadow-color)';
                        block.style.transform = 'translateY(-2px)';
                    });
                    block.addEventListener('mouseout', () => {
                        block.style.boxShadow = '';
                        block.style.transform = '';
                    });
                });
            }, 100);
        } else {
            messageContent.textContent = content;
        }
        
        // Add timestamp with fade-in effect
        const timestamp = document.createElement('div');
        timestamp.classList.add('message-timestamp');
        timestamp.textContent = getCurrentTime();
        timestamp.style.opacity = '0';
        messageContent.appendChild(timestamp);
        
        // Append content to message
        messageDiv.appendChild(messageContent);
        
        // Append message to chat
        messagesContainer.appendChild(messageDiv);
        
        // Apply enhanced entrance animation
        setTimeout(() => {
            messageContent.style.transition = 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
            messageContent.style.opacity = '1';
            messageContent.style.transform = 'translateY(0) scale(1)';
            
            // Fade in timestamp after message appears
            setTimeout(() => {
                timestamp.style.transition = 'opacity 0.3s ease';
                timestamp.style.opacity = '1';
            }, 300);
        }, 50);
        
        // Scroll to bottom with smooth animation
        scrollToBottom();
    }
    
    // Function to add loading indicator
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'assistant', 'loading-message');
        
        const loadingContent = document.createElement('div');
        loadingContent.classList.add('message-content');
        
        const loadingContainer = document.createElement('div');
        loadingContainer.classList.add('loading-container');
        
        const loadingDots = document.createElement('div');
        loadingDots.classList.add('loading-dots');
        
        // Create 3 loading dots
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.classList.add('loading-dot');
            loadingDots.appendChild(dot);
        }
        
        loadingContainer.appendChild(loadingDots);
        loadingContent.appendChild(loadingContainer);
        loadingDiv.appendChild(loadingContent);
        messagesContainer.appendChild(loadingDiv);
        
        scrollToBottom();
    }
    
    // Function to remove loading indicator
    function removeLoadingIndicator() {
        const loadingMessage = document.querySelector('.loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    // Function to get current time in elegant format
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + 
               ' · ' + now.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
    
    // Function to initialize ripple effects on interactive elements
    function initRippleEffects() {
        // Add ripple effect to send button
        sendButton.addEventListener('click', createRippleEffect);
        
        // Add ripple effect to theme toggle
        themeToggle.addEventListener('click', createRippleEffect);
        
        // Add ripple effect to example queries
        exampleQueries.forEach(query => {
            query.addEventListener('click', createRippleEffect);
        });
    }
    
    // Function to create ripple effect
    function createRippleEffect(event) {
        const button = event.currentTarget;
        
        // Remove any existing ripples
        const existingRipple = button.querySelector('.ripple');
        if (existingRipple) {
            existingRipple.remove();
        }
        
        // Create ripple element
        const ripple = document.createElement('span');
        ripple.className = 'ripple';
        button.appendChild(ripple);
        
        // Get ripple size (use largest dimension to ensure circle covers element)
        const buttonRect = button.getBoundingClientRect();
        const diameter = Math.max(buttonRect.width, buttonRect.height);
        const radius = diameter / 2;
        
        // Position ripple relative to click position
        ripple.style.width = ripple.style.height = `${diameter}px`;
        ripple.style.left = `${event.clientX - buttonRect.left - radius}px`;
        ripple.style.top = `${event.clientY - buttonRect.top - radius}px`;
        
        // Add active class to start animation
        ripple.classList.add('active');
        
        // Remove ripple after animation completes
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }
    
    // Function to show welcome message
    function showWelcomeMessage() {
        setTimeout(() => {
            addMessageToChat('assistant', '# Welcome to Reviews Info 👋\n\nI\'m your review analysis assistant, ready to help you gain insights from customer feedback. Try asking me about reviews using the example queries, or type your own question!\n\n**Tip**: You can use natural language to ask about reviews based on date, rating, or sentiment.');
        }, 500);
    }
    
    // Function to scroll to bottom of messages with smooth animation
    function scrollToBottom() {
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    // Function to format markdown to HTML (enhanced implementation)
    function formatMarkdown(text) {
        if (!text) return '';
        
        // Replace code blocks with syntax highlighting preparation
        text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1 highlight-block">$2</code></pre>');
        
        // Replace inline code
        text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
        
        // Replace bold text
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="bold-text">$1</strong>');
        
        // Replace italic text
        text = text.replace(/\*(.*?)\*/g, '<em class="italic-text">$1</em>');
        
        // Replace headers with classes
        text = text.replace(/^### (.*$)/gm, '<h3 class="md-heading md-h3">$1</h3>');
        text = text.replace(/^## (.*$)/gm, '<h2 class="md-heading md-h2">$1</h2>');
        text = text.replace(/^# (.*$)/gm, '<h1 class="md-heading md-h1">$1</h1>');
        
        // Replace lists with better formatting
        text = text.replace(/^\* (.*$)/gm, '<li class="md-list-item">$1</li>');
        text = text.replace(/^\d+\. (.*$)/gm, '<li class="md-list-item md-numbered">$1</li>');
        
        // Replace links with styled class
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="md-link" target="_blank">$1</a>');
        
        // Replace blockquotes
        text = text.replace(/^> (.*$)/gm, '<blockquote class="md-blockquote">$1</blockquote>');
        
        // Replace paragraphs
        text = text.replace(/^(?!<[a-z][a-z0-9]*>)(.+)$/gm, '<p class="md-paragraph">$1</p>');
        
        return text;
    }

    // Simulate a welcome message
    setTimeout(() => {
        // This is just for the demo. In production, this would come from the backend.
        const welcomeMessage = "I'm ready to analyze your reviews! You can ask me about specific ratings, time periods, or request summaries of customer feedback.";
        addMessageToChat('assistant', welcomeMessage);
    }, 500);
});
