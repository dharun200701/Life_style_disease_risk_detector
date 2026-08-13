// ============================================================
// RESULT PAGE JAVASCRIPT
// ============================================================


// ============================================================
// PAGE INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

    // --------------------------------------------------------
    // Prediction form
    // --------------------------------------------------------

    const form =
        document.getElementById("predictionForm");

    const button =
        document.getElementById("predictBtn");


    if (form && button) {

        form.addEventListener(
            "submit",
            function () {

                button.disabled = true;

                button.innerHTML = `
                    <span>
                        <i class="fa-solid fa-spinner fa-spin"></i>
                        Analyzing Health Data...
                    </span>

                    <i class="fa-solid fa-hourglass-half"></i>
                `;

            }
        );

    }


    // --------------------------------------------------------
    // Initialize SHAP chart
    // --------------------------------------------------------

    initializeShapChart();


    // --------------------------------------------------------
    // Chat input - Enter key
    // --------------------------------------------------------

    const chatInput =
        document.getElementById("chatInput");


    if (chatInput) {

        chatInput.addEventListener(
            "keydown",
            function (event) {

                if (event.key === "Enter") {

                    event.preventDefault();

                    sendChatMessage();

                }

            }
        );

    }


    // --------------------------------------------------------
    // Scroll chat to bottom
    // --------------------------------------------------------

    const chatMessages =
        document.getElementById("chatMessages");


    if (chatMessages) {

        chatMessages.scrollTop =
            chatMessages.scrollHeight;

    }

});


// ============================================================
// SHAP BAR CHART
// ============================================================

function initializeShapChart() {

    const shapBars =
        document.querySelectorAll(".shap-bar");


    if (!shapBars.length) {

        console.log(
            "No SHAP bars found."
        );

        return;

    }


    let maxValue = 0;


    // --------------------------------------------------------
    // Find largest absolute SHAP value
    // --------------------------------------------------------

    shapBars.forEach(
        function (bar) {

            const value =
                parseFloat(
                    bar.getAttribute(
                        "data-value"
                    ) || "0"
                );


            if (value > maxValue) {

                maxValue = value;

            }

        }
    );


    if (maxValue === 0) {

        return;

    }


    // --------------------------------------------------------
    // Set bar widths
    // --------------------------------------------------------

    shapBars.forEach(
        function (bar) {

            const value =
                parseFloat(
                    bar.getAttribute(
                        "data-value"
                    ) || "0"
                );


            /*
             * Maximum bar length is 50%
             * because the chart grows from
             * the center zero line.
             */

            const percentage =
                (value / maxValue) * 50;


            bar.style.width =
                Math.min(
                    percentage,
                    50
                ) + "%";

        }
    );

}


// ============================================================
// GET PREDICTION CONTEXT
// ============================================================

function getPredictionContext() {

    const context =
        window.predictionContext || {};


    const reasons =
        Array.isArray(context.reasons)
            ? context.reasons
            : [];


    const tips =
        Array.isArray(context.tips)
            ? context.tips
            : [];


    return `
Prediction: ${context.prediction || "Not available"}

Risk Level: ${context.risk || "Not available"}

Confidence: ${context.confidence || "Not available"}%

Risk Score: ${context.score || "Not available"}/100


Key Factors:

${reasons.map(
    function (reason) {

        return "- " + reason;

    }
).join("\n")}


Lifestyle Recommendations:

${tips.map(
    function (tip) {

        return "- " + tip;

    }
).join("\n")}
`;

}


// ============================================================
// ADD CHAT MESSAGE
// ============================================================

function addChatMessage(
    message,
    sender
) {

    const chatMessages =
        document.getElementById(
            "chatMessages"
        );


    if (!chatMessages) {

        return;

    }


    // --------------------------------------------------------
    // Message row
    // --------------------------------------------------------

    const messageRow =
        document.createElement(
            "div"
        );


    messageRow.className =
        sender === "user"
            ? "chat-message user-message"
            : "chat-message assistant-message";


    // --------------------------------------------------------
    // Avatar
    // --------------------------------------------------------

    const avatar =
        document.createElement(
            "div"
        );


    avatar.className =
        "chat-avatar";


    avatar.textContent =
        sender === "user"
            ? "👤"
            : "🤖";


    // --------------------------------------------------------
    // Message bubble
    // --------------------------------------------------------

    const bubble =
        document.createElement(
            "div"
        );


    bubble.className =
        "chat-bubble";


    /*
     * textContent is intentionally used here.
     * This prevents AI responses from being
     * interpreted as HTML.
     */

    bubble.textContent =
        message;


    // --------------------------------------------------------
    // Build message
    // --------------------------------------------------------

    messageRow.appendChild(
        avatar
    );

    messageRow.appendChild(
        bubble
    );

    chatMessages.appendChild(
        messageRow
    );


    // --------------------------------------------------------
    // Scroll to latest message
    // --------------------------------------------------------

    chatMessages.scrollTop =
        chatMessages.scrollHeight;

}


// ============================================================
// SEND CHAT MESSAGE
// ============================================================

async function sendChatMessage() {

    const input =
        document.getElementById(
            "chatInput"
        );


    const sendButton =
        document.getElementById(
            "chatSendBtn"
        );


    const chatMessages =
        document.getElementById(
            "chatMessages"
        );


    if (
        !input ||
        !sendButton ||
        !chatMessages
    ) {

        return;

    }


    // --------------------------------------------------------
    // Get message
    // --------------------------------------------------------

    const message =
        input.value.trim();


    if (!message) {

        return;

    }


    // --------------------------------------------------------
    // Add user's message
    // --------------------------------------------------------

    addChatMessage(
        message,
        "user"
    );


    // Clear input

    input.value = "";


    // --------------------------------------------------------
    // Disable send button
    // --------------------------------------------------------

    sendButton.disabled =
        true;


    sendButton.textContent =
        "Thinking...";


    // --------------------------------------------------------
    // Show typing indicator
    // --------------------------------------------------------

    const typingId =
        showTypingIndicator();


    try {

        // ----------------------------------------------------
        // Send request to Flask
        // ----------------------------------------------------

        const response =
            await fetch(
                "/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({

                            message:
                                message,

                            context:
                                getPredictionContext()

                        })

                }
            );


        // ----------------------------------------------------
        // Read response
        // ----------------------------------------------------

        const data =
            await response.json();


        // Remove typing animation

        removeTypingIndicator(
            typingId
        );


        // ----------------------------------------------------
        // Check server response
        // ----------------------------------------------------

        if (!response.ok) {

            throw new Error(
                data.reply ||
                "Unable to contact the AI assistant."
            );

        }


        // ----------------------------------------------------
        // Add AI response
        // ----------------------------------------------------

        addChatMessage(
            data.reply ||
            "I couldn't generate a response.",
            "assistant"
        );


    } catch (error) {

        console.error(
            "Chat error:",
            error
        );


        // Remove typing indicator

        removeTypingIndicator(
            typingId
        );


        // Show friendly error

        addChatMessage(

            "Sorry, I couldn't connect to the AI Health Assistant right now. Please check your Groq configuration.",

            "assistant"

        );

    } finally {

        // ----------------------------------------------------
        // Enable button again
        // ----------------------------------------------------

        sendButton.disabled =
            false;


        sendButton.textContent =
            "Send";


        input.focus();

    }

}


// ============================================================
// TYPING INDICATOR
// ============================================================

function showTypingIndicator() {

    const chatMessages =
        document.getElementById(
            "chatMessages"
        );


    if (!chatMessages) {

        return null;

    }


    const id =
        "typing-" +
        Date.now();


    // --------------------------------------------------------
    // Row
    // --------------------------------------------------------

    const row =
        document.createElement(
            "div"
        );


    row.className =
        "chat-message assistant-message";


    row.id =
        id;


    // --------------------------------------------------------
    // Avatar
    // --------------------------------------------------------

    const avatar =
        document.createElement(
            "div"
        );


    avatar.className =
        "chat-avatar";


    avatar.textContent =
        "🤖";


    // --------------------------------------------------------
    // Bubble
    // --------------------------------------------------------

    const bubble =
        document.createElement(
            "div"
        );


    bubble.className =
        "chat-bubble";


    bubble.innerHTML = `
        <span class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </span>
    `;


    // --------------------------------------------------------
    // Build
    // --------------------------------------------------------

    row.appendChild(
        avatar
    );

    row.appendChild(
        bubble
    );


    chatMessages.appendChild(
        row
    );


    // --------------------------------------------------------
    // Scroll
    // --------------------------------------------------------

    chatMessages.scrollTop =
        chatMessages.scrollHeight;


    return id;

}


// ============================================================
// REMOVE TYPING INDICATOR
// ============================================================

function removeTypingIndicator(id) {

    if (!id) {

        return;

    }


    const element =
        document.getElementById(
            id
        );


    if (element) {

        element.remove();

    }

}


// ============================================================
// SUGGESTED QUESTION
// ============================================================

function askSuggestedQuestion(
    question
) {

    const input =
        document.getElementById(
            "chatInput"
        );


    if (!input) {

        return;

    }


    input.value =
        question;


    sendChatMessage();

}