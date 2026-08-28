// ============================================================
// RESULT PAGE JAVASCRIPT
// Lifestyle Disease Prediction System
// Groq AI Health Assistant + SHAP Visualization
// ============================================================


// ============================================================
// PAGE INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

    // Prediction form
    const form = document.getElementById("predictionForm");
    const button = document.getElementById("predictBtn");

    if (form && button) {

        form.addEventListener("submit", function () {

            button.disabled = true;

            button.innerHTML = `
                <span>
                    <i class="fa-solid fa-spinner fa-spin"></i>
                    Analyzing Health Data...
                </span>
                <i class="fa-solid fa-hourglass-half"></i>
            `;

        });

    }


    // Initialize SHAP chart
    initializeShapChart();


    // Chat input
    const chatInput = document.getElementById("chatInput");

    if (chatInput) {

        chatInput.addEventListener("keydown", function (event) {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendChatMessage();

            }

        });

    }


    // Chat button
    const sendButton =
        document.getElementById("chatSendBtn");

    if (sendButton) {

        sendButton.addEventListener("click", function () {

            sendChatMessage();

        });

    }


    // Scroll chat to bottom
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

        return;

    }


    let maxValue = 0;


    // Find maximum absolute SHAP value
    shapBars.forEach(function (bar) {

        const value =
            parseFloat(
                bar.getAttribute("data-value") || "0"
            );

        if (value > maxValue) {

            maxValue = value;

        }

    });


    if (maxValue <= 0) {

        return;

    }


    // Calculate relative width
    shapBars.forEach(function (bar) {

        const value =
            parseFloat(
                bar.getAttribute("data-value") || "0"
            );


        const percentage =
            (value / maxValue) * 50;


        bar.style.width =
            Math.min(percentage, 50) + "%";

    });

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

${reasons.map(function (reason) {

    return "- " + reason;

}).join("\n")}

Lifestyle Recommendations:

${tips.map(function (tip) {

    return "- " + tip;

}).join("\n")}
`;

}


// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHTML(text) {

    if (
        text === null ||
        text === undefined
    ) {

        return "";

    }


    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");

}


// ============================================================
// FORMAT INLINE MARKDOWN
// ============================================================

function formatInlineMarkdown(text) {

    let result =
        escapeHTML(text);


    // Bold: **text**
    result =
        result.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    // Italic: *text*
    result =
        result.replace(
            /(^|[^\*])\*([^*\n]+)\*(?!\*)/g,
            "$1<em>$2</em>"
        );


    // Inline code: `text`
    result =
        result.replace(
            /`([^`]+)`/g,
            "<code>$1</code>"
        );


    return result;

}


// ============================================================
// FORMAT GROQ AI RESPONSE
// ============================================================

function formatAIResponse(message) {

    if (
        message === null ||
        message === undefined
    ) {

        return "";

    }


    const text =
        String(message)
            .replace(/\r\n/g, "\n")
            .replace(/\r/g, "\n")
            .trim();


    if (!text) {

        return "";

    }


    const lines =
        text.split("\n");


    let html = "";

    let paragraph = [];

    let inTable = false;


    // --------------------------------------------------------
    // Flush paragraph
    // --------------------------------------------------------

    function flushParagraph() {

        if (!paragraph.length) {

            return;

        }


        const content =
            paragraph
                .join(" ")
                .trim();


        if (content) {

            html += `
                <p class="ai-paragraph">
                    ${formatInlineMarkdown(content)}
                </p>
            `;

        }


        paragraph = [];

    }


    // --------------------------------------------------------
    // Close table
    // --------------------------------------------------------

    function closeTable() {

        if (inTable) {

            html += `
                    </tbody>
                    </table>
                </div>
            `;

        }


        inTable = false;

    }


    // --------------------------------------------------------
    // Process lines
    // --------------------------------------------------------

    for (
        let i = 0;
        i < lines.length;
        i++
    ) {

        const line =
            lines[i].trim();


        // Empty line
        if (!line) {

            flushParagraph();

            if (inTable) {

                closeTable();

            }

            continue;

        }


        // ----------------------------------------------------
        // Markdown table
        // ----------------------------------------------------

        if (
            line.startsWith("|") &&
            line.endsWith("|")
        ) {

            flushParagraph();


            const cells =
                line
                    .split("|")
                    .slice(1, -1)
                    .map(function (cell) {

                        return cell.trim();

                    });


            // Separator row
            const isSeparator =
                cells.every(function (cell) {

                    return /^:?-{3,}:?$/.test(cell);

                });


            if (isSeparator) {

                continue;

            }


            // Start table
            if (!inTable) {

                html += `
                    <div class="ai-table-wrapper">
                        <table class="ai-table">
                            <thead>
                                <tr>
                `;


                cells.forEach(function (cell) {

                    html += `
                        <th>
                            ${formatInlineMarkdown(cell)}
                        </th>
                    `;

                });


                html += `
                                </tr>
                            </thead>
                            <tbody>
                `;


                inTable = true;

            }

            else {

                // Table row
                html += "<tr>";


                cells.forEach(function (cell) {

                    html += `
                        <td>
                            ${formatInlineMarkdown(cell)}
                        </td>
                    `;

                });


                html += "</tr>";

            }


            continue;

        }


        // Close table before normal content
        if (inTable) {

            closeTable();

        }


        // ----------------------------------------------------
        // H1
        // ----------------------------------------------------

        if (line.startsWith("# ")) {

            flushParagraph();

            html += `
                <h3 class="ai-heading">
                    ${formatInlineMarkdown(
                        line.substring(2)
                    )}
                </h3>
            `;

            continue;

        }


        // ----------------------------------------------------
        // H2
        // ----------------------------------------------------

        if (line.startsWith("## ")) {

            flushParagraph();

            html += `
                <h3 class="ai-heading">
                    ${formatInlineMarkdown(
                        line.substring(3)
                    )}
                </h3>
            `;

            continue;

        }


        // ----------------------------------------------------
        // H3
        // ----------------------------------------------------

        if (line.startsWith("### ")) {

            flushParagraph();

            html += `
                <h4 class="ai-subheading">
                    ${formatInlineMarkdown(
                        line.substring(4)
                    )}
                </h4>
            `;

            continue;

        }


        // ----------------------------------------------------
        // Bullet list
        // ----------------------------------------------------

        if (
            /^[-*•]\s+/.test(line)
        ) {

            flushParagraph();


            const content =
                line.replace(
                    /^[-*•]\s+/,
                    ""
                );


            html += `
                <div class="ai-list-item">
                    <span class="ai-bullet">•</span>
                    <span>
                        ${formatInlineMarkdown(content)}
                    </span>
                </div>
            `;


            continue;

        }


        // ----------------------------------------------------
        // Numbered list
        // ----------------------------------------------------

        if (
            /^\d+\.\s+/.test(line)
        ) {

            flushParagraph();


            const match =
                line.match(
                    /^(\d+)\.\s+(.*)$/
                );


            if (match) {

                html += `
                    <div class="ai-list-item numbered">

                        <span class="ai-number">
                            ${match[1]}
                        </span>

                        <span>
                            ${formatInlineMarkdown(
                                match[2]
                            )}
                        </span>

                    </div>
                `;

            }


            continue;

        }


        // ----------------------------------------------------
        // Blockquote
        // ----------------------------------------------------

        if (
            line.startsWith("> ")
        ) {

            flushParagraph();


            html += `
                <div class="ai-quote">
                    ${formatInlineMarkdown(
                        line.substring(2)
                    )}
                </div>
            `;


            continue;

        }


        // ----------------------------------------------------
        // Horizontal line
        // ----------------------------------------------------

        if (
            /^[-*_]{3,}$/.test(line)
        ) {

            flushParagraph();


            html += `
                <hr class="ai-divider">
            `;


            continue;

        }


        // ----------------------------------------------------
        // Normal text
        // ----------------------------------------------------

        paragraph.push(line);

    }


    // Flush remaining paragraph
    flushParagraph();


    // Close remaining table
    if (inTable) {

        closeTable();

    }


    return html;

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


    // Message row
    const messageRow =
        document.createElement("div");


    messageRow.className =
        sender === "user"
            ? "chat-message user-message"
            : "chat-message assistant-message";


    // Avatar
    const avatar =
        document.createElement("div");


    avatar.className =
        "chat-avatar";


    avatar.textContent =
        sender === "user"
            ? "👤"
            : "🤖";


    // Bubble
    const bubble =
        document.createElement("div");


    bubble.className =
        "chat-bubble";


    // User message
    if (sender === "user") {

        bubble.textContent =
            message;

    }


    // AI message
    else {

        bubble.innerHTML =
            formatAIResponse(message);

    }


    // Add elements
    messageRow.appendChild(avatar);

    messageRow.appendChild(bubble);

    chatMessages.appendChild(messageRow);


    // Scroll
    requestAnimationFrame(function () {

        chatMessages.scrollTop =
            chatMessages.scrollHeight;

    });

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


    const message =
        input.value.trim();


    if (!message) {

        return;

    }


    // Prevent duplicate requests
    if (sendButton.disabled) {

        return;

    }


    // Add user message
    addChatMessage(
        message,
        "user"
    );


    // Clear input
    input.value = "";


    // Disable button
    sendButton.disabled = true;

    sendButton.textContent =
        "Thinking...";


    // Show typing
    const typingId =
        showTypingIndicator();


    try {

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


        let data = {};


        try {

            data =
                await response.json();

        }

        catch (error) {

            throw new Error(
                "Invalid server response."
            );

        }


        // Remove typing
        removeTypingIndicator(
            typingId
        );


        // Server error
        if (!response.ok) {

            throw new Error(
                data.reply ||
                data.error ||
                "Unable to contact AI assistant."
            );

        }


        // Get reply
        const reply =
            data.reply ||
            data.message ||
            "I couldn't generate a response.";


        // Add AI response
        addChatMessage(
            reply,
            "assistant"
        );

    }

    catch (error) {

        console.error(
            "Chat error:",
            error
        );


        removeTypingIndicator(
            typingId
        );


        addChatMessage(

            "Sorry, I couldn't connect to the AI Health Assistant right now. Please check your Groq API configuration and make sure the Flask server is running.",

            "assistant"

        );

    }

    finally {

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


    const row =
        document.createElement("div");


    row.className =
        "chat-message assistant-message";


    row.id =
        id;


    const avatar =
        document.createElement("div");


    avatar.className =
        "chat-avatar";


    avatar.textContent =
        "🤖";


    const bubble =
        document.createElement("div");


    bubble.className =
        "chat-bubble";


    bubble.innerHTML = `
        <span class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </span>
    `;


    row.appendChild(avatar);

    row.appendChild(bubble);

    chatMessages.appendChild(row);


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
        document.getElementById(id);


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