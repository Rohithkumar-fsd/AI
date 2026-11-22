package com.example.agent.service;

import com.example.agent.tools.CartAgentTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class CartAgentService_Context_Window_ToolOutputPruning {

    private static final int MAX_TOOL_OUTPUT_CHARS = 500;

    private static final String SYSTEM_INSTRUCTIONS = """
            You are an independent Shopping Cart AI Agent.
            - Always use the provided cart tools (addItemToCart, removeItemFromCart, getCartItems, calculateCartTotal) to read or modify cart state.
            - Use lookupProductInfo when checking product prices or details.
            - Never guess cart contents or totals without calling the appropriate tool.
            - Resolve references like "that item" or "one more" using prior turns in the conversation.
            - Respond concisely and clearly.
            """;

    private final ChatClient chatClient;
    private final Map<String, List<Message>> sessions = new ConcurrentHashMap<>();

    public CartAgentService_Context_Window_ToolOutputPruning(
            ChatClient.Builder chatClientBuilder,
            CartAgentTools cartAgentTools) {
        this.chatClient = chatClientBuilder
                .defaultSystem(SYSTEM_INSTRUCTIONS)
                .defaultTools(cartAgentTools)
                .build();
    }

    public String chat(String sessionId, String userMessage) {
        String key = (sessionId == null || sessionId.isBlank()) ? "default-session" : sessionId.trim();
        List<Message> history = sessions.computeIfAbsent(key, k -> new ArrayList<>());

        synchronized (history) {
            history.add(new UserMessage(userMessage));

            String reply = chatClient.prompt()
                    .messages(List.copyOf(history))
                    .call()
                    .content();

            if (reply != null) {
                String prunedReply = pruneToolOutput(reply);
                history.add(new AssistantMessage(prunedReply));
            }
            return reply;
        }
    }

    public void resetSession(String sessionId) {
        if (sessionId != null && !sessionId.isBlank()) {
            sessions.remove(sessionId.trim());
        }
    }

    private String pruneToolOutput(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        String cleaned = text.trim().replaceAll("\\s+", " ");
        if (cleaned.length() <= MAX_TOOL_OUTPUT_CHARS) {
            return cleaned;
        }
        int extraChars = cleaned.length() - MAX_TOOL_OUTPUT_CHARS;
        return cleaned.substring(0, MAX_TOOL_OUTPUT_CHARS) + " ... [pruned " + extraChars + " chars]";
    }
}