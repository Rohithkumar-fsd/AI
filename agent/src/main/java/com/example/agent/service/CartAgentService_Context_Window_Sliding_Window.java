package com.example.agent.service;

import com.example.agent.tools.CartAgentTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.*;
import org.springframework.stereotype.Service;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.AssistantMessage;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class CartAgentService_Context_Window_Sliding_Window {

    private static final int MAX_MESSAGES = 16;
    private static final int MAX_WINDOW_CHARS = 8000;

    private static final String SYSTEM_INSTRUCTIONS = """
            You are an independent Shopping Cart AI Agent.
            - Always use the provided cart tools (addItemToCart, removeItemFromCart, getCartItems, calculateCartTotal) to read or modify cart state.
            - Use lookupProductInfo when checking product prices or details.
            - Never guess cart contents or totals without calling the appropriate tool.
            - Resolve references like "that item" or "one more" using prior turns in the conversation.
            - Respond concisely and clearly.
            """;

    private final ChatClient chatClient;
    private final Map<String,List<Message>>sessions=new ConcurrentHashMap<>();

    public CartAgentService_Context_Window_Sliding_Window(
            ChatClient.Builder chatClientBuilder,
            CartAgentTools cartAgentTools) {
        this.chatClient = chatClientBuilder
                .defaultSystem(SYSTEM_INSTRUCTIONS)
                .defaultTools(cartAgentTools)
                .build();
    }

    public String chat(String sessionId, String userMessage) {
        String activeSession = (sessionId == null || sessionId.isBlank())
                ? "default-session"
                : sessionId.trim();

        List<Message> history = sessions.computeIfAbsent(activeSession, k -> new ArrayList<>());

        synchronized (history) {
            history.add(new UserMessage(userMessage));
            applySlidingWindow(history);

            String reply = chatClient.prompt()
                    .messages(history)
                    .call()
                    .content();

            if (reply != null) {
                history.add(new AssistantMessage(reply));
                applySlidingWindow(history);
            }
            return reply;
        }
    }

    public void resetSession(String sessionId) {
        if (sessionId != null && !sessionId.isBlank()) {
            sessions.remove(sessionId.trim());
        }
    }
    public void applySlidingWindow(List<Message>history)
    {
         while(history.size()>MAX_MESSAGES)
         {
             history.remove(0);
         }
    }

}