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
public class CartAgentService_Context_Window_Summarization {

    private static final int RECENT_MESSAGE_CAP = 12;
    private static final int COMPACT_BATCH_SIZE = 6;

    private static final String SYSTEM_INSTRUCTIONS = """
            You are an independent Shopping Cart AI Agent.
            - Always use the provided cart tools (addItemToCart, removeItemFromCart, getCartItems, calculateCartTotal) to read or modify cart state.
            - Use lookupProductInfo when checking product prices or details.
            - Never guess cart contents or totals without calling the appropriate tool.
            - Resolve references like "that item" or "one more" using prior turns in the conversation.
            - Respond concisely and clearly.
            """;

    private static final String SUMMARIZER_PROMPT = """
            Update the rolling conversation summary using the prior summary and older chat messages.
            Preserve concrete entities, item names, quantities, prices, cart actions, and user constraints.
            Drop filler and repetition. Keep it under 120 words.
            """;

    private static final class SessionState {
        String rollingSummary = "";
        final List<Message> recentMessages = new ArrayList<>();
    }

    private final ChatClient chatClient;
    private final Map<String,SessionState>sessions=new ConcurrentHashMap<>();

    public CartAgentService_Context_Window_Summarization(
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

        SessionState state = sessions.computeIfAbsent(activeSession, k -> new SessionState());

        synchronized (state) {
            state.recentMessages.add(new UserMessage(userMessage));
            compactIfNeeded(state);

            String reply = chatClient.prompt()
                    .messages(buildPromptMessages(state))
                    .call()
                    .content();

            if (reply != null) {
                state.recentMessages.add(new AssistantMessage(reply));
                compactIfNeeded(state);
            }
            return reply;
        }
    }

    public void resetSession(String sessionId) {
        if (sessionId != null && !sessionId.isBlank()) {
            sessions.remove(sessionId.trim());
        }
    }
    private void compactIfNeeded(SessionState state) {
        if (state.recentMessages.size() <= RECENT_MESSAGE_CAP) {
            return;
        }
        int batchCount = Math.min(COMPACT_BATCH_SIZE, state.recentMessages.size());
        List<Message> olderBatch = new ArrayList<>(state.recentMessages.subList(0, batchCount));
        state.recentMessages.subList(0, batchCount).clear();

        StringBuilder transcript = new StringBuilder();
        if (!state.rollingSummary.isBlank()) {
            transcript.append("Previous summary:\n").append(state.rollingSummary).append("\n\n");
        }
        transcript.append("Older messages to fold in:\n");
        for (Message msg : olderBatch) {
            transcript.append(msg.getMessageType()).append(": ").append(msg.getText()).append("\n");
        }

        String updatedSummary = chatClient.prompt()
                .system(SUMMARIZER_PROMPT)
                .user(transcript.toString())
                .call()
                .content();

        if (updatedSummary != null && !updatedSummary.isBlank()) {
            state.rollingSummary = updatedSummary.trim();
        }
    }

    private List<Message> buildPromptMessages(SessionState state) {
        List<Message> out = new ArrayList<>();
        if (!state.rollingSummary.isBlank()) {
            out.add(new SystemMessage("Earlier conversation summary:\n" + state.rollingSummary));
        }
        out.addAll(state.recentMessages);
        return out;
    }

}