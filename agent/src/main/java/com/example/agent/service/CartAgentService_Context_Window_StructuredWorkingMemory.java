package com.example.agent.service;
import com.example.agent.tools.CartAgentTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class CartAgentService_Context_Window_StructuredWorkingMemory {

    private static final int MAX_TOOL_OUTPUT_CHARS = 500;
    private static final int MAX_REPLY_TOKENS = 300;
    private static final int KEEP_RECENT_MESSAGES = 4;

    private static final String SYSTEM_INSTRUCTIONS = """
            Role: Shopping cart assistant.
            Rules:
            - Use cart tools for any cart read/write or total calculation; never guess state.
            - Use lookupProductInfo for product prices or details.
            - Resolve pronouns/references ("it", "that", "one more") from prior turns and working state.
            - Keep answers brief and direct.
            """;

    private final ChatClient chatClient;
    private final Map<String, SessionMemory> sessions = new ConcurrentHashMap<>();

    public CartAgentService_Context_Window_StructuredWorkingMemory(
            ChatClient.Builder chatClientBuilder,
            CartAgentTools cartAgentTools) {
        ChatOptions.Builder<?> optionsBuilder = ChatOptions.builder()
                .maxTokens(MAX_REPLY_TOKENS);

        this.chatClient = chatClientBuilder
                .defaultSystem(SYSTEM_INSTRUCTIONS)
                .defaultTools(cartAgentTools)
                .defaultOptions(optionsBuilder)
                .build();
    }

    public String chat(String sessionId, String userMessage) {
        String key = (sessionId == null || sessionId.isBlank()) ? "default-session" : sessionId.trim();
        SessionMemory memory = sessions.computeIfAbsent(key, k -> new SessionMemory());

        synchronized (memory) {
            memory.recentMessages.add(new UserMessage(userMessage));

            List<Message> promptMessages = buildPromptMessages(memory);
            String reply = chatClient.prompt()
                    .messages(promptMessages)
                    .call()
                    .content();

            if (reply != null) {
                String prunedReply = pruneToolOutput(reply);
                memory.recentMessages.add(new AssistantMessage(prunedReply));
                refreshWorkingMemory(memory, userMessage, prunedReply);
                trimRecentMessages(memory.recentMessages);
            }
            return reply;
        }
    }

    public void resetSession(String sessionId) {
        if (sessionId != null && !sessionId.isBlank()) {
            sessions.remove(sessionId.trim());
        }
    }

    private List<Message> buildPromptMessages(SessionMemory memory) {
        List<Message> messages = new ArrayList<>();
        if (!memory.workingState.isBlank()) {
            messages.add(new SystemMessage("Current Session Working State:\n" + memory.workingState));
        }
        messages.addAll(memory.recentMessages);
        return messages;
    }

    private void refreshWorkingMemory(SessionMemory memory, String userMessage, String assistantReply) {
        // Lightweight deterministic state note; in production you can also sync this from cart state
        String compactTurn = "User: " + pruneToolOutput(userMessage) + " | Assistant: " + assistantReply;
        if (memory.workingState.isBlank()) {
            memory.workingState = compactTurn;
        } else {
            String combined = memory.workingState + "\n" + compactTurn;
            memory.workingState = pruneToolOutput(combined);
        }
    }

    private void trimRecentMessages(List<Message> recentMessages) {
        while (recentMessages.size() > KEEP_RECENT_MESSAGES) {
            recentMessages.remove(0);
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

    private static class SessionMemory {
        String workingState = "";
        final List<Message> recentMessages = new ArrayList<>();
    }
}