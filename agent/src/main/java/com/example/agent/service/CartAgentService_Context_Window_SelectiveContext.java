package com.example.agent.service;
import com.example.agent.tools.CartAgentTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class CartAgentService_Context_Window_SelectiveContext {

    private static final int MAX_TOOL_OUTPUT_CHARS = 500;
    private static final int MAX_REPLY_TOKENS = 300;
    private static final int MAX_RELEVANT_PAST_MESSAGES = 4;

    private static final Set<String> STOP_WORDS = Set.of(
            "the", "a", "an", "is", "are", "was", "were", "to", "for", "of",
            "in", "on", "and", "or", "it", "that", "this", "my", "me", "i"
    );

    private static final String SYSTEM_INSTRUCTIONS = """
            Role: Shopping cart assistant.
            Rules:
            - Use cart tools for any cart read/write or total calculation; never guess state.
            - Use lookupProductInfo for product prices or details.
            - Resolve pronouns/references ("it", "that", "one more") from prior turns.
            - Keep answers brief and direct.
            """;

    private final ChatClient chatClient;
    private final Map<String, List<Message>> sessions = new ConcurrentHashMap<>();

    public CartAgentService_Context_Window_SelectiveContext(
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
        List<Message> history = sessions.computeIfAbsent(key, k -> new ArrayList<>());

        synchronized (history) {
            history.add(new UserMessage(userMessage));

            List<Message> selectedContext = selectRelevantContext(history, userMessage);
            String reply = chatClient.prompt()
                    .messages(selectedContext)
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

    private List<Message> selectRelevantContext(List<Message> history, String currentQuery) {
        if (history.size() <= 2) {
            return List.copyOf(history);
        }

        Set<String> queryTerms = extractKeywords(currentQuery);
        List<Message> earlierMessages = history.subList(0, history.size() - 1);
        Message latestMessage = history.get(history.size() - 1);

        List<Message> matched = new ArrayList<>();
        for (int i = earlierMessages.size() - 1; i >= 0; i--) {
            Message candidate = earlierMessages.get(i);
            if (hasKeywordOverlap(candidate.getText(), queryTerms)) {
                matched.add(0, candidate);
                if (matched.size() >= MAX_RELEVANT_PAST_MESSAGES) {
                    break;
                }
            }
        }

        List<Message> promptMessages = new ArrayList<>(matched);
        promptMessages.add(latestMessage);
        return promptMessages;
    }

    private Set<String> extractKeywords(String text) {
        Set<String> keywords = new HashSet<>();
        if (text == null || text.isBlank()) {
            return keywords;
        }
        String[] tokens = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+");
        for (String token : tokens) {
            if (token.length() > 2 && !STOP_WORDS.contains(token)) {
                keywords.add(token);
            }
        }
        return keywords;
    }

    private boolean hasKeywordOverlap(String candidateText, Set<String> queryTerms) {
        if (candidateText == null || candidateText.isBlank() || queryTerms.isEmpty()) {
            return false;
        }
        return Arrays.stream(candidateText.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+"))
                .anyMatch(queryTerms::contains);
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
