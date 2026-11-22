package com.example.agent.service;

import com.example.agent.tools.CartAgentTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Service
public class CartAgentService_Context_Window_SemanticRetrieval {

    private static final int MAX_TOOL_OUTPUT_CHARS = 500;
    private static final int MAX_REPLY_TOKENS = 300;
    private static final int TOP_K_SEMANTIC_TURNS = 3;
    private static final double SIMILARITY_THRESHOLD = 0.55;
    private static final int KEEP_RECENT_MESSAGES = 2;

    private static final String SYSTEM_INSTRUCTIONS = """
            Role: Shopping cart assistant.
            Rules:
            - Use cart tools for any cart read/write or total calculation; never guess state.
            - Use lookupProductInfo for product prices or details.
            - Use retrieved past conversation context and recent turns to resolve references and preferences.
            - Keep answers brief and direct.
            """;

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final Map<String, List<Message>> recentSessions = new ConcurrentHashMap<>();
    private final Map<String, List<String>> sessionDocumentIds = new ConcurrentHashMap<>();

    public CartAgentService_Context_Window_SemanticRetrieval(
            ChatClient.Builder chatClientBuilder,
            CartAgentTools cartAgentTools,
            EmbeddingModel embeddingModel) {
        this.chatClient = chatClientBuilder
                .defaultSystem(SYSTEM_INSTRUCTIONS)
                .defaultTools(cartAgentTools)
                .defaultOptions(ChatOptions.builder().maxTokens(MAX_REPLY_TOKENS))
                .build();

        // Zero-setup in-memory vector store backed by your configured EmbeddingModel
        this.vectorStore = SimpleVectorStore.builder(embeddingModel).build();
    }

    public String chat(String sessionId, String userMessage) {
        String key = (sessionId == null || sessionId.isBlank()) ? "default-session" : sessionId.trim();
        List<Message> recentHistory = recentSessions.computeIfAbsent(key, k -> new ArrayList<>());

        synchronized (recentHistory) {
            // 1. Retrieve semantically similar earlier turns for this sessionId
            List<Document> similarTurns = retrieveSemanticHistory(key, userMessage);

            // 2. Build prompt: semantic context + recent turns + current user message
            List<Message> promptMessages = buildPromptMessages(similarTurns, recentHistory, userMessage);

            String reply = chatClient.prompt()
                    .messages(promptMessages)
                    .call()
                    .content();

            if (reply != null) {
                String prunedReply = pruneToolOutput(reply);

                // 3. Update short recent window (keeps last 2 messages for pronoun follow-ups)
                recentHistory.add(new UserMessage(userMessage));
                recentHistory.add(new AssistantMessage(prunedReply));
                while (recentHistory.size() > KEEP_RECENT_MESSAGES) {
                    recentHistory.remove(0);
                }

                // 4. Embed & store this completed turn in SimpleVectorStore under sessionId
                storeTurnInVectorStore(key, userMessage, prunedReply);
            }
            return reply;
        }
    }

    public void resetSession(String sessionId) {
        if (sessionId == null || sessionId.isBlank()) {
            return;
        }
        String key = sessionId.trim();
        recentSessions.remove(key);
        List<String> docIds = sessionDocumentIds.remove(key);
        if (docIds != null && !docIds.isEmpty()) {
            vectorStore.delete(docIds);
        }
    }

    private List<Document> retrieveSemanticHistory(String sessionId, String currentQuery) {
        if (!sessionDocumentIds.containsKey(sessionId) || sessionDocumentIds.get(sessionId).isEmpty()) {
            return List.of();
        }
        SearchRequest request = SearchRequest.builder()
                .query(currentQuery)
                .topK(TOP_K_SEMANTIC_TURNS)
                .similarityThreshold(SIMILARITY_THRESHOLD)
                .filterExpression("sessionId == '" + sessionId.replace("'", "") + "'")
                .build();
        return vectorStore.similaritySearch(request);
    }

    private List<Message> buildPromptMessages(
            List<Document> similarTurns,
            List<Message> recentHistory,
            String currentUserMessage) {
        List<Message> messages = new ArrayList<>();

        if (!similarTurns.isEmpty()) {
            String retrievedContext = similarTurns.stream()
                    .map(Document::getText)
                    .collect(Collectors.joining("\n---\n"));
            messages.add(new SystemMessage(
                    "Relevant Past Conversation Turns (Retrieved by Semantic Similarity):\n" + retrievedContext
            ));
        }

        messages.addAll(recentHistory);
        messages.add(new UserMessage(currentUserMessage));
        return messages;
    }

    private void storeTurnInVectorStore(String sessionId, String userMessage, String assistantReply) {
        String turnText = "User: " + pruneToolOutput(userMessage) + "\nAssistant: " + assistantReply;
        String docId = UUID.randomUUID().toString();

        Document turnDoc = new Document(
                docId,
                turnText,
                Map.of("sessionId", sessionId)
        );
        vectorStore.add(List.of(turnDoc));
        sessionDocumentIds.computeIfAbsent(sessionId, k -> new ArrayList<>()).add(docId);
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