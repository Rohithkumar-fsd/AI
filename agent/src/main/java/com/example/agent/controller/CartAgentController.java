package com.example.agent.controller;

import com.example.agent.service.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/agent")
public class CartAgentController {

    private final CartAgentService_Context_Window_Summarization summarization;
    private final CartAgentService_Context_Window_ToolOutputPruning toolOutputPruning;
    private final CartAgentService_Context_Window_SelectiveContext selectiveContext;
    private final CartAgentService_Context_Window_Sliding_Window slidingWindow;
    private final CartAgentService_Context_Window_StructuredWorkingMemory structuredWorkingMemory;
    private final CartAgentService_Context_Window_SemanticRetrieval semanticRetrieval;

    public CartAgentController(CartAgentService_Context_Window_Summarization summarization, CartAgentService_Context_Window_ToolOutputPruning toolOutputPruning, CartAgentService_Context_Window_SelectiveContext selectiveContext, CartAgentService_Context_Window_Sliding_Window slidingWindow, CartAgentService_Context_Window_StructuredWorkingMemory structuredWorkingMemory, CartAgentService_Context_Window_SemanticRetrieval semanticRetrieval) {
        this.summarization = summarization;
        this.toolOutputPruning = toolOutputPruning;
        this.selectiveContext = selectiveContext;
        this.slidingWindow = slidingWindow;
        this.structuredWorkingMemory = structuredWorkingMemory;
        this.semanticRetrieval = semanticRetrieval;
    }

    public record ChatRequest(String sessionId, String message) {}

    @PostMapping("tooloutputpruning/chat")
    public Map<String, String> toolOutputPruning(@RequestBody ChatRequest request) {
        String sessionId = (request.sessionId() == null || request.sessionId().isBlank())
                ? "default-session"
                : request.sessionId();
        String response = toolOutputPruning.chat(sessionId, request.message());
        return Map.of(
                "sessionId", sessionId,
                "response", response != null ? response : ""
        );
    }

    @PostMapping("slidingwindow/chat")
    public Map<String, String> slidingWindowOptimization(@RequestBody ChatRequest request) {
        String sessionId = (request.sessionId() == null || request.sessionId().isBlank())
                ? "default-session"
                : request.sessionId();
        String response = slidingWindow.chat(sessionId, request.message());
        return Map.of(
                "sessionId", sessionId,
                "response", response != null ? response : ""
        );
    }

    @PostMapping("selectivecontext/chat")
    public Map<String, String> selectiveContext(@RequestBody ChatRequest request) {
        String sessionId = (request.sessionId() == null || request.sessionId().isBlank())
                ? "default-session"
                : request.sessionId();
        String response = selectiveContext.chat(sessionId, request.message());
        return Map.of(
                "sessionId", sessionId,
                "response", response != null ? response : ""
        );
    }

    @PostMapping("summarization/chat")
    public Map<String, String> contextSummarization(@RequestBody ChatRequest request) {
        String sessionId = (request.sessionId() == null || request.sessionId().isBlank())
                ? "default-session"
                : request.sessionId();
        String response = summarization.chat(sessionId, request.message());
        return Map.of(
                "sessionId", sessionId,
                "response", response != null ? response : ""
        );
    }

    @PostMapping("contextstructured/chat")
    public Map<String, String> structuredWorkingMemory(@RequestBody ChatRequest request) {
        String sessionId = (request.sessionId() == null || request.sessionId().isBlank())
                ? "default-session"
                : request.sessionId();
        String response = structuredWorkingMemory.chat(sessionId, request.message());
        return Map.of(
                "sessionId", sessionId,
                "response", response != null ? response : ""
        );
    }

    @DeleteMapping("/sessions/{sessionId}")
    public ResponseEntity<Map<String, String>> clearSession(@PathVariable String sessionId) {
        summarization.resetSession(sessionId);
        return ResponseEntity.ok(Map.of("status", "cleared", "sessionId", sessionId));
    }

    @PostMapping("semanticretrieval/chat")
    public Map<String, String> semanticRetrieval(@RequestBody ChatRequest request) {
        String sessionId = (request.sessionId() == null || request.sessionId().isBlank())
                ? "default-session"
                : request.sessionId();
        String response = semanticRetrieval.chat(sessionId, request.message());
        return Map.of(
                "sessionId", sessionId,
                "response", response != null ? response : ""
        );
    }
}