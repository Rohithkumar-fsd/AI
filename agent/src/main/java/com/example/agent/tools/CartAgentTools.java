package com.example.agent.tools;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class CartAgentTools {

    public record CartEntry(String itemName, int quantity, double unitPrice) {
        public double lineTotal() {
            return quantity * unitPrice;
        }
    }

    // Replace this in-memory map with your Spring Data JPA repository backed by MySQL if desired
    private final Map<String, CartEntry> cart = new ConcurrentHashMap<>();

    private static final Map<String, Double> CATALOG_PRICES = Map.of(
            "iphone 16", 79900.0,
            "airpods pro", 24900.0,
            "macbook air", 114900.0,
            "usb-c hub", 2499.0
    );

    @Tool(description = "Add an item and quantity to the user's shopping cart and return the updated item entry.")
    public CartEntry addItemToCart(
            @ToolParam(description = "Product name, e.g., iPhone 16") String itemName,
            @ToolParam(description = "Quantity to add (at least 1)") int quantity,
            @ToolParam(description = "Unit price in INR if known; pass 0 to look up catalog or default price") double unitPrice) {

        int safeQty = Math.max(1, quantity);
        String key = itemName.trim().toLowerCase();
        double resolvedPrice = unitPrice > 0
                ? unitPrice
                : CATALOG_PRICES.getOrDefault(key, 999.0);

        return cart.compute(key, (k, existing) -> {
            if (existing == null) {
                return new CartEntry(itemName.trim(), safeQty, resolvedPrice);
            }
            return new CartEntry(existing.itemName(), existing.quantity() + safeQty, resolvedPrice);
        });
    }

    @Tool(description = "Remove an item completely from the user's shopping cart by item name.")
    public String removeItemFromCart(
            @ToolParam(description = "Product name to remove") String itemName) {
        CartEntry removed = cart.remove(itemName.trim().toLowerCase());
        return removed != null
                ? "Removed " + removed.itemName() + " from cart."
                : "Item '" + itemName + "' was not found in the cart.";
    }

    @Tool(description = "List all items currently in the shopping cart.")
    public Map<String, CartEntry> getCartItems() {
        return Map.copyOf(cart);
    }

    @Tool(description = "Calculate the total value of all items currently in the shopping cart.")
    public double calculateCartTotal() {
        return cart.values().stream()
                .mapToDouble(CartEntry::lineTotal)
                .sum();
    }

    @Tool(description = "Look up product pricing or quick product info when a price or detail is needed.")
    public String lookupProductInfo(
            @ToolParam(description = "Product name or search query") String query) {
        String normalized = query.trim().toLowerCase();
        for (Map.Entry<String, Double> entry : CATALOG_PRICES.entrySet()) {
            if (normalized.contains(entry.getKey())) {
                return entry.getKey() + " current catalog price: INR " + entry.getValue();
            }
        }
        return "No fixed catalog entry for '" + query + "'. If adding to cart, confirm or pass a unit price.";
    }
}