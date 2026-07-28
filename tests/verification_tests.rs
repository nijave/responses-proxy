/// Verification tests: real code with realistic payloads.
/// Run with: cargo test verification
use responses_proxy::convert::{chat_to_responses, responses_to_chat};
use responses_proxy::types::streaming::{
    StreamEvent, StreamState, build_completion_events, process_chunk_value,
};
use responses_proxy::types::{chat, responses};
use serde_json::json;

fn test_state() -> responses_proxy::app::State {
    use responses_proxy::config::ResolvedConfig;
    let config = ResolvedConfig {
        listen: String::new(),
        timeout: 30,
        auth_keys: std::collections::HashSet::new(),
        cors_allow_origins: vec![],
        allowed_tool_types: vec!["function".into()],
        log_level: "info".into(),
        models: std::collections::HashMap::new(),
        model_names: vec![],
        compact_encryption_key: String::new(),
        max_body_bytes: 100 * 1024 * 1024,
    };
    responses_proxy::app::State::new(config)
}

fn test_state_with_rewrite(
    rewrite: responses_proxy::config::RewriteConfig,
) -> responses_proxy::app::State {
    use responses_proxy::config::{ResolvedConfig, ResolvedProvider, RewriteProfile};
    use std::time::Duration;

    let mut models = std::collections::HashMap::new();
    models.insert(
        "gpt-5.5".into(),
        ResolvedProvider {
            base_url: "http://example.test".into(),
            api_key: "test-key".into(),
            model: "upstream-model".into(),
            timeout: Duration::from_secs(30),
            rewrite: RewriteProfile {
                chat_out: rewrite,
                ..Default::default()
            },
            max_input_chars: None,
            max_input_messages: 1000,
            stream_structured_output: true,
        },
    );

    let config = ResolvedConfig {
        listen: String::new(),
        timeout: 30,
        auth_keys: std::collections::HashSet::new(),
        cors_allow_origins: vec![],
        allowed_tool_types: vec!["function".into()],
        log_level: "info".into(),
        models,
        model_names: vec!["gpt-5.5".into()],
        compact_encryption_key: String::new(),
        max_body_bytes: 100 * 1024 * 1024,
    };
    responses_proxy::app::State::new(config)
}

fn deepseek_chat_out_rewrite() -> responses_proxy::config::RewriteConfig {
    let config = responses_proxy::config::load_config("deepseek.config.yaml").unwrap();
    config.models["gpt-5.5"].rewrite.chat_out.clone()
}

// ── Scenario 1: Simple text, no streaming ────────────────────────────

#[tokio::test]
async fn s1_simple_text_request() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "What is 2+2? Reply with just the number."
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["model"], "gpt-5.5");
    assert_eq!(j["messages"].as_array().unwrap().len(), 1);
    assert_eq!(j["messages"][0]["role"], "user");
    assert_eq!(
        j["messages"][0]["content"],
        "What is 2+2? Reply with just the number."
    );
    assert!(j.get("thinking").is_none());
    assert!(j.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn s1_simple_text_response() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1715550000u64,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "4"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 1,
            "total_tokens": 13,
            "completion_tokens_details": {"reasoning_tokens": 0}
        }
    }))
    .unwrap();

    let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    let j = serde_json::to_value(&resp).unwrap();

    assert_eq!(j["object"], "response");
    assert_eq!(j["status"], "completed");
    assert_eq!(j["model"], "gpt-5.5");
    assert!(j["id"].as_str().unwrap().starts_with("resp_"));

    let output = j["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["role"], "assistant");
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(output[0]["content"][0]["text"], "4");

    assert_eq!(j["usage"]["input_tokens"], 12);
    assert_eq!(j["usage"]["output_tokens"], 1);
    assert_eq!(j["usage"]["total_tokens"], 13);
    assert_eq!(j["usage"]["input_tokens_details"]["cached_tokens"], 0);
    assert_eq!(j["usage"]["output_tokens_details"]["reasoning_tokens"], 0);
}

// ── Scenario 2: Instructions + Reasoning xhigh ───────────────────────

#[tokio::test]
async fn s2_instructions_and_reasoning_request() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Solve the complex equation.",
        "instructions": "You are a math tutor. Always show your work.",
        "reasoning": {"effort": "xhigh"}
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();
    let msgs = j["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(
        msgs[0]["content"],
        "You are a math tutor. Always show your work."
    );
    assert_eq!(msgs[1]["role"], "user");

    assert_eq!(j["reasoning_effort"], "xhigh");
}

#[tokio::test]
async fn s2_reasoning_content_response() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "chatcmpl-def456",
        "object": "chat.completion",
        "created": 1715550000u64,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "x = 5",
                "reasoning_content": "First, we isolate x by..."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 50,
            "total_tokens": 90,
            "completion_tokens_details": {"reasoning_tokens": 30}
        }
    }))
    .unwrap();

    let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    let j = serde_json::to_value(&resp).unwrap();

    let output = j["output"].as_array().unwrap();
    assert_eq!(output.len(), 2);
    // reasoning item first
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
    assert_eq!(output[0]["content"][0]["text"], "First, we isolate x by...");
    // message second
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "x = 5");
    // usage
    assert_eq!(j["usage"]["output_tokens_details"]["reasoning_tokens"], 30);
}

// ── Scenario 3: All reasoning effort levels ──────────────────────────

#[tokio::test]
async fn s3_reasoning_effort_all_levels() {
    let cases = &[
        ("none", true, Some("none")),
        ("minimal", true, Some("minimal")),
        ("low", true, Some("low")),
        ("medium", true, Some("medium")),
        ("high", true, Some("high")),
        ("xhigh", true, Some("xhigh")),
        // gpt-5.6-class tiers above xhigh — passed through verbatim.
        ("max", true, Some("max")),
        ("ultra", true, Some("ultra")),
    ];

    for (effort, expect_think, expect_re) in cases {
        let req: responses::Request = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": "Hi",
            "reasoning": {"effort": effort}
        }))
        .unwrap();

        let chat = responses_to_chat(req, &test_state()).await.unwrap();
        let j = serde_json::to_value(&chat).unwrap();

        if *expect_think {
            assert_eq!(j["reasoning_effort"], expect_re.unwrap(), "effort={effort}");
        } else {
            assert!(
                j.get("reasoning_effort").is_none(),
                "effort={effort} should have no reasoning_effort"
            );
        }
    }
}

// ── Codex gpt-5.6 additional_tools (code-mode) → Chat function tools ─
#[tokio::test]
async fn additional_tools_flattened_to_chat_functions() {
    // gpt-5.6 delivers its tools inside an `additional_tools` input item using
    // the code-mode custom/namespace protocol. Chat Completions can't take
    // those, so every entry must be flattened into a plain `function` tool.
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {"type": "custom", "name": "exec", "description": "Run JS"},
                    {"type": "namespace", "name": "shell", "tools": [
                        {"type": "function", "name": "exec_command", "description": "run",
                         "parameters": {"type": "object",
                                        "properties": {"cmd": {"type": "string"}},
                                        "required": ["cmd"]}},
                        {"type": "custom", "name": "apply_patch", "description": "patch"}
                    ]}
                ]
            }
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();
    let tools = j["tools"].as_array().expect("tools present");

    // namespace expands in place; names preserved verbatim for Codex routing.
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["exec", "exec_command", "apply_patch"]);

    // Every hoisted tool is function-typed (nothing left as custom/namespace).
    for t in tools {
        assert_eq!(t["type"], "function");
    }
    // A real function keeps its own schema.
    assert_eq!(
        j["tools"][1]["function"]["parameters"]["properties"]["cmd"]["type"],
        "string"
    );
    // Freeform custom tools collapse to a single string `input` argument.
    assert_eq!(
        j["tools"][0]["function"]["parameters"]["properties"]["input"]["type"],
        "string"
    );
    assert_eq!(
        j["tools"][2]["function"]["parameters"]["properties"]["input"]["type"],
        "string"
    );
}

#[tokio::test]
async fn mcp_namespace_tools_flattened_to_chat_functions() {
    // gpt-5.6 wraps MCP server tools in a proprietary `namespace` entry (member
    // names like `mcp__server__tool`). Chat Completions can't unwrap that, so
    // the members must be flattened to plain function tools with names intact.
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {"type": "namespace", "name": "mcp", "tools": [
                        {"type": "function", "name": "mcp__github__list_issues",
                         "description": "List issues",
                         "parameters": {"type": "object",
                                        "properties": {"repo": {"type": "string"}},
                                        "required": ["repo"]}}
                    ]},
                    // A remote MCP server reference carries no per-tool schema
                    // and must be dropped, not crash the request.
                    {"type": "mcp", "server_label": "github",
                     "server_url": "https://example.invalid/mcp"}
                ]
            }
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();
    let tools = j["tools"].as_array().expect("tools present");

    assert_eq!(
        tools.len(),
        1,
        "namespace member kept, mcp server ref dropped"
    );
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "mcp__github__list_issues");
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["repo"]["type"],
        "string"
    );
}

#[tokio::test]
async fn multi_agent_collaboration_tools_dropped() {
    // gpt-5.6 multi-agent mode delivers the hosted collaboration actions
    // (spawn_agent, …) inside a `collaboration` namespace. They execute in
    // OpenAI's hosted Responses runtime, not the client, so a Chat Completions
    // upstream cannot fulfil them — advertising them lures the model into
    // `spawn_agent` calls the client rejects as `unsupported call`. They must be
    // dropped while the client-executable tools (exec/wait/request_user_input)
    // are kept, so the model falls back to plain single-agent code-mode.
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {"type": "custom", "name": "exec", "description": "Run JS"},
                    {"type": "function", "name": "wait", "description": "wait",
                     "parameters": {"type": "object", "properties": {}}},
                    {"type": "function", "name": "request_user_input",
                     "description": "ask",
                     "parameters": {"type": "object", "properties": {}}},
                    {"type": "namespace", "name": "collaboration",
                     "description": "Tools for spawning and managing sub-agents.",
                     "tools": [
                        {"type": "function", "name": "spawn_agent",
                         "parameters": {"type": "object", "properties": {}}},
                        {"type": "function", "name": "followup_task",
                         "parameters": {"type": "object", "properties": {}}},
                        {"type": "function", "name": "interrupt_agent",
                         "parameters": {"type": "object", "properties": {}}},
                        {"type": "function", "name": "list_agents",
                         "parameters": {"type": "object", "properties": {}}},
                        {"type": "function", "name": "send_message",
                         "parameters": {"type": "object", "properties": {}}},
                        {"type": "function", "name": "wait_agent",
                         "parameters": {"type": "object", "properties": {}}}
                    ]}
                ]
            }
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();
    let tools = j["tools"].as_array().expect("tools present");
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["exec", "wait", "request_user_input"]);
    for hosted in [
        "spawn_agent",
        "followup_task",
        "interrupt_agent",
        "list_agents",
        "send_message",
        "wait_agent",
    ] {
        assert!(!names.contains(&hosted), "{hosted} must be dropped");
    }
}

#[tokio::test]
async fn multi_agent_hosted_action_dropped_outside_collaboration_namespace() {
    // Defensive: even if a hosted action arrives as a bare top-level tool or in
    // a differently-named namespace, it must still be dropped by name.
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {"type": "function", "name": "spawn_agent",
                     "parameters": {"type": "object", "properties": {}}},
                    {"type": "namespace", "name": "misc", "tools": [
                        {"type": "function", "name": "list_agents",
                         "parameters": {"type": "object", "properties": {}}},
                        {"type": "function", "name": "keep_me",
                         "parameters": {"type": "object", "properties": {}}}
                    ]}
                ]
            }
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();
    let names: Vec<&str> = j["tools"]
        .as_array()
        .expect("tools present")
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["keep_me"]);
}

#[tokio::test]
async fn code_mode_tools_restored_on_continuation_turn() {
    // Codex delivers `additional_tools` only on a new user turn; a tool-result
    // continuation references `previous_response_id` and omits them. The proxy
    // caches the derived registry under the response id and restores it on the
    // continuation so the model isn't left with an empty tool set (which made it
    // stall after a single call).
    let state = test_state();

    // Turn 1: additional_tools present → tools built and cached under a rid.
    let turn1: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "custom", "name": "exec", "description": "Run JS"},
                {"type": "function", "name": "wait",
                 "parameters": {"type": "object", "properties": {}}}
            ]}
        ]
    }))
    .unwrap();
    let turn1_input = turn1.input.clone();
    let chat1 = responses_to_chat(turn1, &state).await.unwrap();
    let tools1 = chat1.tools.clone().expect("turn 1 has tools");
    assert_eq!(tools1.len(), 2);
    // Cache exactly as the handler does: tools + the custom-name set.
    let custom_names1 = responses_proxy::convert::custom_tool_names(&turn1_input);
    assert!(custom_names1.contains("exec"));
    state
        .store()
        .put_tools("resp_turn1".to_string(), tools1, custom_names1)
        .await;

    // Turn 2: continuation — no additional_tools, references the prior response.
    let turn2: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "previous_response_id": "resp_turn1",
        "input": [
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ]
    }))
    .unwrap();
    let chat2 = responses_to_chat(turn2, &state).await.unwrap();
    let j = serde_json::to_value(&chat2).unwrap();
    let names: Vec<&str> = j["tools"]
        .as_array()
        .expect("continuation restored tools")
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["exec", "wait"]);

    // The custom-name set is restorable too, so the continuation's `exec`
    // response is re-emitted as a custom_tool_call rather than a function_call
    // (which Codex would cancel).
    let restored = state
        .store()
        .get_custom_names("resp_turn1")
        .await
        .expect("custom names cached");
    assert!(restored.contains("exec"));
}

#[tokio::test]
async fn resolve_custom_tool_names_falls_back_to_cached() {
    use responses_proxy::types::item::InputItem;
    let state = test_state();

    // Fresh turn with additional_tools resolves directly and is cached.
    let fresh: Vec<InputItem> = serde_json::from_value(json!([
        {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "custom", "name": "exec"}
        ]}
    ]))
    .unwrap();
    let names = responses_proxy::convert::resolve_custom_tool_names(&state, &fresh, None).await;
    assert!(names.contains("exec"));
    state
        .store()
        .put_tools("resp_a".to_string(), vec![], names)
        .await;

    // Continuation (no additional_tools) falls back to the cached set.
    let cont: Vec<InputItem> = serde_json::from_value(json!([
        {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    ]))
    .unwrap();
    let restored =
        responses_proxy::convert::resolve_custom_tool_names(&state, &cont, Some("resp_a")).await;
    assert!(restored.contains("exec"));

    // Unknown previous id → no invention.
    let none =
        responses_proxy::convert::resolve_custom_tool_names(&state, &cont, Some("resp_x")).await;
    assert!(none.is_empty());
}

#[tokio::test]
async fn continuation_without_cached_tools_stays_toolless() {
    // No regression: a continuation that never had code-mode tools (e.g. models
    // below 5.6) resolves to no tools rather than inventing any.
    let state = test_state();
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "previous_response_id": "resp_unknown",
        "input": [
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ]
    }))
    .unwrap();
    let chat = responses_to_chat(req, &state).await.unwrap();
    assert!(chat.tools.is_none());
}

// ── Codex gpt-5.6 code-mode: custom ↔ function round-trip ────────────
#[tokio::test]
async fn custom_tool_names_extracted_from_additional_tools() {
    use responses_proxy::types::item::InputItem;
    let input: Vec<InputItem> = serde_json::from_value(json!([
        {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "custom", "name": "exec"},
            {"type": "namespace", "name": "mcp", "tools": [
                {"type": "function", "name": "mcp__gh__list", "parameters": {"type": "object"}},
                {"type": "custom", "name": "collab"}
            ]}
        ]}
    ]))
    .unwrap();
    let names = responses_proxy::convert::custom_tool_names(&input);
    assert!(names.contains("exec"));
    assert!(names.contains("collab"));
    // namespace *function* members stay function → not in the custom set.
    assert!(!names.contains("mcp__gh__list"));
}

#[tokio::test]
async fn nonstreaming_function_call_remapped_to_custom_by_name() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "c", "object": "chat.completion", "created": 1u64, "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "exec", "arguments": "{\"input\":\"pwd\"}"}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "update_plan", "arguments": "{\"x\":1}"}}
            ]
        }}]
    }))
    .unwrap();
    let mut resp = chat_to_responses(chat, "gpt-5.6-sol".into(), None);
    let custom: std::collections::HashSet<String> = ["exec".to_string()].into_iter().collect();
    responses_proxy::convert::remap_custom_tool_calls(&mut resp, &custom);

    let j = serde_json::to_value(&resp).unwrap();
    let out = j["output"].as_array().unwrap();
    let exec = out.iter().find(|i| i["name"] == "exec").unwrap();
    assert_eq!(exec["type"], "custom_tool_call");
    assert_eq!(exec["input"], "pwd", "the {{input}} wrapper is unwrapped");
    let plan = out.iter().find(|i| i["name"] == "update_plan").unwrap();
    assert_eq!(
        plan["type"], "function_call",
        "non-custom names stay function"
    );
}

#[tokio::test]
async fn nonstreaming_remap_is_noop_without_custom_names() {
    // Regression guard for models below 5.6: empty set leaves function calls intact.
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "c", "object": "chat.completion", "created": 1u64, "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "exec", "arguments": "{}"}}]
        }}]
    }))
    .unwrap();
    let mut resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    responses_proxy::convert::remap_custom_tool_calls(&mut resp, &std::collections::HashSet::new());
    let j = serde_json::to_value(&resp).unwrap();
    assert_eq!(j["output"][0]["type"], "function_call");
}

#[tokio::test]
async fn streaming_custom_tool_emits_custom_tool_call_events() {
    let mut state = StreamState::new("resp".into(), "msg".into(), "gpt-5.6-sol".into());
    state.custom_tool_names.insert("exec".to_string());

    let mut events = process_chunk_value(
        &mut state,
        serde_json::from_str(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"exec","arguments":"{\"input\":\"pwd\"}"}}]}}]}"#).unwrap(),
    )
    .unwrap_or_default();
    events.extend(build_completion_events(&mut state));

    let types: Vec<String> = events
        .iter()
        .map(|e| {
            serde_json::to_value(e).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    // Custom tool call, not a function call, on the wire.
    assert!(
        types
            .iter()
            .any(|t| t == "response.custom_tool_call_input.done")
    );
    assert!(
        !types
            .iter()
            .any(|t| t == "response.function_call_arguments.delta"),
        "function arg deltas must be suppressed for custom tools"
    );

    let jsons: Vec<serde_json::Value> = events
        .iter()
        .map(|e| serde_json::to_value(e).unwrap())
        .collect();
    let added = jsons
        .iter()
        .find(|e| e["type"] == "response.output_item.added")
        .unwrap();
    assert_eq!(added["item"]["type"], "custom_tool_call");
    let done = jsons
        .iter()
        .find(|e| e["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(done["item"]["type"], "custom_tool_call");
    assert_eq!(done["item"]["input"], "pwd");
    // completed_items keeps the function shape for upstream replay.
    assert!(matches!(
        state.completed_items.first(),
        Some(responses_proxy::types::item::OutputItem::FunctionCall(_))
    ));
}

#[tokio::test]
async fn history_custom_tool_call_wrapped_as_input_json() {
    // A replayed code-mode call must use the `{ input }` function-arg shape.
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.6-sol",
        "input": [
            {"type": "additional_tools", "role": "developer",
             "tools": [{"type": "custom", "name": "exec"}]},
            {"type": "custom_tool_call", "call_id": "call_1", "name": "exec",
             "input": "tools.exec_command({cmd:[\"pwd\"]})"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "ok"}
        ]
    }))
    .unwrap();
    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();
    let msgs = j["messages"].as_array().unwrap();
    let assistant = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
    let args = assistant["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(args).unwrap();
    assert_eq!(parsed["input"], "tools.exec_command({cmd:[\"pwd\"]})");
}

#[tokio::test]
async fn s3_reasoning_summary_without_effort_does_not_force_effort() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "reasoning": {"summary": "auto"}
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert!(j.get("reasoning_effort").is_none());
}

// ── Scenario 4: Thinking mode with reasoning ────────────────────────

#[tokio::test]
async fn s4_thinking_disables_logprobs() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "reasoning": {"effort": "high"}
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    // Reasoning present -> no logprobs fields sent downstream
    assert!(j.get("logprobs").is_none());
    assert!(j.get("top_logprobs").is_none());
}

// ── Scenario 5: Full tool conversation roundtrip ─────────────────────

#[tokio::test]
async fn s5_tool_conversation_request() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in NYC?"}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"New York\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 72F"},
            {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "NYC is sunny, 72F."}]}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    let msgs = j["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "Weather in NYC?");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(msgs[2]["role"], "tool");
    assert_eq!(msgs[2]["tool_call_id"], "call_1");
    assert_eq!(msgs[2]["content"], "Sunny, 72F");
    assert_eq!(msgs[3]["role"], "assistant");
    assert_eq!(msgs[3]["content"], "NYC is sunny, 72F.");
}

#[tokio::test]
async fn s5_tool_call_response() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "chatcmpl-tools",
        "object": "chat.completion",
        "created": 1715550000u64,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"q\":\"test\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }))
    .unwrap();

    let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    let j = serde_json::to_value(&resp).unwrap();

    let output = j["output"].as_array().unwrap();
    // Only function_call, no message (content=null + tool_calls present)
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["call_id"], "call_xyz");
    assert_eq!(output[0]["name"], "search");
}

// ── Scenario 6: Content filter ───────────────────────────────────────

#[tokio::test]
async fn s6_content_filter_response() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "chatcmpl-cf",
        "object": "chat.completion",
        "created": 1715550000u64,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": null},
            "finish_reason": "content_filter"
        }]
    }))
    .unwrap();

    let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    let j = serde_json::to_value(&resp).unwrap();

    // content_filter with incomplete_details → overall status is "incomplete" per doc
    assert_eq!(j["status"], "incomplete");
    assert_eq!(j["output"][0]["status"], "incomplete");
    assert_eq!(j["output"][0]["content"][0]["type"], "refusal");
    assert_eq!(j["output"][0]["content"][0]["refusal"], "content_filter");
    assert_eq!(j["incomplete_details"]["reason"], "content_filter");
}

// ── Scenario 7: Error response ───────────────────────────────────────

#[tokio::test]
async fn s7_error_response() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "",
        "object": "error",
        "created": 0,
        "model": "",
        "choices": [],
        "error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}
    }))
    .unwrap();

    let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    let j = serde_json::to_value(&resp).unwrap();

    assert_eq!(j["status"], "failed");
    assert!(j["output"].as_array().unwrap().is_empty());
    assert_eq!(j["error"]["code"], "invalid_api_key");
    assert_eq!(j["error"]["message"], "Invalid API key");
}

// ── Scenario 8: All finish reasons ───────────────────────────────────

#[tokio::test]
async fn s8_finish_reason_all() {
    let cases = &[
        ("stop", "completed", &None),
        ("tool_calls", "completed", &None),
        ("length", "completed", &Some("max_output_tokens")),
        ("content_filter", "incomplete", &Some("content_filter")),
        ("insufficient_system_resource", "completed", &None),
    ];

    for (reason, exp_status, exp_details) in cases {
        let chat: chat::Completion = serde_json::from_value(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1715550000u64,
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": reason
            }]
        }))
        .unwrap();

        let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
        let j = serde_json::to_value(&resp).unwrap();

        assert_eq!(
            j["output"][0]["status"].as_str().unwrap(),
            *exp_status,
            "finish_reason={reason}"
        );
        match exp_details {
            Some(expected) => {
                assert_eq!(
                    j["incomplete_details"]["reason"].as_str().unwrap(),
                    *expected,
                    "finish_reason={reason}"
                );
            }
            None => {
                assert!(
                    j.get("incomplete_details").is_none_or(|v| v.is_null()),
                    "finish_reason={reason} expected no incomplete_details, got {:?}",
                    j.get("incomplete_details")
                );
            }
        }
    }
}

// ── Scenario 9: Tool normalization (flat→nested, allowlist filter) ──

#[tokio::test]
async fn s9_tool_normalization() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "tools": [
            {"type": "function", "name": "get_weather", "description": "Weather", "parameters": {"type": "object"}, "strict": true},
            {"type": "web_search_preview"}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    let tools = j["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
}

// ── Scenario 10: Response format from text.format ────────────────────

#[tokio::test]
async fn s10_text_format_to_response_format() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Output JSON",
        "text": {"format": {"type": "json_object"}}
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["response_format"]["type"], "json_object");
}

#[tokio::test]
async fn s10_json_schema_is_preserved() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Output JSON",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "description": "An answer object",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                },
                "strict": true
            }
        }
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["response_format"]["type"], "json_schema");
    assert_eq!(j["response_format"]["json_schema"]["name"], "answer");
    assert_eq!(
        j["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"],
        "string"
    );
    assert_eq!(j["response_format"]["json_schema"]["strict"], true);
}

#[tokio::test]
async fn s10_max_output_and_parallel_false_are_preserved() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "max_output_tokens": 123,
        "parallel_tool_calls": false
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["max_completion_tokens"], 123);
    assert!(j.get("max_tokens").is_none());
    assert_eq!(j["parallel_tool_calls"], false);
}

#[tokio::test]
async fn s10_max_output_tokens_can_map_to_deprecated_max_tokens_by_config() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "max_output_tokens": 123
    }))
    .unwrap();
    let state = test_state_with_rewrite(responses_proxy::config::RewriteConfig {
        steps: vec![responses_proxy::config::RewriteStep::Rename(vec![(
            "max_completion_tokens".into(),
            "max_tokens".into(),
        )])],
    });

    let chat = responses_to_chat(req, &state).await.unwrap();
    let provider = state.config().models.get("gpt-5.5").unwrap();
    let mut j = serde_json::to_value(&chat).unwrap();
    responses_proxy::rewrite::apply_rewrite(&mut j, &provider.rewrite.chat_out).unwrap();

    assert_eq!(j["max_tokens"], 123);
    assert!(j.get("max_completion_tokens").is_none());
}

#[tokio::test]
async fn deepseek_rewrite_preview() {
    let rewrite = deepseek_chat_out_rewrite();

    for effort in ["none", "minimal", "low", "medium", "high", "xhigh"] {
        let req: responses::Request = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": "Output JSON",
            "max_output_tokens": 123,
            "reasoning": {"effort": effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }
        }))
        .unwrap();

        let chat = responses_to_chat(req, &test_state()).await.unwrap();
        let mut body = serde_json::to_value(&chat).unwrap();
        responses_proxy::rewrite::apply_rewrite(&mut body, &rewrite).unwrap();

        println!(
            "{effort}:\n{}",
            serde_json::to_string_pretty(&body).unwrap()
        );
    }
}

#[tokio::test]
async fn deepseek_rewrite_reasoning_request_before_after() {
    let rewrite = deepseek_chat_out_rewrite();

    let request_body = json!({
        "model": "gpt-5.5",
        "input": "Output JSON",
        "max_output_tokens": 123,
        "reasoning": {"effort": "high"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                },
                "strict": true
            }
        }
    });

    let req: responses::Request = serde_json::from_value(request_body.clone()).unwrap();
    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let before_rewrite = serde_json::to_value(&chat).unwrap();
    let mut after_rewrite = before_rewrite.clone();
    responses_proxy::rewrite::apply_rewrite(&mut after_rewrite, &rewrite).unwrap();

    println!(
        "responses request:\n{}",
        serde_json::to_string_pretty(&request_body).unwrap()
    );
    println!(
        "chat before rewrite:\n{}",
        serde_json::to_string_pretty(&before_rewrite).unwrap()
    );
    println!(
        "chat after rewrite:\n{}",
        serde_json::to_string_pretty(&after_rewrite).unwrap()
    );
}

#[tokio::test]
async fn s10_prompt_is_rejected_until_supported() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "prompt": {"id": "pmpt_123"}
    }))
    .unwrap();

    let err = responses_to_chat(req, &test_state()).await.unwrap_err();
    assert!(err.contains(&"prompt".to_string()));
}

#[tokio::test]
async fn s10_no_text_no_response_format() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi"
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert!(j.get("response_format").is_none());
}

// ── Scenario 11: Passthrough fields (temperature, top_p, etc.) ──────

#[tokio::test]
async fn s11_passthrough_fields() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 2048,
        "stop": ["END"],
        "tool_choice": "required"
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["temperature"], 0.7);
    assert_eq!(j["top_p"], 0.9);
    assert_eq!(j["max_completion_tokens"], 2048);
    assert!(j.get("max_tokens").is_none());
    assert_eq!(j["stop"][0], "END");
    assert_eq!(j["tool_choice"], "required");
}

// ── Scenario 12: Streaming usage chunk ───────────────────────────────

#[tokio::test]
async fn s12_streaming_usage_captured() {
    let mut state = StreamState::new(
        "resp_test".into(),
        "msg_test".into(),
        "deepseek-v4-pro".into(),
    );
    state.accumulated_text = "Answer".into();

    // Usage-only chunk (needs all required ChatCompletionChunk fields)
    let events = process_chunk_value(
        &mut state,
        serde_json::from_str(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1715550000,"model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":3,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}}}"#).unwrap(),
    );
    assert!(events.is_none());
    assert!(state.usage.is_some());
    assert_eq!(state.usage.as_ref().unwrap().prompt_tokens, 10);

    // [DONE] includes usage
    let events = build_completion_events(&mut state);
    let completed = events
        .iter()
        .find(|e| matches!(e, StreamEvent::Completed(_)))
        .unwrap();
    let c = match completed {
        StreamEvent::Completed(v) => v,
        _ => panic!(),
    };
    let j = serde_json::to_value(c).unwrap();
    assert_eq!(j["response"]["usage"]["input_tokens"], 10);
    assert_eq!(j["response"]["usage"]["output_tokens"], 5);
    assert_eq!(j["response"]["usage"]["total_tokens"], 15);
    assert_eq!(
        j["response"]["usage"]["output_tokens_details"]["reasoning_tokens"],
        3
    );
}

// ── Scenario 13: Streaming output_index no duplicates ────────────────

#[tokio::test]
async fn s13_streaming_output_index_unique() {
    let mut state = StreamState::new(
        "resp_test".into(),
        "msg_test".into(),
        "deepseek-v4-pro".into(),
    );

    let events = process_chunk_value(
        &mut state,
        serde_json::from_str(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"Let me check.","tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"search","arguments":"{}"}}]}}]}"#).unwrap(),
    )
    .unwrap();

    let indices: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::OutputItemAdded(v) => Some(v.output_index as u64),
            _ => None,
        })
        .collect();
    assert_eq!(indices.len(), 2);
    assert_ne!(indices[0], indices[1]);
    assert_eq!(indices[1], indices[0] + 1);
}

#[tokio::test]
async fn s13_streaming_output_index_with_reasoning() {
    let mut state = StreamState::new(
        "resp_test".into(),
        "msg_test".into(),
        "deepseek-v4-pro".into(),
    );

    process_chunk_value(
        &mut state,
        serde_json::from_str(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"reasoning_content":"Let me think"}}]}"#).unwrap(),
    );
    process_chunk_value(
        &mut state,
        serde_json::from_str(r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"Answer","tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"search","arguments":"{}"}}]}}]}"#).unwrap(),
    );

    // reasoning was closed at transition, message=1, tool_call=2
    assert!(state.reasoning_content.is_empty()); // cleared when text started
    assert!(!state.accumulated_text.is_empty());
    assert_eq!(state.msg_output_index, 1);
    assert_eq!(state.tool_calls[0].output_index, 2);
}

// ── Scenario 14: Streaming in_progress after created ─────────────────

#[tokio::test]
async fn s14_streaming_in_progress_emitted() {
    let mut state = StreamState::new(
        "resp_test".into(),
        "msg_test".into(),
        "deepseek-v4-pro".into(),
    );

    let events =
        process_chunk_value(&mut state, serde_json::from_str(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#).unwrap()).unwrap();

    let types: Vec<String> = events
        .iter()
        .map(|e| {
            serde_json::to_value(e).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    let created = types.iter().position(|t| *t == "response.created").unwrap();
    let in_progress = types
        .iter()
        .position(|t| *t == "response.in_progress")
        .unwrap();
    assert!(created < in_progress);
}

#[tokio::test]
async fn s14_streaming_in_progress_only_once() {
    let mut state = StreamState::new(
        "resp_test".into(),
        "msg_test".into(),
        "deepseek-v4-pro".into(),
    );

    process_chunk_value(
        &mut state,
        serde_json::from_str(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"A"}}]}"#).unwrap(),
    );
    // second chunk — no duplicate in_progress
    let events = process_chunk_value(&mut state, serde_json::from_str(r#"{"id":"c2","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"B"}}]}"#).unwrap()).unwrap();

    let has_in_progress = events
        .iter()
        .any(|e| matches!(e, StreamEvent::InProgress(_)));
    assert!(!has_in_progress);
}

// ── Scenario 15: Cached tokens ───────────────────────────────────────

#[tokio::test]
async fn s15_cached_tokens_openai_style() {
    let chat: chat::Completion = serde_json::from_value(json!({
        "id": "chatcmpl-cache",
        "object": "chat.completion",
        "created": 1715550000u64,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "response"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 80}
        }
    }))
    .unwrap();

    let resp = chat_to_responses(chat, "gpt-5.5".into(), None);
    let j = serde_json::to_value(&resp).unwrap();

    assert_eq!(j["usage"]["input_tokens"], 100);
    assert_eq!(j["usage"]["input_tokens_details"]["cached_tokens"], 80);
}

// ── Scenario 16: Instructions merged with input system message ───────

#[tokio::test]
async fn s16_instructions_merge_with_system() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are helpful."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ],
        "instructions": "Top-level instructions."
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    let msgs = j["messages"].as_array().unwrap();
    // instructions → system, input system → system, user → user = 3 messages
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "Top-level instructions.");
    assert_eq!(msgs[1]["role"], "system");
    assert_eq!(msgs[1]["content"], "You are helpful.");
    assert_eq!(msgs[2]["role"], "user");
}

// ── Scenario 17: Developer role preserved ─────────────────────────────

#[tokio::test]
async fn s17_developer_to_system() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Dev rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    // Developer message stays as developer (not converted to system)
    assert_eq!(j["messages"][0]["role"], "developer");
    assert_eq!(j["messages"][0]["content"], "Dev rules");
}

// ── Scenario 18: Reasoning in input history ──────────────────────────

#[tokio::test]
async fn s18_reasoning_item_in_input() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's weather?"}]},
            {"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "Let me check the API."}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    let msgs = j["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2); // user + assistant(tool_calls)
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_1");
    // reasoning content attached to the assistant(tool_calls) message
    assert_eq!(msgs[1]["reasoning_content"], "Let me check the API.");
}

// ── Scenario 19: Multiple content blocks joined ──────────────────────

#[tokio::test]
async fn s19_multiple_content_blocks_joined() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Hello"},
                {"type": "input_text", "text": "World"}
            ]}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["messages"][0]["content"], "Hello\nWorld");
}

// ── Scenario 20: Empty instructions ignored ──────────────────────────

#[tokio::test]
async fn s20_empty_instructions_ignored() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "instructions": ""
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["messages"].as_array().unwrap().len(), 1);
    assert_eq!(j["messages"][0]["role"], "user");
}

// ── Scenario 21: Image/file blocks silently dropped ──────────────────

#[tokio::test]
async fn s21_image_file_blocks_dropped() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Describe:"},
            {"type": "input_image", "image_url": "https://example.com/img.png"},
            {"type": "input_file", "file_id": "file-abc"}
        ]}]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    // Multimodal content is now passed through as Parts array
    let content = j["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Describe:");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(
        content[1]["image_url"]["url"],
        "https://example.com/img.png"
    );
    assert_eq!(content[2]["type"], "file");
    assert_eq!(content[2]["file"]["file_id"], "file-abc");
}

// ── Scenario 22: Unknown item/content silently skipped ───────────────

#[tokio::test]
async fn s22_unknown_items_skipped() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "item_reference", "id": "item_abc"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            {"type": "future_item", "data": "ignored"}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["messages"].as_array().unwrap().len(), 1);
    assert_eq!(j["messages"][0]["content"], "Hello");
}

// ── Scenario 23: String input with instructions ──────────────────────

#[tokio::test]
async fn s23_string_input_with_instructions() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "What is Rust?",
        "instructions": "You are a helpful assistant."
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    let msgs = j["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "What is Rust?");
}

// ── Scenario 24: Array input with username ──── (tests position) ─────

#[tokio::test]
async fn s24_function_call_output_array() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [{"type": "function_call_output", "call_id": "call_1", "output": [
            {"type": "input_text", "text": "Result text here"}
        ]}]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["messages"][0]["role"], "tool");
    assert_eq!(j["messages"][0]["content"], "Result text here");
    assert_eq!(j["messages"][0]["tool_call_id"], "call_1");
}

// ── Scenario 25: Stream options passthrough ──────────────────────────

#[tokio::test]
async fn s25_stream_true() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": "Hi",
        "stream": true
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    assert_eq!(j["stream"], true);
}

// ── Scenario 26: Consecutive function calls merge ────────────────────

#[tokio::test]
async fn s26_consecutive_function_calls_merge() {
    let req: responses::Request = serde_json::from_value(json!({
        "model": "gpt-5.5",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Get weather and time"}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "get_time", "arguments": "{\"tz\":\"EST\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
            {"type": "function_call_output", "call_id": "call_2", "output": "3pm"}
        ]
    }))
    .unwrap();

    let chat = responses_to_chat(req, &test_state()).await.unwrap();
    let j = serde_json::to_value(&chat).unwrap();

    let msgs = j["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 4);
    // Both function_calls merged into one assistant message
    assert_eq!(msgs[1]["role"], "assistant");
    assert!(msgs[1]["content"].is_null());
    assert_eq!(msgs[1]["tool_calls"].as_array().unwrap().len(), 2);
    // tool messages in order
    assert_eq!(msgs[2]["role"], "tool");
    assert_eq!(msgs[2]["tool_call_id"], "call_1");
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "call_2");
}
