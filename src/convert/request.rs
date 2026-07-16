use crate::types::{MessageRole, chat, item::*, responses};

// ── Main conversion: Responses API → Chat Completions API ────────────────

/// Convert a Responses API request into a Chat Completions API request.
///
/// Fetches previous conversation history from the store (via
/// `req.previous_response_id`) and handles prepending history + merging
/// `req.instructions` into the leading system message.
///
/// Returns an error with the list of unsupported Responses-only features.
pub async fn responses_to_chat(
    #[allow(unused_variables)] mut req: responses::Request,
    state: &crate::app::State,
) -> Result<chat::Request, Vec<String>> {
    let mut unsupported_features = Vec::new();
    if req.prompt.is_some() {
        unsupported_features.push("prompt".to_string());
    }
    if req.conversation.is_some() {
        unsupported_features.push("conversation".to_string());
    }
    if !unsupported_features.is_empty() {
        return Err(unsupported_features);
    }

    let mut messages: Vec<chat::MessageRequest> = Vec::new();
    let mut pending_reasoning: Option<String> = None;

    // Map reasoning effort — Responses API values match Chat API directly.
    // reasoning.summary has no Chat Completions equivalent and is left for rewrite profiles.
    let reasoning_str: Option<String> = {
        let r = req.reasoning.as_ref();
        let effort = r.and_then(|r| r.effort.as_ref());
        effort.map(|e| {
            match e {
                crate::types::ReasoningEffort::None => "none",
                crate::types::ReasoningEffort::Minimal => "minimal",
                crate::types::ReasoningEffort::Low => "low",
                crate::types::ReasoningEffort::Medium => "medium",
                crate::types::ReasoningEffort::High => "high",
                crate::types::ReasoningEffort::Xhigh => "xhigh",
                crate::types::ReasoningEffort::Max => "max",
                crate::types::ReasoningEffort::Ultra => "ultra",
            }
            .into()
        })
    };

    // Structured output → response_format
    let response_format = req
        .text
        .as_ref()
        .and_then(|t| t.format.as_ref())
        .map(|f| match f {
            responses::TextFormat::JsonSchema {
                name,
                schema,
                strict,
                description,
            } => chat::ResponseFormat::JsonSchema(chat::JsonSchemaFormat {
                format_type: "json_schema".into(),
                json_schema: chat::JsonSchema {
                    name: name.clone(),
                    description: description.clone(),
                    schema: Some(schema.clone()),
                    strict: *strict,
                },
            }),
            responses::TextFormat::JsonObject => {
                chat::ResponseFormat::JsonObject(chat::JsonObjectFormat {
                    format_type: "json_object".into(),
                })
            }
            responses::TextFormat::Text => chat::ResponseFormat::Text(chat::TextFormat {
                format_type: "text".into(),
            }),
        });

    // ── History + instructions ────────────────────────────────────────────
    let instructions = std::mem::take(&mut req.instructions).filter(|i| !i.is_empty());

    let prev_messages: Vec<chat::MessageRequest> = match req.previous_response_id {
        Some(ref prev_id) => state.store().get(prev_id).await.unwrap_or_default(),
        None => vec![],
    };

    if !prev_messages.is_empty() {
        messages = prev_messages;

        // The first message in stored history is always a system message
        // holding the instructions from the previous turn.
        if let Some(first) = messages.first_mut()
            && matches!(first, chat::MessageRequest::System(_))
        {
            if let Some(new_instructions) = instructions {
                *first = chat::MessageRequest::System(chat::SystemMessage {
                    content: chat::MessageContent::Text(new_instructions),
                    name: None,
                });
            }
        } else if let Some(new_instructions) = instructions {
            messages.insert(
                0,
                chat::MessageRequest::System(chat::SystemMessage {
                    content: chat::MessageContent::Text(new_instructions),
                    name: None,
                }),
            );
        }
    } else if let Some(ref new_instructions) = instructions {
        messages.push(chat::MessageRequest::System(chat::SystemMessage {
            content: chat::MessageContent::Text(new_instructions.clone()),
            name: None,
        }));
    }

    // Codex gpt-5.6 code-mode tool definitions arrive inside `additional_tools`
    // input items; collect their raw entries here and flatten them into Chat
    // function tools further down.
    let mut additional_tool_defs: Vec<serde_json::Value> = Vec::new();

    // Names Codex declared as custom (code-mode) — used to keep replayed history
    // calls consistent with the `{ input }` function schema we present.
    let custom_names = custom_tool_names(&req.input);

    // Walk input items
    let items: Vec<InputItem> = req.input;
    if !items.is_empty() {
        let cutoff = old_tool_output_cutoff(&items, KEEP_LAST_TURNS);
        let mut deferred: Vec<chat::MessageRequest> = Vec::new();
        let mut pending_tool_calls: Vec<chat::ToolCallRequest> = Vec::new();

        let flush_tools = |msgs: &mut Vec<chat::MessageRequest>,
                           p: &mut Vec<chat::ToolCallRequest>,
                           r: &mut Option<String>| {
            if !p.is_empty() {
                msgs.push(chat::MessageRequest::Assistant(chat::AssistantMessage {
                    content: None,
                    name: None,
                    refusal: None,
                    audio: None,
                    tool_calls: Some(std::mem::take(p)),
                    function_call: None,
                    reasoning_content: r.take(),
                }));
            }
        };

        for (idx, item) in items.into_iter().enumerate() {
            match item {
                InputItem::FunctionCallOutput(fco) => {
                    let cs = match &fco.output {
                        FunctionOutputValue::String(s) => s.clone(),
                        FunctionOutputValue::Array(blocks) => {
                            extract_text_from_output_blocks(blocks)
                        }
                    };
                    deferred.push(chat::MessageRequest::Tool(chat::ToolMessage {
                        content: chat::MessageContent::Text(tool_output_for_age(cs, idx, cutoff)),
                        tool_call_id: fco.call_id.clone(),
                    }));
                }
                InputItem::Reasoning(r) => {
                    flush_tools(
                        &mut messages,
                        &mut pending_tool_calls,
                        &mut pending_reasoning,
                    );
                    messages.append(&mut deferred);
                    if let Some(t) = extract_reasoning(&r, state.compact_key()) {
                        pending_reasoning = Some(match pending_reasoning.take() {
                            Some(e) if e.contains(&t) => e,
                            Some(e) => format!("{}\n{}", e, t),
                            None => t,
                        });
                    }
                }
                InputItem::Compaction(c) => {
                    flush_tools(
                        &mut messages,
                        &mut pending_tool_calls,
                        &mut pending_reasoning,
                    );
                    messages.append(&mut deferred);
                    // Decrypt encrypted_content into a system message
                    if let Some(ref encrypted) = c.encrypted_content
                        && let Some(key) = state.compact_key()
                        && let Some(decrypted) = crate::crypto::decrypt(key, encrypted)
                        && !decrypted.is_empty()
                    {
                        messages.push(chat::MessageRequest::System(chat::SystemMessage {
                            content: chat::MessageContent::Text(decrypted),
                            name: None,
                        }));
                    }
                }
                InputItem::ContextCompaction(c) => {
                    flush_tools(
                        &mut messages,
                        &mut pending_tool_calls,
                        &mut pending_reasoning,
                    );
                    messages.append(&mut deferred);
                    if let Some(ref encrypted) = c.encrypted_content
                        && let Some(key) = state.compact_key()
                        && let Some(decrypted) = crate::crypto::decrypt(key, encrypted)
                        && !decrypted.is_empty()
                    {
                        messages.push(chat::MessageRequest::System(chat::SystemMessage {
                            content: chat::MessageContent::Text(decrypted),
                            name: None,
                        }));
                    }
                }
                InputItem::FunctionCall(fc) => {
                    pending_tool_calls.push(chat::ToolCallRequest::Function {
                        id: fc.call_id.clone(),
                        function: chat::ToolCallFunction {
                            name: fc.name.clone(),
                            arguments: fc.arguments.clone(),
                        },
                    });
                }
                InputItem::Message(msg) => {
                    flush_tools(
                        &mut messages,
                        &mut pending_tool_calls,
                        &mut pending_reasoning,
                    );
                    messages.append(&mut deferred);
                    let reasoning = match msg.role {
                        MessageRole::Assistant => pending_reasoning.clone(),
                        _ => {
                            pending_reasoning.take();
                            None
                        }
                    };
                    if let Some(chat_msg) = convert_input_message(msg, reasoning) {
                        messages.push(chat_msg);
                    }
                }
                InputItem::CustomToolCall(ctc) => {
                    // We present code-mode custom tools to the model as functions
                    // taking `{ input: string }`, so the replayed assistant call
                    // must use that JSON shape (raw freeform is not valid JSON
                    // arguments). Other custom tools keep their raw input.
                    let arguments = if custom_names.contains(&ctc.name) {
                        serde_json::json!({ "input": ctc.input }).to_string()
                    } else {
                        ctc.input.clone()
                    };
                    pending_tool_calls.push(chat::ToolCallRequest::Function {
                        id: ctc.call_id.clone(),
                        function: chat::ToolCallFunction {
                            name: ctc.name.clone(),
                            arguments,
                        },
                    });
                }
                InputItem::CustomToolCallOutput(ctco) => {
                    let cs = match &ctco.output {
                        CustomToolOutputValue::String(s) => s.clone(),
                        CustomToolOutputValue::Array(blocks) => {
                            extract_text_from_output_blocks(blocks)
                        }
                    };
                    deferred.push(chat::MessageRequest::Tool(chat::ToolMessage {
                        content: chat::MessageContent::Text(tool_output_for_age(cs, idx, cutoff)),
                        tool_call_id: ctco.call_id.clone(),
                    }));
                }
                // Trigger for remote compaction v2: handled before reaching the
                // chat converter, so silently drop if it slips through here.
                InputItem::CompactionTrigger(_) => {}
                // Items Codex replays every turn that have no Chat-Completions
                // equivalent. Drop silently — they carry large opaque payloads
                // (image base64, search results) that must not be logged.
                InputItem::WebSearchCall(_)
                | InputItem::ImageGenerationCall(_)
                | InputItem::ToolSearchCall(_)
                | InputItem::ToolSearchOutput(_) => {}
                // Codex gpt-5.6 code-mode tool definitions — collect for
                // flattening into Chat function tools (not a conversation item).
                InputItem::AdditionalTools(at) => {
                    additional_tool_defs.extend(at.tools);
                }
                other => {
                    tracing::warn!(
                        item = %unconvertible_item_label(&other),
                        "InputItem variant not convertible to Chat API — skipping"
                    );
                }
            }
        }
        flush_tools(
            &mut messages,
            &mut pending_tool_calls,
            &mut pending_reasoning,
        );
        messages.append(&mut deferred);
    }

    // top_logprobs is a Chat Completions concept; Responses API uses include: ["message.output_text.logprobs"]
    let logprobs: Option<bool> = None;
    let top_logprobs_val: Option<i64> = None;

    let reasoning_effort = reasoning_str.as_ref().map(|s| match s.as_str() {
        "none" => crate::types::ReasoningEffort::None,
        "minimal" => crate::types::ReasoningEffort::Minimal,
        "low" => crate::types::ReasoningEffort::Low,
        "medium" => crate::types::ReasoningEffort::Medium,
        "high" => crate::types::ReasoningEffort::High,
        "xhigh" => crate::types::ReasoningEffort::Xhigh,
        // gpt-5.6-class tiers above xhigh — passed through verbatim.
        "max" => crate::types::ReasoningEffort::Max,
        "ultra" => crate::types::ReasoningEffort::Ultra,
        _ => crate::types::ReasoningEffort::None,
    });

    // Responses streaming completion events include final usage. Chat upstreams only
    // provide that reliably when include_usage is enabled.
    let stream_options = if req.stream {
        Some(chat::StreamOptions {
            include_usage: Some(true),
            include_obfuscation: Some(
                req.stream_options
                    .as_ref()
                    .is_none_or(|so| so.include_obfuscation),
            ),
        })
    } else {
        req.stream_options.as_ref().map(|so| chat::StreamOptions {
            include_usage: None,
            include_obfuscation: Some(so.include_obfuscation),
        })
    };

    // Merge verbosity: TextConfig.verbosity takes precedence over top-level verbosity
    let verbosity = req
        .text
        .as_ref()
        .and_then(|t| t.verbosity.as_ref())
        .or(req.verbosity.as_ref())
        .cloned();

    // Reason: summary / generate_summary — log if set (profile decides downstream field)
    if let Some(ref r) = req.reasoning {
        if r.summary.is_some() {
            tracing::debug!(?r.summary, "reasoning.summary set");
        }
        if r.generate_summary.is_some() {
            tracing::debug!(?r.generate_summary, "reasoning.generate_summary set (deprecated)");
        }
    }

    // Build the Chat tool list. Top-level `req.tools` keeps its existing
    // conversion untouched (no behavior change for gpt-5.5 and earlier). Codex
    // gpt-5.6 instead ships tool definitions inside `additional_tools` input
    // items using the code-mode `custom`/`namespace` protocol, which Chat
    // Completions rejects — those are flattened into plain `function` tools and
    // appended. When there are no `additional_tools`, this is a no-op.
    let mut chat_tools: Vec<chat::ToolRequest> = req
        .tools
        .as_ref()
        .map(|tools| {
            let allow = &state.config().allowed_tool_types;
            tools
                .iter()
                .filter_map(|t| match t {
                    crate::types::tool::ToolRequest::Function(f) => {
                        if allow.contains(&"function".to_string()) {
                            Some(chat::ToolRequest::Function {
                                function: chat::FunctionTool {
                                    name: f.name.clone().unwrap_or_default(),
                                    description: f.description.clone(),
                                    parameters: f.parameters.clone(),
                                    strict: f.strict,
                                },
                            })
                        } else {
                            None
                        }
                    }
                    crate::types::tool::ToolRequest::Custom(c) => {
                        if allow.contains(&"custom".to_string()) {
                            Some(chat::ToolRequest::Custom {
                                custom: chat::CustomTool {
                                    name: c.name.clone().unwrap_or_default(),
                                    description: c.description.clone(),
                                    format: c.format.as_ref().map(|f| match f {
                                        crate::types::tool::CustomToolFormat::Text(_) => {
                                            chat::CustomToolFormat::Text
                                        }
                                        crate::types::tool::CustomToolFormat::Grammar(g) => {
                                            chat::CustomToolFormat::Grammar {
                                                grammar: chat::Grammar {
                                                    definition: g.definition.clone(),
                                                    syntax: g.syntax.clone(),
                                                },
                                            }
                                        }
                                    }),
                                },
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for v in &additional_tool_defs {
        chat_tools.extend(additional_tools_to_chat_functions(v));
    }
    if !additional_tool_defs.is_empty() {
        tracing::debug!(
            hoisted = chat_tools.len(),
            names = ?chat_tools.iter().map(chat_tool_name).collect::<Vec<_>>(),
            "hoisted Codex additional_tools into chat function tools"
        );
    }
    let tools = if chat_tools.is_empty() {
        None
    } else {
        Some(chat_tools)
    };

    Ok(chat::Request {
        model: req.model,
        messages,
        temperature: Some(req.temperature),
        top_p: Some(req.top_p),
        max_completion_tokens: req.max_output_tokens,
        stream: Some(req.stream),
        stream_options,
        // Pass through request metadata
        prompt_cache_key: req.prompt_cache_key.clone(),
        prompt_cache_retention: req.prompt_cache_retention.clone(),
        safety_identifier: req.safety_identifier.clone(),
        service_tier: req.service_tier.clone(),
        verbosity,
        parallel_tool_calls: Some(req.parallel_tool_calls),
        store: Some(req.store),
        tools,
        tool_choice: req.tool_choice.and_then(convert_tool_choice),
        response_format,
        stop: req.stop,
        logprobs,
        top_logprobs: top_logprobs_val,
        reasoning_effort,
        ..Default::default()
    })
}

/// Names of tools Codex declared as `custom` (code-mode `exec`, `namespace`
/// custom members). Their calls must be mapped back to the `custom_tool_call`
/// shape Codex expects on the response side, and their replayed history calls
/// wrapped to match the `{ input }` function schema we present. Mirrors the
/// naming used by [`additional_tools_to_chat_functions`].
pub fn custom_tool_names(input: &[InputItem]) -> std::collections::HashSet<String> {
    use crate::types::tool::{NamespaceToolItem, ToolRequest as Rt};
    let mut names = std::collections::HashSet::new();
    for item in input {
        let InputItem::AdditionalTools(at) = item else {
            continue;
        };
        for v in &at.tools {
            let Ok(parsed) = serde_json::from_value::<Rt>(v.clone()) else {
                continue;
            };
            match parsed {
                Rt::Custom(c) => {
                    if let Some(n) = c.name {
                        if !is_multi_agent_hosted_action(&n) {
                            names.insert(n);
                        }
                    }
                }
                Rt::Namespace(ns) => {
                    let is_collab = ns.name.as_deref() == Some("collaboration");
                    for m in ns.tools.unwrap_or_default() {
                        if let NamespaceToolItem::Custom(nc) = m {
                            if !is_collab && !is_multi_agent_hosted_action(&nc.name) {
                                names.insert(nc.name);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    names
}

/// Hosted multi-agent collaboration actions (Responses "multi-agent" beta).
/// These spawn and coordinate sub-agents inside OpenAI's hosted Responses
/// runtime — the client never executes them, so a Chat Completions upstream
/// (which has no sub-agent orchestration) cannot fulfil them. Advertising them
/// lures the model into calls the client rejects as `unsupported call`, which
/// derails the whole session, so they are dropped during conversion.
fn is_multi_agent_hosted_action(name: &str) -> bool {
    matches!(
        name,
        "spawn_agent"
            | "send_message"
            | "followup_task"
            | "wait_agent"
            | "interrupt_agent"
            | "list_agents"
    )
}

/// Flatten one Codex `additional_tools` entry (code-mode
/// `custom`/`namespace`/`function`) into Chat Completions `function` tools.
/// Unrecognized entries are skipped. Tool names are preserved verbatim so the
/// `function_call` the model emits routes back to the matching Codex tool.
fn additional_tools_to_chat_functions(v: &serde_json::Value) -> Vec<chat::ToolRequest> {
    use crate::types::tool::{NamespaceToolItem, ToolRequest as Rt};
    let parsed: Rt = match serde_json::from_value(v.clone()) {
        Ok(t) => t,
        Err(e) => {
            tracing::debug!(error = %e, "additional_tools: unrecognized tool entry, skipping");
            return Vec::new();
        }
    };
    match parsed {
        Rt::Function(f) => {
            let name = f.name.unwrap_or_default();
            if is_multi_agent_hosted_action(&name) {
                tracing::warn!(
                    skipped = %name,
                    "additional_tools: dropping hosted multi-agent tool — not executable via Chat Completions"
                );
                return Vec::new();
            }
            vec![function_tool(name, f.description, f.parameters, f.strict)]
        }
        Rt::Namespace(ns) => {
            // The `collaboration` namespace carries the hosted multi-agent
            // actions (spawn_agent, …). They run inside OpenAI's hosted
            // Responses runtime, not the client, so a Chat Completions upstream
            // cannot fulfil them. Drop the whole namespace so the model never
            // emits a `spawn_agent` call the client rejects as `unsupported
            // call` — it falls back to the client-executable tools instead.
            let is_collab = ns.name.as_deref() == Some("collaboration");
            let mut skipped: Vec<String> = Vec::new();
            let out: Vec<_> = ns
                .tools
                .unwrap_or_default()
                .into_iter()
                .filter_map(|item| {
                    let name = match &item {
                        NamespaceToolItem::Function(nf) => nf.name.as_str(),
                        NamespaceToolItem::Custom(nc) => nc.name.as_str(),
                    };
                    if is_collab || is_multi_agent_hosted_action(name) {
                        skipped.push(name.to_string());
                        return None;
                    }
                    Some(match item {
                        NamespaceToolItem::Function(nf) => {
                            function_tool(nf.name, nf.description, nf.parameters, nf.strict)
                        }
                        NamespaceToolItem::Custom(nc) => {
                            custom_as_function(nc.name, nc.description)
                        }
                    })
                })
                .collect();
            if !skipped.is_empty() {
                tracing::warn!(
                    skipped = ?skipped,
                    "additional_tools: dropping hosted multi-agent tools — not executable via Chat Completions"
                );
            }
            out
        }
        Rt::Custom(c) => {
            let name = c.name.unwrap_or_default();
            if is_multi_agent_hosted_action(&name) {
                tracing::warn!(
                    skipped = %name,
                    "additional_tools: dropping hosted multi-agent tool — not executable via Chat Completions"
                );
                return Vec::new();
            }
            vec![custom_as_function(name, c.description)]
        }
        // e.g. `mcp` (remote server reference — no per-tool schema to flatten),
        // `tool_search`, `web_search`. Codex 5.6 delivers usable MCP tools as a
        // `namespace` (handled above); anything landing here has no Chat
        // Completions function equivalent.
        _ => {
            tracing::debug!(
                tool_type = %v.get("type").and_then(|t| t.as_str()).unwrap_or("?"),
                "additional_tools: tool type has no Chat function equivalent — skipping"
            );
            Vec::new()
        }
    }
}

fn function_tool(
    name: String,
    description: Option<String>,
    parameters: Option<serde_json::Value>,
    strict: Option<bool>,
) -> chat::ToolRequest {
    chat::ToolRequest::Function {
        function: chat::FunctionTool {
            name,
            description,
            parameters,
            strict,
        },
    }
}

/// Represent a code-mode `custom` tool (freeform text / code) as a Chat
/// `function` tool taking a single freeform string argument — Chat Completions
/// has no freeform/grammar tool type.
fn custom_as_function(name: String, description: Option<String>) -> chat::ToolRequest {
    function_tool(
        name,
        description,
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Raw tool input (freeform text or code)."
                }
            },
            "required": ["input"],
            "additionalProperties": false
        })),
        Some(false),
    )
}

fn chat_tool_name(t: &chat::ToolRequest) -> String {
    match t {
        chat::ToolRequest::Function { function } => function.name.clone(),
        chat::ToolRequest::Custom { custom } => custom.name.clone(),
    }
}

fn convert_tool_choice(tc: crate::types::tool::ToolChoice) -> Option<chat::ToolChoice> {
    match tc {
        crate::types::tool::ToolChoice::String(s) => Some(chat::ToolChoice::Mode(s)),
        crate::types::tool::ToolChoice::Specific(s) if s.tool_type == "custom" => {
            Some(chat::ToolChoice::Custom(chat::ToolChoiceCustom {
                choice_type: "custom".into(),
                custom: chat::ToolChoiceCustomName { name: s.name },
            }))
        }
        crate::types::tool::ToolChoice::Specific(s) if s.tool_type == "function" => {
            Some(chat::ToolChoice::Function(chat::ToolChoiceFunction {
                choice_type: "function".into(),
                function: chat::ToolChoiceFunctionName { name: s.name },
            }))
        }
        crate::types::tool::ToolChoice::Mode(m) => Some(chat::ToolChoice::AllowedTools(
            chat::ToolChoiceAllowedTools {
                choice_type: "allowed_tools".into(),
                mode: m.mode,
                tools: m.tools,
            },
        )),
        other => {
            tracing::warn!(
                ?other,
                "tool_choice variant not convertible to Chat API — skipping"
            );
            None
        }
    }
}

// ── Tool-output truncation (age-based) ────────────────────────────────────
//
// Tool outputs dominate replayed context (~78% of bytes in real Codex
// sessions). Outputs from the last `KEEP_LAST_TURNS` user turns are kept
// verbatim because the model is likely still acting on them; older ones are
// truncated to head+tail with a marker.

const KEEP_LAST_TURNS: usize = 6;
const MAX_OLD_TOOL_OUTPUT_CHARS: usize = 2048;
const OLD_TOOL_OUTPUT_HEAD: usize = 1024;
const OLD_TOOL_OUTPUT_TAIL: usize = 512;

/// Index in `items` before which tool outputs are "old" and may be truncated.
/// Everything from this index onward belongs to the last `keep_last_turns`
/// user turns and is preserved verbatim. Returns 0 when there are not enough
/// turns to truncate anything.
fn old_tool_output_cutoff(items: &[InputItem], keep_last_turns: usize) -> usize {
    let user_positions: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, it)| matches!(it, InputItem::Message(m) if m.role == MessageRole::User))
        .map(|(i, _)| i)
        .collect();
    if user_positions.len() <= keep_last_turns {
        return 0;
    }
    user_positions[user_positions.len() - keep_last_turns]
}

fn floor_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut i = idx;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Truncate an old tool output to head+tail with a marker, respecting UTF-8
/// boundaries. Returns the input unchanged when within budget or when the
/// head+tail would not actually save anything.
fn truncate_old_tool_output(s: String) -> String {
    if s.len() <= MAX_OLD_TOOL_OUTPUT_CHARS {
        return s;
    }
    let head_end = floor_char_boundary(&s, OLD_TOOL_OUTPUT_HEAD);
    let tail_start = ceil_char_boundary(&s, s.len().saturating_sub(OLD_TOOL_OUTPUT_TAIL));
    if tail_start <= head_end {
        return s;
    }
    let omitted = tail_start - head_end;
    format!(
        "{}\n…[truncated {} bytes]…\n{}",
        &s[..head_end],
        omitted,
        &s[tail_start..]
    )
}

/// Apply age-based truncation to a tool-output string given its item index and
/// the precomputed cutoff.
fn tool_output_for_age(cs: String, idx: usize, cutoff: usize) -> String {
    if idx < cutoff {
        truncate_old_tool_output(cs)
    } else {
        cs
    }
}

/// Serialized character size of a converted Chat message list.
fn messages_total_chars(messages: &[chat::MessageRequest]) -> usize {
    messages
        .iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0))
        .sum()
}

/// Enforce a total character ceiling on a converted Chat request. Age-based
/// truncation shrinks per-output; this is the global backstop the per-model
/// `history.max-input-chars` config drives. It walks tool messages from oldest
/// to newest, replacing their content with a head+tail marker, until the total
/// serialized size fits under `max_chars` (or no tool output remains to shrink).
/// Tool-call/output pairing is never broken — only the textual content shrinks.
/// Returns the number of messages whose content was reduced.
pub fn enforce_input_budget(messages: &mut [chat::MessageRequest], max_chars: usize) -> usize {
    if messages_total_chars(messages) <= max_chars {
        return 0;
    }
    let mut shrunk = 0;
    for idx in 0..messages.len() {
        if let chat::MessageRequest::Tool(t) = &messages[idx] {
            let text = match &t.content {
                chat::MessageContent::Text(s) => s.clone(),
                chat::MessageContent::Parts(parts) => {
                    parts.iter().map(|p| p.text.as_str()).collect::<String>()
                }
            };
            if text.len() > MAX_OLD_TOOL_OUTPUT_CHARS {
                let truncated = truncate_old_tool_output(text);
                if let chat::MessageRequest::Tool(t) = &mut messages[idx] {
                    t.content = chat::MessageContent::Text(truncated);
                }
                shrunk += 1;
                if messages_total_chars(messages) <= max_chars {
                    break;
                }
            }
        }
    }
    shrunk
}

/// Trim the converted Chat message list to at most `max_messages` by dropping
/// the oldest conversation turns. Leading System/Developer messages (the
/// instructions block) are always kept. Cuts only at user-message boundaries so
/// assistant tool_calls stay paired with their tool responses. Returns the
/// number of messages dropped. `max_messages == 0` means unlimited.
pub fn enforce_message_budget(
    messages: &mut Vec<chat::MessageRequest>,
    max_messages: usize,
) -> usize {
    if max_messages == 0 || messages.len() <= max_messages {
        return 0;
    }

    // Leading run of System/Developer messages — always retained.
    let prefix_len = messages
        .iter()
        .take_while(|m| {
            matches!(
                m,
                chat::MessageRequest::System(_) | chat::MessageRequest::Developer(_)
            )
        })
        .count();

    // User-message positions after the prefix — each starts a conversation turn.
    let user_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .skip(prefix_len)
        .filter(|(_, m)| matches!(m, chat::MessageRequest::User(_)))
        .map(|(i, _)| i)
        .collect();

    // Earliest turn start that makes prefix + tail fit under the limit. Falls
    // back to the last turn if even one turn plus prefix still overflows (a
    // single enormous turn we cannot split without breaking tool pairing).
    let start = match user_starts
        .iter()
        .copied()
        .find(|&s| prefix_len + (messages.len() - s) <= max_messages)
    {
        Some(s) => s,
        None => match user_starts.last() {
            Some(&s) => s,
            None => return 0,
        },
    };

    if start <= prefix_len {
        return 0;
    }
    let removed = start - prefix_len;
    messages.drain(prefix_len..start);
    removed
}

/// A short, log-safe label for an input item that the converter does not map to
/// a Chat message. Codex replays large items (image base64, encrypted
/// summaries) that must never be logged verbatim — only the discriminant `type`
/// and length are emitted.
fn unconvertible_item_label(item: &InputItem) -> String {
    match serde_json::to_value(item) {
        Ok(value) => {
            let kind = value
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown")
                .to_string();
            let bytes = value.to_string().len();
            format!("{kind} ({bytes} bytes)")
        }
        Err(_) => "unserializable".to_string(),
    }
}

// ── Bulk conversion: Vec<InputItem> → Vec<MessageRequest> ─

pub fn items_to_chat_messages(
    items: &[InputItem],
    state: &crate::app::State,
) -> Vec<chat::MessageRequest> {
    let cutoff = old_tool_output_cutoff(items, KEEP_LAST_TURNS);
    let mut messages: Vec<chat::MessageRequest> = Vec::new();
    let mut pending_reasoning: Option<String> = None;
    let mut deferred: Vec<chat::MessageRequest> = Vec::new();
    let mut pending_tool_calls: Vec<chat::ToolCallRequest> = Vec::new();

    let flush = |msgs: &mut Vec<chat::MessageRequest>,
                 p: &mut Vec<chat::ToolCallRequest>,
                 r: &mut Option<String>| {
        if !p.is_empty() {
            msgs.push(chat::MessageRequest::Assistant(chat::AssistantMessage {
                content: None,
                name: None,
                refusal: None,
                audio: None,
                tool_calls: Some(std::mem::take(p)),
                function_call: None,
                reasoning_content: r.take(),
            }));
        }
    };

    for (idx, item) in items.iter().enumerate() {
        match item {
            InputItem::FunctionCallOutput(fco) => {
                let cs = match &fco.output {
                    FunctionOutputValue::String(s) => s.clone(),
                    FunctionOutputValue::Array(blocks) => extract_text_from_output_blocks(blocks),
                };
                deferred.push(chat::MessageRequest::Tool(chat::ToolMessage {
                    content: chat::MessageContent::Text(tool_output_for_age(cs, idx, cutoff)),
                    tool_call_id: fco.call_id.clone(),
                }));
            }
            InputItem::Reasoning(r) => {
                flush(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                );
                messages.append(&mut deferred);
                if let Some(t) = extract_reasoning(r, state.compact_key()) {
                    pending_reasoning = Some(match pending_reasoning.take() {
                        Some(e) if e.contains(&t) => e,
                        Some(e) => format!("{}\n{}", e, t),
                        None => t,
                    });
                }
            }
            InputItem::Compaction(c) => {
                flush(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                );
                messages.append(&mut deferred);
                // Decrypt encrypted_content into a system message
                if let Some(ref encrypted) = c.encrypted_content
                    && let Some(key) = state.compact_key()
                    && let Some(text) = crate::crypto::decrypt(key, encrypted)
                    && !text.is_empty()
                {
                    messages.push(chat::MessageRequest::System(chat::SystemMessage {
                        content: chat::MessageContent::Text(text),
                        name: None,
                    }));
                }
            }
            InputItem::ContextCompaction(c) => {
                flush(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                );
                messages.append(&mut deferred);
                if let Some(ref encrypted) = c.encrypted_content
                    && let Some(key) = state.compact_key()
                    && let Some(text) = crate::crypto::decrypt(key, encrypted)
                    && !text.is_empty()
                {
                    messages.push(chat::MessageRequest::System(chat::SystemMessage {
                        content: chat::MessageContent::Text(text),
                        name: None,
                    }));
                }
            }
            InputItem::FunctionCall(fc) => {
                pending_tool_calls.push(chat::ToolCallRequest::Function {
                    id: fc.call_id.clone(),
                    function: chat::ToolCallFunction {
                        name: fc.name.clone(),
                        arguments: fc.arguments.clone(),
                    },
                });
            }
            InputItem::Message(msg) => {
                flush(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                );
                messages.append(&mut deferred);
                let r = match msg.role {
                    MessageRole::Assistant => pending_reasoning.clone(),
                    _ => {
                        pending_reasoning.take();
                        None
                    }
                };
                if let Some(chat_msg) = convert_input_message(msg.clone(), r) {
                    messages.push(chat_msg);
                }
            }
            InputItem::CustomToolCall(ctc) => {
                pending_tool_calls.push(chat::ToolCallRequest::Function {
                    id: ctc.call_id.clone(),
                    function: chat::ToolCallFunction {
                        name: ctc.name.clone(),
                        arguments: ctc.input.clone(),
                    },
                });
            }
            InputItem::CustomToolCallOutput(ctco) => {
                let cs = match &ctco.output {
                    CustomToolOutputValue::String(s) => s.clone(),
                    CustomToolOutputValue::Array(blocks) => extract_text_from_output_blocks(blocks),
                };
                deferred.push(chat::MessageRequest::Tool(chat::ToolMessage {
                    content: chat::MessageContent::Text(tool_output_for_age(cs, idx, cutoff)),
                    tool_call_id: ctco.call_id.clone(),
                }));
            }
            // Trigger for remote compaction v2 — handled upstream of conversion.
            InputItem::CompactionTrigger(_) => {}
            // Items Codex replays every turn with no Chat-Completions
            // equivalent. Drop silently — large opaque payloads.
            InputItem::WebSearchCall(_)
            | InputItem::ImageGenerationCall(_)
            | InputItem::ToolSearchCall(_)
            | InputItem::ToolSearchOutput(_) => {}
            other => {
                tracing::warn!(
                    item = %unconvertible_item_label(other),
                    "InputItem variant not convertible to Chat API in items_to_chat_messages — skipping"
                );
            }
        }
    }
    flush(
        &mut messages,
        &mut pending_tool_calls,
        &mut pending_reasoning,
    );
    messages.append(&mut deferred);
    messages
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn convert_input_message(
    msg: InputMessage,
    reasoning: Option<String>,
) -> Option<chat::MessageRequest> {
    if msg.content.is_empty() && msg.role != MessageRole::Assistant {
        return None;
    }
    let mapped_role = match msg.role {
        MessageRole::User => "user",
        MessageRole::System => "system",
        MessageRole::Developer => "developer",
        MessageRole::Assistant => "assistant",
        _ => return None,
    };

    match mapped_role {
        "system" => {
            let text = extract_text(&msg.content);
            Some(chat::MessageRequest::System(chat::SystemMessage {
                content: chat::MessageContent::Text(text),
                name: None,
            }))
        }
        "developer" => {
            let text = extract_text(&msg.content);
            Some(chat::MessageRequest::Developer(chat::DeveloperMessage {
                content: chat::MessageContent::Text(text),
                name: None,
            }))
        }
        "user" => {
            let parts = convert_content_to_user_parts(&msg.content);
            if parts.is_empty() {
                None
            } else if parts
                .iter()
                .all(|p| matches!(p, chat::ContentPart::Text { .. }))
            {
                // All-text content → join into plain string format for compatibility
                let text: String = parts
                    .iter()
                    .filter_map(|p| match p {
                        chat::ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Some(chat::MessageRequest::User(chat::UserMessage {
                    content: chat::UserContent::Text(text),
                    name: None,
                }))
            } else {
                // Multimodal content → Parts format
                Some(chat::MessageRequest::User(chat::UserMessage {
                    content: chat::UserContent::Parts(parts),
                    name: None,
                }))
            }
        }
        "assistant" => {
            let text = extract_text(&msg.content);
            Some(chat::MessageRequest::Assistant(chat::AssistantMessage {
                content: if text.is_empty() {
                    None
                } else {
                    Some(chat::AssistantContent::Text(text))
                },
                name: None,
                refusal: None,
                audio: None,
                tool_calls: None,
                function_call: None,
                reasoning_content: reasoning,
            }))
        }
        _ => None,
    }
}

/// Convert Responses input content blocks to Chat user content parts.
fn convert_content_to_user_parts(blocks: &[InputContentBlock]) -> Vec<chat::ContentPart> {
    blocks
        .iter()
        .filter_map(|b| match b {
            InputContentBlock::Text { text } => {
                Some(chat::ContentPart::Text { text: text.clone() })
            }
            InputContentBlock::Image {
                image_url, detail, ..
            } => {
                let url = image_url.clone().unwrap_or_default();
                if url.is_empty() {
                    None
                } else {
                    Some(chat::ContentPart::Image {
                        image_url: chat::ImageUrl {
                            url,
                            detail: detail.clone(),
                        },
                    })
                }
            }
            InputContentBlock::File {
                file_id,
                file_url,
                file_data,
                filename,
            } => {
                if file_id.is_none() && file_url.is_none() && file_data.is_none() {
                    None
                } else {
                    Some(chat::ContentPart::File {
                        file: chat::FileData {
                            file_data: file_data.clone(),
                            file_id: file_id.clone(),
                            filename: filename.clone(),
                        },
                    })
                }
            }
            InputContentBlock::Audio { data, format } => Some(chat::ContentPart::Audio {
                input_audio: chat::InputAudio {
                    data: data.clone(),
                    format: format.clone(),
                },
            }),
        })
        .collect()
}

fn extract_text(blocks: &[InputContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            InputContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_text_from_output_blocks(blocks: &[InputContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            InputContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_reasoning(r: &Reasoning, decrypt_key: Option<&[u8; 32]>) -> Option<String> {
    // Try encrypted_content first (with decryption), then plain content, then summary.
    if let Some(ref encrypted) = r.encrypted_content
        && let Some(key) = decrypt_key
        && let Some(decrypted) = crate::crypto::decrypt(key, encrypted)
        && !decrypted.is_empty()
    {
        return Some(decrypted);
    }
    let mut parts = Vec::new();
    for v in &r.summary {
        parts.push(v.text.clone());
    }
    if let Some(ref content) = r.content {
        for v in content {
            parts.push(v.text.clone());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> InputItem {
        InputItem::Message(InputMessage {
            role: MessageRole::User,
            content: vec![InputContentBlock::Text {
                text: text.to_string(),
            }],
            status: None,
        })
    }

    fn tool_out(call_id: &str, output: &str) -> InputItem {
        InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: call_id.to_string(),
            output: FunctionOutputValue::String(output.to_string()),
            id: None,
            status: None,
        })
    }

    #[test]
    fn cutoff_zero_when_few_turns() {
        let items = vec![user("a"), tool_out("c1", "x"), user("b")];
        assert_eq!(old_tool_output_cutoff(&items, 6), 0);
    }

    #[test]
    fn cutoff_marks_old_turns() {
        // 8 user turns, keep last 6 → cutoff at the 3rd user position (index 2).
        let mut items = Vec::new();
        for _ in 0..8 {
            items.push(user("u"));
        }
        let cutoff = old_tool_output_cutoff(&items, 6);
        // user positions are 0..8; keep last 6 → boundary at position index 2.
        assert_eq!(cutoff, 2);
    }

    #[test]
    fn truncate_old_output_adds_marker() {
        let big = "A".repeat(10_000);
        let out = truncate_old_tool_output(big);
        assert!(out.contains("…[truncated"));
        assert!(out.starts_with(&"A".repeat(OLD_TOOL_OUTPUT_HEAD)));
        assert!(out.ends_with(&"A".repeat(OLD_TOOL_OUTPUT_TAIL)));
        assert!(out.len() < 10_000);
    }

    #[test]
    fn truncate_keeps_small_output() {
        let small = "tiny".to_string();
        assert_eq!(truncate_old_tool_output(small.clone()), small);
    }

    #[test]
    fn truncate_respects_utf8_boundaries() {
        // Multi-byte chars around the cut points must not panic or split.
        let s = "𝔘".repeat(5_000); // 4 bytes each = 20_000 bytes
        let out = truncate_old_tool_output(s);
        assert!(out.contains("…[truncated"));
        // Round-trips as valid UTF-8 (String guarantees it; assert non-empty cut).
        assert!(out.len() < 20_000);
    }

    #[test]
    fn tool_output_for_age_truncates_only_old() {
        let big = "B".repeat(10_000);
        // idx < cutoff → truncated
        let old = tool_output_for_age(big.clone(), 0, 5);
        assert!(old.contains("…[truncated"));
        // idx >= cutoff → verbatim
        let fresh = tool_output_for_age(big.clone(), 5, 5);
        assert_eq!(fresh, big);
    }

    fn tool_msg(call_id: &str, text: &str) -> chat::MessageRequest {
        chat::MessageRequest::Tool(chat::ToolMessage {
            content: chat::MessageContent::Text(text.to_string()),
            tool_call_id: call_id.to_string(),
        })
    }

    fn sys_msg(text: &str) -> chat::MessageRequest {
        chat::MessageRequest::System(chat::SystemMessage {
            content: chat::MessageContent::Text(text.to_string()),
            name: None,
        })
    }

    fn user_msg(text: &str) -> chat::MessageRequest {
        chat::MessageRequest::User(chat::UserMessage {
            content: chat::UserContent::Text(text.to_string()),
            name: None,
        })
    }

    fn assistant_msg(text: &str) -> chat::MessageRequest {
        chat::MessageRequest::Assistant(chat::AssistantMessage {
            content: Some(chat::AssistantContent::Text(text.to_string())),
            name: None,
            refusal: None,
            audio: None,
            tool_calls: None,
            function_call: None,
            reasoning_content: None,
        })
    }

    #[test]
    fn message_budget_noop_when_within_limit() {
        let mut msgs = vec![sys_msg("s"), user_msg("u"), assistant_msg("a")];
        assert_eq!(enforce_message_budget(&mut msgs, 10), 0);
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn message_budget_zero_is_unlimited() {
        let mut msgs = vec![user_msg("u1"), user_msg("u2"), user_msg("u3")];
        assert_eq!(enforce_message_budget(&mut msgs, 0), 0);
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn message_budget_drops_oldest_turns_keeps_prefix() {
        // system prefix + 4 turns of (user, assistant). Limit 5 → keep prefix
        // (1) + last 2 turns (4) = 5.
        let mut msgs = vec![sys_msg("s")];
        for i in 0..4 {
            msgs.push(user_msg(&format!("u{i}")));
            msgs.push(assistant_msg(&format!("a{i}")));
        }
        assert_eq!(msgs.len(), 9);
        let dropped = enforce_message_budget(&mut msgs, 5);
        assert_eq!(dropped, 4);
        assert_eq!(msgs.len(), 5);
        // Prefix retained, first surviving turn starts at the u2 user message.
        assert!(matches!(msgs[0], chat::MessageRequest::System(_)));
        match &msgs[1] {
            chat::MessageRequest::User(u) => match &u.content {
                chat::UserContent::Text(t) => assert_eq!(t, "u2"),
                _ => panic!("expected text"),
            },
            _ => panic!("expected user message at turn boundary"),
        }
    }

    #[test]
    fn message_budget_never_leaves_orphan_tool_message() {
        // Turn: user, assistant(tool_call), tool. Cutting mid-turn would orphan
        // the tool message; trimmer must cut only at user boundaries.
        let mut msgs = vec![sys_msg("s")];
        for i in 0..3 {
            msgs.push(user_msg(&format!("u{i}")));
            msgs.push(assistant_msg(&format!("a{i}")));
            msgs.push(tool_msg(&format!("c{i}"), "out"));
        }
        // 1 + 3*3 = 10 messages. Limit 5.
        let dropped = enforce_message_budget(&mut msgs, 5);
        assert!(dropped > 0);
        // First message after prefix must be a User (turn start), never a Tool.
        assert!(matches!(msgs[0], chat::MessageRequest::System(_)));
        assert!(
            matches!(msgs[1], chat::MessageRequest::User(_)),
            "first non-prefix message must be a user turn start, not an orphan"
        );
    }

    #[test]
    fn message_budget_keeps_last_turn_when_single_turn_overflows() {
        // Prefix + one giant turn that alone exceeds the limit → keep it.
        let mut msgs = vec![sys_msg("s"), user_msg("u0")];
        for i in 0..5 {
            msgs.push(assistant_msg(&format!("a{i}")));
            msgs.push(tool_msg(&format!("c{i}"), "out"));
        }
        let len_before = msgs.len();
        // Only one user turn exists, so nothing can be dropped without breaking it.
        let dropped = enforce_message_budget(&mut msgs, 3);
        assert_eq!(dropped, 0);
        assert_eq!(msgs.len(), len_before);
    }

    #[test]
    fn enforce_budget_noop_when_within_limit() {
        let mut msgs = vec![tool_msg("c1", "small")];
        assert_eq!(enforce_input_budget(&mut msgs, 1_000_000), 0);
        match &msgs[0] {
            chat::MessageRequest::Tool(t) => match &t.content {
                chat::MessageContent::Text(s) => assert_eq!(s, "small"),
                _ => panic!("expected text content"),
            },
            _ => panic!("expected tool message"),
        }
    }

    #[test]
    fn enforce_budget_shrinks_oldest_first_and_preserves_pairing() {
        // Two big tool outputs; budget only fits one truncated + one full.
        let mut msgs = vec![
            tool_msg("call_old", &"O".repeat(20_000)),
            tool_msg("call_new", &"N".repeat(5_000)),
        ];
        let before = messages_total_chars(&msgs);
        let shrunk = enforce_input_budget(&mut msgs, before - 10_000);
        assert!(shrunk >= 1, "should have truncated at least one output");
        // tool_call_id pairing untouched.
        match &msgs[0] {
            chat::MessageRequest::Tool(t) => {
                assert_eq!(t.tool_call_id, "call_old");
                match &t.content {
                    chat::MessageContent::Text(s) => assert!(s.contains("…[truncated")),
                    _ => panic!("expected text"),
                }
            }
            _ => panic!("expected tool message"),
        }
        assert!(messages_total_chars(&msgs) < before);
    }
}
