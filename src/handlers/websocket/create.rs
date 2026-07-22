//! Handler for the `response.create` WebSocket event.

use crate::convert::responses_to_chat;
use crate::types::chat::{self, MessageRequest};
use crate::types::event;
use crate::types::event::StreamEvent;
use crate::types::responses::{Error, Request, Response, ResponseStatus};
use crate::types::streaming::*;
use crate::types::websocket;
use axum::extract::ws::{Message as WsMsg, WebSocket};
use futures::StreamExt;

/// Build a minimal Response for lifecycle events.  Full output / usage are populated later.
fn ws_response(rid: &str, model: &str, now: i64, status: ResponseStatus) -> Response {
    Response {
        id: rid.to_string(),
        model: model.to_string(),
        status,
        created_at: now,
        parallel_tool_calls: true,
        ..Default::default()
    }
}

/// Handle a `response.create` event: parse, forward to upstream, stream back results.
pub(super) async fn handle(state: &crate::app::State, socket: &mut WebSocket, mut req: Request) {
    tracing::debug!("input items {}", req.input.len());

    let provider = match state.config().models.get(&req.model) {
        Some(p) => p.clone(),
        None => {
            let ws_err = websocket::ErrorEvent::new(
                400,
                Error::TYPE_INVALID_REQUEST,
                "model_not_found",
                format!("Unknown model: {}", req.model),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };

    if !provider.rewrite.responses_in.is_empty() {
        let mut body = match serde_json::to_value(&req) {
            Ok(body) => body,
            Err(e) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    e.to_string(),
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        };
        if let Err(message) =
            crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.responses_in)
        {
            let ws_err = websocket::ErrorEvent::new(
                500,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                message,
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
        req = match serde_json::from_value(body) {
            Ok(req) => req,
            Err(e) => {
                let ws_err = websocket::ErrorEvent::new(
                    400,
                    Error::TYPE_INVALID_REQUEST,
                    Error::CODE_SERVER_ERROR,
                    e.to_string(),
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        };
    }

    // Codex remote-compaction-v2 short-circuit: when `input` carries a
    // `compaction_trigger`, run the summary turn through the shared compaction
    // helper and emit a single `compaction` output item instead of forwarding
    // to the chat upstream.
    if req
        .input
        .iter()
        .any(|i| matches!(i, crate::types::item::InputItem::CompactionTrigger(_)))
    {
        handle_compaction_trigger(state, &provider, socket, req).await;
        return;
    }

    let model = req.model.clone();
    let generate = req.generate;

    let rid = format!("resp_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let mid = format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));

    // gpt-5.6 code-mode: names Codex declared as custom tools (computed before
    // `req` is moved), so streamed function calls can be re-emitted as
    // custom_tool_call. Restored from the previous response on a continuation
    // turn. Empty for models below 5.6 → no behavior change.
    let custom_names = crate::convert::resolve_custom_tool_names(
        state,
        &req.input,
        req.previous_response_id.as_deref(),
    )
    .await;
    // Cache alongside the tools so the next continuation restores it too.
    let stored_custom_names = custom_names.clone();

    // Convert to Chat API (responses_to_chat handles history + instructions)
    let mut chat_req = match responses_to_chat(req, state).await {
        Ok(cr) => cr,
        Err(unsupported) => {
            let ws_err = websocket::ErrorEvent::new(
                400,
                Error::TYPE_INVALID_REQUEST,
                "unsupported_feature",
                format!("Unsupported features: {}", unsupported.join(", ")),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };
    chat_req.model = provider.model.clone();
    // Content-character size before truncation. When we truncate, the upstream's
    // real input_tokens is scaled up by full/sent so Codex's auto-compaction
    // threshold (keyed off server-reported total_tokens) fires on time instead
    // of being masked by our truncation.
    let full_chars = crate::handlers::input_tokens::content_chars(&chat_req.messages);
    let dropped =
        crate::convert::enforce_message_budget(&mut chat_req.messages, provider.max_input_messages);
    if dropped > 0 {
        tracing::info!(
            max_messages = provider.max_input_messages,
            dropped_messages = dropped,
            "Input exceeded history.max-input-messages — dropped oldest turns"
        );
    }
    let mut shrunk = 0;
    if let Some(max_chars) = provider.max_input_chars {
        shrunk = crate::convert::enforce_input_budget(&mut chat_req.messages, max_chars);
        if shrunk > 0 {
            tracing::info!(
                max_chars,
                shrunk_tool_outputs = shrunk,
                "Input exceeded history.max-input-chars — truncated old tool outputs"
            );
        }
    }
    let sent_chars = crate::handlers::input_tokens::content_chars(&chat_req.messages);
    let input_char_scale =
        (dropped > 0 || shrunk > 0).then_some((full_chars as u64, sent_chars as u64));
    let mut full_input_messages = chat_req.messages.clone();
    // Cache the code-mode tool registry so tool-result continuations (which
    // reference this response via `previous_response_id` but omit
    // `additional_tools`) can restore it instead of reaching the model with no
    // tools. Empty for models below 5.6 → no-op.
    let response_tools = chat_req.tools.clone().unwrap_or_default();

    // If generate=false, just echo lifecycle events without calling upstream
    if !generate {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let resp_in_progress = ws_response(&rid, &model, now, ResponseStatus::InProgress);
        let resp_completed = ws_response(&rid, &model, now, ResponseStatus::Completed);

        for event in [
            StreamEvent::Created(event::Created {
                response: resp_in_progress.clone(),
                sequence_number: 0,
            }),
            StreamEvent::InProgress(event::InProgress {
                response: resp_in_progress,
                sequence_number: 1,
            }),
            StreamEvent::Completed(event::Completed {
                response: resp_completed,
                sequence_number: 2,
            }),
        ] {
            match prepare_stream_event(event, &provider.rewrite.responses_out) {
                Ok(prepared) => {
                    super::send(socket, &prepared.body.to_string()).await;
                }
                Err(message) => {
                    let ws_err = websocket::ErrorEvent::new(
                        500,
                        Error::TYPE_SERVER_ERROR,
                        Error::CODE_SERVER_ERROR,
                        message,
                    );
                    super::send(socket, &ws_err.to_json_string()).await;
                    return;
                }
            }
        }

        state
            .store()
            .put_tools(rid.clone(), response_tools, stored_custom_names)
            .await;
        state.store().put(rid, full_input_messages).await;
        return;
    }

    let url = format!("{}/chat/completions", provider.base_url);
    let request = state
        .http_client()
        .post(&url)
        .timeout(provider.timeout)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header("Content-Type", "application/json");
    let request = if provider.rewrite.chat_out.is_empty() {
        tracing::debug!(
            "chat request: {}",
            serde_json::to_string(&chat_req).unwrap_or("".to_string())
        );
        request.json(&chat_req)
    } else {
        let mut body = match serde_json::to_value(&chat_req) {
            Ok(body) => body,
            Err(e) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    e.to_string(),
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        };
        if let Err(message) = crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.chat_out) {
            let ws_err = websocket::ErrorEvent::new(
                500,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                message,
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
        tracing::debug!(
            "chat request: {}",
            serde_json::to_string(&body).unwrap_or("".to_string())
        );
        request.json(&body)
    };
    let stream_resp = match request.send().await {
        Ok(r) => r,
        Err(e) => {
            let ws_err = websocket::ErrorEvent::new(
                502,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                format!("Upstream error: {e}"),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };

    if !stream_resp.status().is_success() {
        let status_code = stream_resp.status().as_u16();
        let ws_err = websocket::ErrorEvent::new(
            status_code,
            Error::TYPE_SERVER_ERROR,
            Error::CODE_SERVER_ERROR,
            format!(
                "Upstream error:  {}",
                stream_resp.text().await.unwrap_or("".into())
            ),
        );
        super::send(socket, &ws_err.to_json_string()).await;
        return;
    }

    // Send lifecycle start events (typed, with sequence_number)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let resp = ws_response(&rid, &model, now, ResponseStatus::InProgress);
    let mut initial_events = Vec::new();
    for event in [
        StreamEvent::Created(event::Created {
            response: resp.clone(),
            sequence_number: 0,
        }),
        StreamEvent::InProgress(event::InProgress {
            response: resp,
            sequence_number: 1,
        }),
    ] {
        match prepare_stream_event(event, &provider.rewrite.responses_out) {
            Ok(prepared) => {
                initial_events.push(prepared.event);
                super::send(socket, &prepared.body.to_string()).await;
            }
            Err(message) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    message,
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        }
    }

    // Register cancellation token for HTTP cancel endpoint
    let cancel_rx = state.store().register_cancel_token(&rid).await;

    // Stream loop: read SSE chunks + handle cancel
    let stream_context = WsStreamContext {
        rid: &rid,
        mid: &mid,
        model: &model,
        chat_in: &provider.rewrite.chat_in,
        responses_out: &provider.rewrite.responses_out,
        now,
        compact_key: state.compact_key(),
        custom_tool_names: custom_names,
        input_char_scale,
    };
    let (response_msg, cancelled, stream_events) =
        run_stream(socket, stream_resp, stream_context, cancel_rx).await;
    let events = initial_events;
    // stream_events are already sent, just used for counting
    let total_events = events.len() + stream_events.len();

    // Clean up cancel token (run_stream already handled the actual cancellation check)
    state.store().unregister_cancel_token(&rid).await;

    // Persist accumulated history so `previous_response_id` chains resolve.
    // Codex (store:false) replays full history on new user turns but sends
    // tool-result *deltas* referencing previous_response_id after a
    // function_call — without the stored assistant tool_calls message those
    // deltas would orphan the tool result and the upstream would 400. The
    // client `store` flag governs client-side GET retrieval, not this internal
    // chaining, so persist regardless of it.
    if !cancelled {
        // Append assistant response to input messages and store
        let assistant_msg: MessageRequest = response_msg.into();
        let has_reasoning =
            matches!(&assistant_msg, MessageRequest::Assistant(a) if a.reasoning_content.is_some());
        tracing::info!(
            total_events,
            has_reasoning,
            msg_count = full_input_messages.len() + 1,
            "WS: storing history"
        );
        full_input_messages.push(assistant_msg);
        state
            .store()
            .put_tools(rid.clone(), response_tools, stored_custom_names)
            .await;
        state.store().put(rid, full_input_messages).await;
    }
}

/// Relay SSE chunks from upstream to WebSocket, with cancel detection.
/// Returns (response_message, cancelled, collected_events).
struct WsStreamContext<'a> {
    rid: &'a str,
    mid: &'a str,
    model: &'a str,
    chat_in: &'a crate::config::RewriteConfig,
    responses_out: &'a crate::config::RewriteConfig,
    now: i64,
    compact_key: Option<&'a [u8; 32]>,
    custom_tool_names: std::collections::HashSet<String>,
    input_char_scale: Option<(u64, u64)>,
}

async fn run_stream(
    socket: &mut WebSocket,
    stream_resp: reqwest::Response,
    context: WsStreamContext<'_>,
    mut cancel_rx: tokio::sync::watch::Receiver<bool>,
) -> (chat::ResponseMessage, bool, Vec<StreamEvent>) {
    let mut buf = String::new();
    let mut ss = StreamState::new(
        context.rid.to_string(),
        context.mid.to_string(),
        context.model.to_string(),
    );
    ss.has_started = true;
    ss.created = context.now;
    ss.compact_key = context.compact_key.copied();
    ss.custom_tool_names = context.custom_tool_names;
    ss.input_char_scale = context.input_char_scale;
    let mut byte_stream = stream_resp.bytes_stream();
    let mut cancelled = false;
    let mut collected_events: Vec<StreamEvent> = Vec::new();

    loop {
        tokio::select! {
            _ = cancel_rx.changed() => {
                tracing::info!(response_id = %context.rid, "WS stream cancelled via HTTP");
                cancelled = true;
                break;
            }
            chunk = byte_stream.next() => {
                match chunk {
                    Some(Ok(b)) => {
                        buf.push_str(&String::from_utf8_lossy(&b));
                        while let Some(pos) = buf.find("\n\n") {
                            let ev = buf[..pos].trim().to_string();
                            buf = buf[pos + 2..].to_string();
                            if let Some(data) = ev.lines()
                                .find(|l| l.starts_with("data:"))
                                .and_then(|l| l.strip_prefix("data:").map(|s| s.trim()))
                            {
                                tracing::trace!(%data, "Chat API delta");
                                match process_upstream_stream_data(
                                    &mut ss,
                                    data,
                                    context.chat_in,
                                    context.responses_out,
                                ) {
                                    Ok(events) => {
                                        for prepared in events {
                                            if !prepared.event_type.ends_with("delta") {
                                                tracing::info!("WS event: {}", prepared.event_type);
                                                tracing::debug!("WS event details: {}", prepared.body);
                                            }
                                            let msg = prepared.body.to_string();
                                            tracing::debug!("WS send: {msg}");
                                            collected_events.push(prepared.event);
                                            if socket.send(WsMsg::Text(msg.into())).await.is_err() {
                                                tracing::info!("WS send failed");
                                                return (ss.to_response_message(), false, collected_events);
                                            }
                                        }
                                    }
                                    Err(message) => {
                                        let ws_err = websocket::ErrorEvent::new(
                                            500,
                                            Error::TYPE_SERVER_ERROR,
                                            Error::CODE_SERVER_ERROR,
                                            message,
                                        );
                                        let msg = ws_err.to_json_string();
                                        tracing::debug!("WS send: {msg}");
                                        let _ = socket.send(WsMsg::Text(msg.into())).await;
                                        return (ss.to_response_message(), false, collected_events);
                                    }
                                }
                            }
                        }
                    }
                    _ => break,
                }
            }
            ws_msg = socket.recv() => {
                match ws_msg {
                    Some(Ok(WsMsg::Text(t)))
                        if t.trim() == r#"{"type":"response.cancel"}"# =>
                    {
                        tracing::info!("WS cancel received during streaming");
                        cancelled = true;
                        break;
                    }
                    _ => break,
                }
            }
        }
    }
    (ss.to_response_message(), cancelled, collected_events)
}

/// Run the shared compaction helper and emit a four-event lifecycle on the
/// WebSocket: `response.created` → `response.in_progress` →
/// `response.output_item.done` (carrying the single `compaction` item) →
/// `response.completed`. Mirrors the SSE flow in handlers::responses.
async fn handle_compaction_trigger(
    state: &crate::app::State,
    provider: &crate::config::ResolvedProvider,
    socket: &mut WebSocket,
    req: Request,
) {
    // Codex replays the full history in `input` alongside the trigger
    // (store:false), so the request itself is the summary source.
    let current_input: Vec<crate::types::item::InputItem> = req
        .input
        .iter()
        .filter(|i| !matches!(i, crate::types::item::InputItem::CompactionTrigger(_)))
        .cloned()
        .collect();
    let current_messages = crate::convert::items_to_chat_messages(&current_input, state);
    let (output, usage, created_at) = match crate::handlers::compact::build_compaction_output(
        state,
        provider,
        req.previous_response_id.as_deref(),
        current_messages,
    )
    .await
    {
        Ok(triple) => triple,
        Err((status, body)) => {
            let message = body
                .0
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("compaction failed")
                .to_string();
            let ws_err = websocket::ErrorEvent::new(
                status.as_u16(),
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                message,
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };

    let rid = format!("resp_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let mut resp = Response {
        id: rid.clone(),
        model: req.model.clone(),
        status: ResponseStatus::Completed,
        created_at,
        output,
        usage: Some(usage),
        ..Default::default()
    };
    if let Some(ref meta) = req.metadata {
        resp.metadata = Some(meta.clone());
    }
    resp.previous_response_id = req.previous_response_id.clone();
    resp.parallel_tool_calls = req.parallel_tool_calls;

    let item = match resp.output.first().cloned() {
        Some(item) => item,
        None => {
            let ws_err = websocket::ErrorEvent::new(
                500,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                "compaction helper returned no output".into(),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };

    // Compute persisted summary before `resp` is moved into the Completed event.
    // Persist regardless of req.store so a later previous_response_id resolves.
    let store_messages = Some(crate::handlers::compaction_output_to_chat_messages(
        &resp.output,
        state,
    ));

    let lifecycle = Response {
        status: ResponseStatus::InProgress,
        output: vec![],
        usage: None,
        ..resp.clone()
    };
    let events = [
        StreamEvent::Created(event::Created {
            response: lifecycle.clone(),
            sequence_number: 0,
        }),
        StreamEvent::InProgress(event::InProgress {
            response: lifecycle,
            sequence_number: 1,
        }),
        StreamEvent::OutputItemDone(event::OutputItemDone {
            item,
            output_index: 0,
            sequence_number: 2,
        }),
        StreamEvent::Completed(event::Completed {
            response: resp,
            sequence_number: 3,
        }),
    ];
    for event in events {
        match prepare_stream_event(event, &provider.rewrite.responses_out) {
            Ok(prepared) => super::send(socket, &prepared.body.to_string()).await,
            Err(message) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    message,
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        }
    }

    if let Some(store_messages) = store_messages {
        state.store().put(rid, store_messages).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_response_builds_correctly() {
        let resp = ws_response("resp_1", "gpt-5", 1000, ResponseStatus::InProgress);
        assert_eq!(resp.id, "resp_1");
        assert_eq!(resp.model, "gpt-5");
        assert_eq!(resp.created_at, 1000);
        assert_eq!(resp.status, ResponseStatus::InProgress);
        assert!(resp.parallel_tool_calls);
        assert!(resp.output.is_empty());
    }

    #[test]
    fn ws_response_completed_status() {
        let resp = ws_response("resp_x", "claude", 500, ResponseStatus::Completed);
        assert_eq!(resp.status, ResponseStatus::Completed);
    }

    #[test]
    fn ws_response_all_statuses() {
        let statuses = &[
            ResponseStatus::Queued,
            ResponseStatus::InProgress,
            ResponseStatus::Completed,
            ResponseStatus::Failed,
            ResponseStatus::Incomplete,
            ResponseStatus::Cancelled,
        ];
        for st in statuses {
            let resp = ws_response("r", "m", 0, *st);
            assert_eq!(resp.id, "r");
        }
    }
}
