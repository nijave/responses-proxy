//! Shared upstream Chat Completions API request builder.

use crate::config::{ResolvedProvider, RewriteConfig};
use crate::types::chat;
use crate::types::streaming::{StreamState, process_upstream_stream_data};
use futures::StreamExt;

/// Build a POST request to the upstream Chat Completions endpoint.
pub fn build_chat_request(
    client: &reqwest::Client,
    provider: &ResolvedProvider,
    body: &serde_json::Value,
) -> reqwest::RequestBuilder {
    let url = format!("{}/chat/completions", provider.base_url);
    client
        .post(&url)
        .timeout(provider.timeout)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header("Content-Type", "application/json")
        .json(body)
}

/// Build a POST request from a typed ChatRequest, applying `chat_out` rewrite if configured.
pub fn build_typed_chat_request(
    client: &reqwest::Client,
    provider: &ResolvedProvider,
    chat_req: &chat::Request,
) -> Result<reqwest::RequestBuilder, String> {
    if provider.rewrite.chat_out.is_empty() {
        let body = serde_json::to_value(chat_req).map_err(|e| e.to_string())?;
        return Ok(build_chat_request(client, provider, &body));
    }

    let mut body = serde_json::to_value(chat_req).map_err(|e| e.to_string())?;
    crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.chat_out)?;
    Ok(build_chat_request(client, provider, &body))
}

/// Drain an upstream Chat Completions **SSE** response into a [`StreamState`],
/// applying the `chat_in` rewrite. Client-facing events are discarded — only the
/// accumulated `accumulated_text`/`usage`/`created` in the returned state matter.
///
/// Used by the compaction path so the summary request can be sent with
/// `stream: true`: the origin emits its first byte quickly, keeping the
/// connection under Cloudflare's ~100s origin-timeout window even when the full
/// summary generation is slow (the buffered non-streaming call would 524).
pub(crate) async fn drain_chat_stream(
    response: reqwest::Response,
    chat_in: &RewriteConfig,
) -> Result<StreamState, String> {
    let mut ss = StreamState::new(String::new(), String::new(), String::new());
    let discard_out = RewriteConfig::default();
    let mut bytes = response.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        drain_sse_buffer(&mut buf, &mut ss, chat_in, &discard_out)?;
    }
    // Flush a trailing frame that arrived without a terminating blank line.
    let tail = buf.trim().to_string();
    if !tail.is_empty() {
        feed_sse_frame(&tail, &mut ss, chat_in, &discard_out)?;
    }
    Ok(ss)
}

/// Consume every complete `\n\n`-delimited SSE frame from `buf`, feeding each
/// into `ss`. Leaves any partial trailing frame in `buf`. Mirrors the client
/// relay loop in `handlers::responses::handle_streaming`.
fn drain_sse_buffer(
    buf: &mut String,
    ss: &mut StreamState,
    chat_in: &RewriteConfig,
    responses_out: &RewriteConfig,
) -> Result<(), String> {
    while let Some(pos) = buf.find("\n\n") {
        let frame = buf[..pos].trim().to_string();
        *buf = buf[pos + 2..].to_string();
        feed_sse_frame(&frame, ss, chat_in, responses_out)?;
    }
    Ok(())
}

/// Extract the `data:` payload from one SSE frame and accumulate it into `ss`.
/// Emitted Responses events are discarded — only the accumulator state is kept.
fn feed_sse_frame(
    frame: &str,
    ss: &mut StreamState,
    chat_in: &RewriteConfig,
    responses_out: &RewriteConfig,
) -> Result<(), String> {
    if let Some(data) = frame
        .lines()
        .find(|l| l.starts_with("data:"))
        .and_then(|l| l.strip_prefix("data:").map(|s| s.trim()))
    {
        process_upstream_stream_data(ss, data, chat_in, responses_out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> RewriteConfig {
        RewriteConfig::default()
    }

    #[test]
    fn drains_text_and_usage_from_sse() {
        let mut ss = StreamState::new(String::new(), String::new(), String::new());
        let ci = empty();
        let co = empty();
        let sse = concat!(
            "data: {\"id\":\"x\",\"model\":\"m\",\"created\":123,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"}}]}\n\n",
            "data: {\"id\":\"x\",\"model\":\"m\",\"created\":123,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"}}]}\n\n",
            "data: {\"id\":\"x\",\"model\":\"m\",\"created\":123,\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"x\",\"model\":\"m\",\"created\":123,\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut buf = sse.to_string();
        drain_sse_buffer(&mut buf, &mut ss, &ci, &co).unwrap();

        assert_eq!(ss.accumulated_text, "Hello world");
        let usage = ss.usage.expect("usage captured");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 13);
    }

    #[test]
    fn handles_frames_split_across_chunks() {
        let mut ss = StreamState::new(String::new(), String::new(), String::new());
        let ci = empty();
        let co = empty();
        // A single frame delivered in two buffer appends.
        let mut buf = String::from(
            "data: {\"id\":\"x\",\"model\":\"m\",\"created\":1,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par",
        );
        drain_sse_buffer(&mut buf, &mut ss, &ci, &co).unwrap();
        assert_eq!(ss.accumulated_text, "");
        buf.push_str("tial\"}}]}\n\n");
        drain_sse_buffer(&mut buf, &mut ss, &ci, &co).unwrap();
        assert_eq!(ss.accumulated_text, "partial");
    }
}
