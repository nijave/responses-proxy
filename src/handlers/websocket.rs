//! WebSocket handler for the bidirectional Responses API.
//!
//! Submodules — one per event type:
//! - [`create`] — `response.create`
//! - [`cancel`] — `response.cancel`
//! - [`ping`]   — `ping` / `pong` keepalive

mod create;
mod ping;

use crate::types::websocket::ClientEvent;
use axum::extract::{
    State,
    ws::{Message as WsMsg, WebSocket, WebSocketUpgrade},
};
use axum::response::IntoResponse;

/// WebSocket upgrade handler for the bidirectional Responses API.
pub async fn websocket(
    State(state): State<crate::app::State>,
    headers: axum::http::HeaderMap,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Optional per-instance store namespace, taken from the upgrade request and
    // applied to every turn on this connection.
    let ns = super::namespace_from_headers(&headers);
    ws.on_upgrade(move |s| run(s, state, ns))
}

/// Send a text frame with debug logging.
pub(super) async fn send(socket: &mut WebSocket, text: &str) {
    tracing::debug!("WS send: {text}");
    let _ = socket.send(WsMsg::Text(text.into())).await;
}

// ── Main event loop ────────────────────────────────────────────────────────

/// Receive loop — dispatches incoming events to their handlers.
async fn run(mut socket: WebSocket, state: crate::app::State, ns: String) {
    tracing::info!("WebSocket connection established");

    while let Some(Ok(msg)) = socket.recv().await {
        let text = match msg {
            WsMsg::Text(t) => t.to_string(),
            WsMsg::Close(_) => {
                tracing::info!("WebSocket client sent close frame");
                break;
            }
            _ => continue,
        };

        tracing::debug!("WS recv: {text}");

        let event: ClientEvent = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!("WS invalid JSON: {e}");
                continue;
            }
        };

        match event {
            ClientEvent::ResponseCreate(req) => {
                tracing::info!("WS received event: response.create");
                create::handle(&state, &mut socket, req, &ns).await;
            }

            ClientEvent::ResponseCancel => {
                tracing::info!("WS received event: response.cancel");
            }

            ClientEvent::Ping => {
                ping::handle(&mut socket).await;
            }
        }
    }

    tracing::info!("WebSocket connection closed");
}
