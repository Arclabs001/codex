//! Request-only recovery for historical images after an HTTP payload rejection.
//! Original evidence is retained until a summarization request succeeds.

use std::collections::HashMap;

use crate::utils::json::serialized_json_bytes;
use codex_context_fragments::AnnotatedContent;
use codex_context_fragments::set_annotated_content;
use codex_context_fragments::to_annotated_content;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::ContentItemMetadata;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ImageReference;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseItem;
use http::StatusCode;

pub(crate) const OMITTED_IMAGE_NOTICE: &str = "[Historical image omitted from compaction input after HTTP 413. Use surrounding text and prior observations; the original remains in the thread history.]";

/// A handshake rejection proves that no output from this request was dispatched.
/// Stream errors with an HTTP status are intentionally excluded.
pub(crate) fn is_request_too_large(error: &CodexErr) -> bool {
    matches!(error.details(), CodexErrorDetails::RequestPayloadTooLarge(error)
        if error.status == StatusCode::PAYLOAD_TOO_LARGE)
}

/// A contiguous, identified active turn is required before older evidence can be omitted.
pub(crate) fn active_turn_start(items: &[ResponseItem], current_turn_id: &str) -> Option<usize> {
    active_turn_start_in_iter(items.iter(), current_turn_id)
}

pub(crate) fn active_turn_start_in_iter<'a>(
    items: impl Iterator<Item = &'a ResponseItem>,
    current_turn_id: &str,
) -> Option<usize> {
    if current_turn_id.is_empty() {
        return None;
    }
    let mut start = None;
    for (index, item) in items.enumerate() {
        if start.is_some()
            && item
                .turn_id()
                .is_some_and(|turn_id| turn_id != current_turn_id)
        {
            return None;
        }
        if start.is_none()
            && item.turn_id() == Some(current_turn_id)
            && crate::compact::is_real_user_or_agent_message(item)
        {
            start = Some(index);
        }
    }
    start
}

pub(crate) fn has_recoverable_images(items: &[ResponseItem], current_turn_id: &str) -> bool {
    !historical_images(items, current_turn_id).is_empty()
}

/// Reduces a rejected compaction request by at least 25% when eligible images permit it.
/// Older messages are processed first; all eligible images in a selected message are
/// replaced together so a message with many images is scanned and serialized once.
/// Only call on a request copy after an exact HTTP 413.
pub(crate) fn reduce_historical_images_for_retry(
    items: &mut [ResponseItem],
    current_turn_id: &str,
) -> serde_json::Result<bool> {
    let before = serialized_json_bytes(items)?;
    let target = before.saturating_sub(before / 4);
    let mut remaining = before;
    let candidates = historical_images(items, current_turn_id);
    let mut next = 0;
    while next < candidates.len() {
        let item_index = candidates[next].0;
        let start = next;
        while next < candidates.len() && candidates[next].0 == item_index {
            next += 1;
        }
        let item = &mut items[item_index];
        let original = item.clone();
        let item_before = serialized_json_bytes(&original)?;
        match item {
            ResponseItem::Message { .. } => {
                let Some(mut content) = to_annotated_content(item) else {
                    continue;
                };
                for &(_, content_index) in &candidates[start..next] {
                    content[content_index] = AnnotatedContent::text(
                        OMITTED_IMAGE_NOTICE,
                        ContentItemKind("images.omitted_for_compaction".to_string()),
                        ContentItemMetadata::harness(),
                    );
                }
                let _ = set_annotated_content(item, content);
            }
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                let Some(content) = output.content_items_mut() else {
                    continue;
                };
                for &(_, content_index) in &candidates[start..next] {
                    content[content_index] = FunctionCallOutputContentItem::InputText {
                        text: OMITTED_IMAGE_NOTICE.to_string(),
                    };
                }
            }
            _ => continue,
        }
        let item_after = serialized_json_bytes(item)?;
        if item_after < item_before {
            // JSON array separators do not change because record counts stay fixed.
            remaining = remaining.saturating_sub(item_before - item_after);
            if remaining <= target {
                break;
            }
        } else {
            *item = original;
        }
    }
    Ok(remaining < before)
}

fn historical_images(items: &[ResponseItem], current_turn_id: &str) -> Vec<(usize, usize)> {
    let Some(active_start) = active_turn_start(items, current_turn_id) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    let mut observed_turns = HashMap::new();
    for (index, item) in items[..active_start].iter().enumerate() {
        // Legacy providers omit phase. Commentary and partial answers are not proof that
        // the turn finished; a final or legacy assistant answer is required.
        if let ResponseItem::Message { role, phase, status, content, .. } = item
            && role == "assistant"
            && matches!(phase, None | Some(MessagePhase::FinalAnswer))
            && matches!(status.as_deref(), None | Some("completed"))
            && content.iter().any(|part| matches!(part, ContentItem::OutputText { text, .. } if !text.trim().is_empty()))
            && let Some(turn_id) = item.turn_id().filter(|turn_id| !turn_id.is_empty())
        {
            observed_turns.insert(turn_id, index);
        }
    }
    for (item_index, item) in items[..active_start].iter().enumerate() {
        let Some(turn_id) = item.turn_id().filter(|turn_id| !turn_id.is_empty()) else {
            continue;
        };
        if !observed_turns
            .get(turn_id)
            .is_some_and(|index| *index > item_index)
        {
            continue;
        }
        match item {
            ResponseItem::Message { content, .. } => {
                for (content_index, part) in content.iter().enumerate() {
                    if matches!(part, ContentItem::InputImage { image, .. } if large_inline_image(image, part))
                    {
                        candidates.push((item_index, content_index));
                    }
                }
            }
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                if let Some(content) = output.content_items() {
                    for (content_index, part) in content.iter().enumerate() {
                        if matches!(part, FunctionCallOutputContentItem::InputImage { image, .. } if large_inline_image(image, part))
                        {
                            candidates.push((item_index, content_index));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    candidates
}

fn large_inline_image(image: &ImageReference, part: &impl serde::Serialize) -> bool {
    matches!(image, ImageReference::Inline { image_url }
        if image_url.get(..11).is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:image/")))
        // Includes a conservative allowance for replacement attribution metadata.
        && serialized_json_bytes(part).is_ok_and(|bytes| bytes > OMITTED_IMAGE_NOTICE.len() + 512)
}

#[cfg(test)]
#[path = "request_payload_tests.rs"]
mod tests;
