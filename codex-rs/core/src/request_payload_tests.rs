use super::*;
use codex_protocol::error::UnexpectedResponseError;
use codex_protocol::models::FunctionCallOutputPayload;
use serde_json::json;

fn item(value: serde_json::Value, turn: Option<&str>) -> ResponseItem {
    let mut item: ResponseItem = serde_json::from_value(value).unwrap();
    if let Some(turn) = turn {
        item.set_turn_id_if_missing(turn);
    }
    item
}

fn image_message(turn: Option<&str>, image: &str) -> ResponseItem {
    item(
        json!({
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Read this chart"},
                        {"type": "input_image", "image_url": image}]
        }),
        turn,
    )
}

fn answer(turn: &str, phase: Option<&str>) -> ResponseItem {
    let mut value = json!({
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "I read the chart."}]
    });
    if let Some(phase) = phase {
        value["phase"] = phase.into();
    }
    item(value, Some(turn))
}

fn data_url() -> String {
    format!("data:image/png;base64,{}", "A".repeat(4096))
}

fn http_error() -> UnexpectedResponseError {
    UnexpectedResponseError {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        body: "too big".to_string(),
        user_message: None,
        url: None,
        cf_ray: None,
        request_id: None,
        identity_authorization_error: None,
        identity_error_code: None,
    }
}

#[test]
fn only_typed_http_handshake_rejection_can_trigger_recovery() {
    assert!(is_request_too_large(&CodexErr::new(
        CodexErrorDetails::RequestPayloadTooLarge(http_error())
    )));
    assert!(!is_request_too_large(&CodexErr::UnexpectedStatus(
        http_error()
    )));
    assert!(!is_request_too_large(&CodexErr::Stream(
        "HTTP 413 payload_too_large".to_string()
    )));
    assert!(!is_request_too_large(&CodexErr::ContextWindowExceeded));
}

#[test]
fn omission_is_request_only_and_strictly_smaller_with_text_and_ids_preserved() {
    let mut original = vec![
        image_message(Some("old"), &data_url()),
        answer("old", None),
        image_message(Some("current"), &data_url()),
    ];
    original[0].set_id(Some(codex_protocol::ResponseItemId::with_suffix(
        "msg",
        "historical",
    )));
    let mut projected = original.clone();
    let before = serialized_json_bytes(&projected).unwrap();
    assert!(has_recoverable_images(&projected, "current"));
    assert!(reduce_historical_images_for_retry(&mut projected, "current").unwrap());
    assert!(serialized_json_bytes(&projected).unwrap() < before);
    assert_eq!(projected[0].id(), original[0].id());
    assert_eq!(projected[2], original[2]);
    assert_eq!(projected.len(), original.len());
    let value = serde_json::to_value(&projected[0]).unwrap();
    assert_eq!(value["content"][0]["text"], "Read this chart");
    assert_eq!(value["content"][1]["text"], OMITTED_IMAGE_NOTICE);
    assert!(
        matches!(&original[0], ResponseItem::Message {content,..} if matches!(&content[1], ContentItem::InputImage{..}))
    );
    assert!(!reduce_historical_images_for_retry(&mut projected, "current").unwrap());
}

#[test]
fn batch_reduction_processes_oldest_message_once_and_stops_at_target() {
    let mut oldest = image_message(Some("oldest"), &data_url());
    if let ResponseItem::Message { content, .. } = &mut oldest {
        let image = content[1].clone();
        content.extend([image.clone(), image]);
    }
    let next = image_message(Some("next"), &data_url());
    let current = image_message(Some("current"), &data_url());
    let mut input = vec![
        oldest,
        answer("oldest", None),
        next.clone(),
        answer("next", None),
        current.clone(),
    ];
    let before = serialized_json_bytes(&input).unwrap();
    assert!(reduce_historical_images_for_retry(&mut input, "current").unwrap());
    assert!(serialized_json_bytes(&input).unwrap() <= before - before / 4);
    let value = serde_json::to_value(&input[0]).unwrap();
    assert_eq!(value["content"].as_array().unwrap().len(), 4);
    for index in 1..4 {
        assert_eq!(value["content"][index]["text"], OMITTED_IMAGE_NOTICE);
    }
    assert_eq!(input[2], next);
    assert_eq!(input[4], current);
}

#[test]
fn unknown_current_unobserved_and_commentary_images_are_protected() {
    for (turn, phase) in [
        (None, None),
        (Some("current"), None),
        (Some("old"), Some("commentary")),
        (Some("old"), Some("partial_answer")),
    ] {
        let mut input = vec![
            image_message(turn, &data_url()),
            answer("old", phase),
            image_message(Some("current"), &data_url()),
        ];
        let original = input.clone();
        assert!(!reduce_historical_images_for_retry(&mut input, "current").unwrap());
        assert_eq!(input, original);
    }
    let mut unobserved = vec![
        image_message(Some("old"), &data_url()),
        image_message(Some("current"), &data_url()),
    ];
    assert!(!reduce_historical_images_for_retry(&mut unobserved, "current").unwrap());
}

#[test]
fn missing_or_noncontiguous_active_turn_is_protected() {
    let input = vec![image_message(Some("old"), &data_url()), answer("old", None)];
    assert!(!has_recoverable_images(&input, "current"));
    let mut input = vec![
        image_message(Some("old"), &data_url()),
        answer("old", None),
        image_message(Some("current"), &data_url()),
        answer("other", None),
    ];
    assert!(!reduce_historical_images_for_retry(&mut input, "current").unwrap());
}

#[test]
fn incomplete_or_empty_assistant_output_is_not_observation_proof() {
    for status in ["in_progress", "incomplete"] {
        let mut finished = answer("old", None);
        if let ResponseItem::Message {
            status: recorded, ..
        } = &mut finished
        {
            *recorded = Some(status.to_string());
        }
        let input = vec![
            image_message(Some("old"), &data_url()),
            finished,
            image_message(Some("current"), &data_url()),
        ];
        assert!(!has_recoverable_images(&input, "current"));
    }
    let input = vec![
        image_message(Some("old"), &data_url()),
        item(
            json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":" "}]}),
            Some("old"),
        ),
        image_message(Some("current"), &data_url()),
    ];
    assert!(!has_recoverable_images(&input, "current"));
}

#[test]
fn tool_image_omission_preserves_call_output_and_other_parts() {
    let call = item(
        json!({"type":"function_call", "call_id":"call-1","name":"view_image","arguments":"{}"}),
        Some("old"),
    );
    let mut output = item(
        json!({"type":"function_call_output","call_id":"call-1","output":"temporary"}),
        Some("old"),
    );
    if let ResponseItem::FunctionCallOutput { output, .. } = &mut output {
        *output = FunctionCallOutputPayload::from_content_items(vec![
            FunctionCallOutputContentItem::InputText {
                text: "Image from tool".to_string(),
            },
            FunctionCallOutputContentItem::InputImage {
                image: ImageReference::Inline {
                    image_url: data_url(),
                },
                detail: None,
            },
        ]);
    }
    let mut input = vec![
        call.clone(),
        output,
        answer("old", Some("final_answer")),
        image_message(Some("current"), &data_url()),
    ];
    assert!(reduce_historical_images_for_retry(&mut input, "current").unwrap());
    assert_eq!(input[0], call);
    let value = serde_json::to_value(&input[1]).unwrap();
    assert_eq!(value["call_id"], "call-1");
    assert_eq!(value["output"][0]["text"], "Image from tool");
    assert_eq!(value["output"][1]["text"], OMITTED_IMAGE_NOTICE);
}

#[test]
fn file_remote_and_small_images_are_not_changed() {
    for url in [
        "https://example.com/image.png",
        "data:image/png;base64,AAAA",
    ] {
        let mut input = vec![
            image_message(Some("old"), url),
            answer("old", None),
            image_message(Some("current"), &data_url()),
        ];
        assert!(!reduce_historical_images_for_retry(&mut input, "current").unwrap());
    }
    let mut input = vec![
        item(
            json!({"type":"message","role":"user","content":[{"type":"input_image","file_id":"file-original"}]}),
            Some("old"),
        ),
        answer("old", None),
        image_message(Some("current"), &data_url()),
    ];
    assert!(!reduce_historical_images_for_retry(&mut input, "current").unwrap());
}

#[test]
fn omission_preserves_neighboring_text_attribution() {
    let mut historical = image_message(Some("old"), &data_url());
    let mut parts = to_annotated_content(&mut historical).unwrap();
    parts[0] = AnnotatedContent::text(
        "Read this chart",
        ContentItemKind("user.input".to_string()),
        ContentItemMetadata::user(),
    );
    set_annotated_content(&mut historical, parts);
    let mut input = vec![
        historical,
        answer("old", None),
        image_message(Some("current"), &data_url()),
    ];
    assert!(reduce_historical_images_for_retry(&mut input, "current").unwrap());
    let parts = to_annotated_content(&mut input[0]).unwrap();
    let (_, kind, metadata) = parts[0].clone().into_parts();
    assert_eq!(kind, ContentItemKind("user.input".to_string()));
    assert_eq!(metadata, ContentItemMetadata::user());
    let (_, kind, metadata) = parts[1].clone().into_parts();
    assert_eq!(
        kind,
        ContentItemKind("images.omitted_for_compaction".to_string())
    );
    assert_eq!(metadata, ContentItemMetadata::harness());
}
