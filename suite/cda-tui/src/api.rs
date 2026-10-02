//! HTTP/1.1 client for the local sidecar. The sidecar speaks plain HTTP, so
//! this crate does not pull in a TLS stack.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::catalog::Case;

#[derive(Clone, Debug, Deserialize)]
pub struct Health {
    pub ok: bool,
    pub backend: String,
    pub model_id: String,
    pub device: String,
    pub n_cases: usize,
    #[serde(default)]
    pub jev_configured: bool,
    #[serde(default)]
    pub jev_model: String,
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct GradeResponse {
    pub ok: bool,
    pub backend: String,
    pub source: String,
    pub model_id: String,
    pub device: String,
    pub family: String,
    #[serde(default)]
    pub case_id: Option<String>,
    pub latency_ms: f64,
    #[serde(default)]
    pub state: Value,
    pub answers: BTreeMap<String, Answer>,
    #[serde(default)]
    pub missing: Vec<String>,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub cost_usd: Option<f64>,
}

/// One case graded by Laya-CDA and, when the key is set, by Jev.
#[derive(Clone, Debug, Deserialize)]
pub struct CompareOutcome {
    pub laya: GradeResponse,
    #[serde(default)]
    pub jev: Option<GradeResponse>,
    #[serde(default)]
    pub jev_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct Answer {
    #[serde(rename = "type")]
    pub qtype: String,
    pub label: String,
    pub probabilities: BTreeMap<String, f64>,
    pub answer_confidence: f64,
    #[serde(default)]
    pub choice: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
struct SchemaBody {
    families: Vec<crate::catalog::Family>,
}

pub fn health(base: &str) -> Result<Health, String> {
    let value = get_json(base, "/v1/health", Duration::from_secs(3))?;
    serde_json::from_value(value).map_err(|err| format!("health payload: {err}"))
}

pub fn schema(base: &str) -> Result<Vec<crate::catalog::Family>, String> {
    let value = get_json(base, "/v1/schema", Duration::from_secs(3))?;
    let body: SchemaBody =
        serde_json::from_value(value).map_err(|err| format!("schema payload: {err}"))?;
    Ok(body.families)
}

pub fn grade(base: &str, case: &Case) -> Result<GradeResponse, String> {
    let value = post_json(
        base,
        "/v1/grade",
        &case_body(case),
        Duration::from_secs(180),
    )?;
    parse_grade(value)
}

/// Laya-CDA and Jev on the same case. A Jev error stays on the outcome so Laya still shows.
pub fn compare(base: &str, case: &Case) -> Result<CompareOutcome, String> {
    let value = post_json(
        base,
        "/v1/compare",
        &case_body(case),
        Duration::from_secs(90),
    )?;
    if value.get("ok").and_then(|flag| flag.as_bool()) == Some(false) {
        let message = value
            .get("error")
            .and_then(|err| err.as_str())
            .unwrap_or("compare failed");
        return Err(message.to_string());
    }
    serde_json::from_value(value).map_err(|err| format!("compare payload: {err}"))
}

pub fn grade_jev(base: &str, case: &Case) -> Result<GradeResponse, String> {
    let value = post_json(base, "/v1/jev", &case_body(case), Duration::from_secs(90))?;
    parse_grade(value)
}

fn case_body(case: &Case) -> Value {
    serde_json::json!({
        "case_id": case.case_id,
        "family": case.family,
        "fields": case.fields,
    })
}

fn parse_grade(value: Value) -> Result<GradeResponse, String> {
    if value.get("ok").and_then(|flag| flag.as_bool()) == Some(false) {
        let message = value
            .get("error")
            .and_then(|err| err.as_str())
            .unwrap_or("grade failed");
        return Err(message.to_string());
    }
    serde_json::from_value(value).map_err(|err| format!("grade payload: {err}"))
}

pub fn schemas_match(local: &[crate::catalog::Family], remote: &[crate::catalog::Family]) -> bool {
    fn view(
        families: &[crate::catalog::Family],
    ) -> Vec<(String, Vec<(String, String, Vec<String>)>)> {
        families
            .iter()
            .map(|family| {
                let questions = family
                    .questions
                    .iter()
                    .map(|question| {
                        (
                            question.id.clone(),
                            question.qtype.clone(),
                            question.options.clone(),
                        )
                    })
                    .collect();
                (family.id.clone(), questions)
            })
            .collect()
    }
    view(local) == view(remote)
}

fn get_json(base: &str, path: &str, timeout: Duration) -> Result<Value, String> {
    request(base, "GET", path, None, timeout)
}

fn post_json(base: &str, path: &str, body: &Value, timeout: Duration) -> Result<Value, String> {
    let bytes = serde_json::to_vec(body).map_err(|err| err.to_string())?;
    request(base, "POST", path, Some(&bytes), timeout)
}

fn request(
    base: &str,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<Value, String> {
    let (host, port) = parse_base(base)?;
    let address = format!("{host}:{port}");
    let socket = resolve(&address)?;
    let mut stream = TcpStream::connect_timeout(&socket, timeout)
        .map_err(|err| format!("connect {address}: {err}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|err| err.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|err| err.to_string())?;

    let mut header = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: cda-tui\r\nAccept: application/json\r\nConnection: close\r\n"
    );
    if let Some(bytes) = body {
        header.push_str("Content-Type: application/json\r\n");
        header.push_str(&format!("Content-Length: {}\r\n", bytes.len()));
    }
    header.push_str("\r\n");
    stream
        .write_all(header.as_bytes())
        .map_err(|err| format!("write: {err}"))?;
    if let Some(bytes) = body {
        stream
            .write_all(bytes)
            .map_err(|err| format!("write body: {err}"))?;
    }

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|err| format!("read: {err}"))?;
    let (status, payload) = split_http(&raw)?;
    let value: Value = serde_json::from_slice(payload).map_err(|err| {
        format!(
            "HTTP {status} was not JSON: {err}; body {}",
            String::from_utf8_lossy(payload)
        )
    })?;
    if status >= 400 {
        let message = value
            .get("error")
            .and_then(|err| err.as_str())
            .unwrap_or("request failed");
        return Err(format!("HTTP {status}: {message}"));
    }
    Ok(value)
}

fn resolve(address: &str) -> Result<std::net::SocketAddr, String> {
    if let Ok(socket) = address.parse() {
        return Ok(socket);
    }
    use std::net::ToSocketAddrs;
    address
        .to_socket_addrs()
        .map_err(|err| format!("resolve {address}: {err}"))?
        .next()
        .ok_or_else(|| format!("could not resolve {address}"))
}

fn parse_base(base: &str) -> Result<(String, u16), String> {
    let rest = base
        .trim()
        .trim_end_matches('/')
        .strip_prefix("http://")
        .ok_or("the sidecar URL must start with http://")?;
    if rest.contains('/') {
        return Err("pass the sidecar origin only, for example http://127.0.0.1:8765".into());
    }
    let (host, port) = rest
        .rsplit_once(':')
        .ok_or("the sidecar URL needs a port, for example http://127.0.0.1:8765")?;
    if host.is_empty() {
        return Err("missing host".into());
    }
    let port: u16 = port.parse().map_err(|_| format!("bad port {port}"))?;
    Ok((host.to_string(), port))
}

fn split_http(buf: &[u8]) -> Result<(u16, &[u8]), String> {
    let split_at = buf
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or("truncated HTTP response")?;
    let head = std::str::from_utf8(&buf[..split_at]).map_err(|err| err.to_string())?;
    let status_line = head.lines().next().unwrap_or("");
    let code = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|token| token.parse().ok())
        .ok_or_else(|| format!("bad status line: {status_line}"))?;
    Ok((code, &buf[split_at + 4..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn splits_a_content_bearing_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
        let (code, body) = split_http(raw).unwrap();
        assert_eq!(code, 200);
        assert_eq!(body, b"{\"ok\":true}");
    }

    #[test]
    fn roundtrips_json_over_a_local_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            loop {
                let n = sock.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let body = br#"{"ok":true,"backend":"mock"}"#;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            sock.write_all(header.as_bytes()).unwrap();
            sock.write_all(body).unwrap();
        });
        let value = get_json(
            &format!("http://127.0.0.1:{port}"),
            "/v1/health",
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(value["ok"], true);
        assert_eq!(value["backend"], "mock");
    }
}
