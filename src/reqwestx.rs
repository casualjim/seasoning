use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use std::time::{Duration, Instant};

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderValue};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::time::sleep;
use tracing::debug;

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct ApiClientConfig {
    pub base_url: String,
    pub api_key: Option<SecretString>,
    pub max_concurrent_requests: usize,
    pub max_requests_per_minute: usize,
    pub max_tokens_per_minute: usize,
    pub max_retries: usize,
    pub timeout: Duration,
}

impl Default for ApiClientConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: None,
            max_concurrent_requests: 300,
            max_requests_per_minute: 1000,
            max_tokens_per_minute: 1_000_000,
            max_retries: 3,
            timeout: Duration::from_secs(90),
        }
    }
}

#[derive(Debug)]
struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self, tokens_needed: f64) -> std::result::Result<(), Duration> {
        self.refill();

        if self.tokens >= tokens_needed {
            self.tokens -= tokens_needed;
            Ok(())
        } else {
            let tokens_short = tokens_needed - self.tokens;
            let wait_seconds = tokens_short / self.refill_rate;
            Err(Duration::from_secs_f64(wait_seconds))
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }
}

#[derive(Debug)]
struct RateLimiter {
    request_bucket: TokenBucket,
    token_bucket: TokenBucket,
}

#[derive(Clone)]
pub struct ApiClient {
    client: Client,
    config: ApiClientConfig,
    concurrent_semaphore: Arc<Semaphore>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl ApiClient {
    pub fn new(config: ApiClientConfig) -> Result<Self> {
        let user_agent = format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        let client = Client::builder()
            .default_headers({
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                headers
            })
            .user_agent(user_agent)
            .timeout(config.timeout)
            .build()
            .map_err(Error::HttpClientBuild)?;

        let concurrent_semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        let request_rate = config.max_requests_per_minute as f64 / 60.0;
        let token_rate = config.max_tokens_per_minute as f64 / 60.0;

        let request_capacity = (request_rate * 10.0).min(config.max_requests_per_minute as f64);
        let token_capacity = (token_rate * 10.0).min(config.max_tokens_per_minute as f64);

        let rate_limiter = Arc::new(Mutex::new(RateLimiter {
            request_bucket: TokenBucket::new(request_capacity, request_rate),
            token_bucket: TokenBucket::new(token_capacity, token_rate),
        }));

        Ok(Self {
            client,
            config,
            concurrent_semaphore,
            rate_limiter,
        })
    }

    pub async fn post_json<Req, Res>(
        &self,
        path: &str,
        payload: &Req,
        token_count: u32,
    ) -> Result<Res>
    where
        Req: Serialize,
        Res: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.config.base_url, path);
        let body_bytes = serde_json::to_vec(payload).map_err(Error::JsonEncode)?;

        let mut retries = 0;
        loop {
            debug!("Attempting request to {} (attempt {})", url, retries + 1);

            self.wait_for_rate_limit(token_count).await?;

            let request_permit = self
                .concurrent_semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| Error::SemaphoreClosed)?;

            let mut request = self.client.post(&url).body(body_bytes.clone());

            if let Some(api_key) = &self.config.api_key {
                let auth_value =
                    HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                        .map_err(Error::InvalidApiKeyHeaderValue)?;
                request = request.header(AUTHORIZATION, auth_value);
            }

            let result = request.send().await;

            match result {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        let result = read_json_with_permit(response, request_permit).await?;
                        debug!("Request to {} succeeded with status {}", url, status);
                        return Ok(result);
                    } else if should_retry(status.as_u16()) && retries < self.config.max_retries {
                        drop(request_permit);

                        retries += 1;
                        let backoff = calculate_backoff(retries);
                        debug!(
                            "Retrying after {} seconds due to status {}",
                            backoff.as_secs(),
                            status
                        );
                        sleep(backoff).await;
                        continue;
                    } else {
                        let body = read_text_with_permit(response, request_permit).await.ok();
                        return Err(Error::ApiStatus { status, body });
                    }
                }
                Err(e) => {
                    drop(request_permit);

                    if is_retryable_error(&e) && retries < self.config.max_retries {
                        retries += 1;
                        let backoff = calculate_backoff(retries);
                        debug!(
                            "Retrying after {} seconds due to error: {}",
                            backoff.as_secs(),
                            e
                        );
                        sleep(backoff).await;
                        continue;
                    } else {
                        return Err(Error::Request(e));
                    }
                }
            }
        }
    }

    async fn wait_for_rate_limit(&self, token_count: u32) -> Result<()> {
        loop {
            let wait_duration = {
                let mut limiter = self.rate_limiter.lock().await;

                let request_result = limiter.request_bucket.try_consume(1.0);
                let token_result = limiter.token_bucket.try_consume(token_count as f64);

                match (request_result, token_result) {
                    (Ok(()), Ok(())) => {
                        return Ok(());
                    }
                    (Ok(()), Err(token_wait)) => {
                        limiter.request_bucket.tokens = (limiter.request_bucket.tokens + 1.0)
                            .min(limiter.request_bucket.capacity);
                        token_wait
                    }
                    (Err(request_wait), Ok(())) => {
                        limiter.token_bucket.tokens = (limiter.token_bucket.tokens
                            + token_count as f64)
                            .min(limiter.token_bucket.capacity);
                        request_wait
                    }
                    (Err(request_wait), Err(token_wait)) => request_wait.max(token_wait),
                }
            };

            let wait_with_buffer = wait_duration + rate_limit_jitter();

            if wait_with_buffer > Duration::from_millis(100) {
                debug!(
                    "Rate limit: waiting {:?} before next request",
                    wait_with_buffer
                );
            }

            sleep(wait_with_buffer).await;
        }
    }
}

fn rate_limit_jitter() -> Duration {
    // Add a small random-ish buffer to avoid synchronization / thundering herds.
    // This is intentionally not cryptographically secure; it's only jitter.
    const MIN_MS: u64 = 5;
    const MAX_MS: u64 = 25;
    let range_ms = (MAX_MS - MIN_MS) + 1;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let jitter_ms = nanos % range_ms;

    Duration::from_millis(MIN_MS + jitter_ms)
}

async fn read_json_with_permit<Res>(
    response: reqwest::Response,
    request_permit: OwnedSemaphorePermit,
) -> Result<Res>
where
    Res: for<'de> Deserialize<'de>,
{
    let permit = request_permit;
    let result = response.json::<Res>().await.map_err(Error::ResponseJson);
    drop(permit);
    result
}

async fn read_text_with_permit(
    response: reqwest::Response,
    request_permit: OwnedSemaphorePermit,
) -> std::result::Result<String, reqwest::Error> {
    let permit = request_permit;
    let result = response.text().await;
    drop(permit);
    result
}

fn should_retry(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

fn is_retryable_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect()
}

fn calculate_backoff(retry_count: usize) -> Duration {
    #[cfg(test)]
    {
        let base = 2u64;
        let millis = base.pow(retry_count as u32).min(60) * 10;
        let backoff = Duration::from_millis(millis);
        let max_jitter = backoff.checked_div(2).unwrap_or(Duration::ZERO);
        backoff + backoff_jitter(Duration::ZERO, max_jitter.min(Duration::from_secs(5)))
    }

    #[cfg(not(test))]
    {
        let base = 2u64;
        let seconds = base.pow(retry_count as u32).min(60);
        let backoff = Duration::from_secs(seconds);
        let max_jitter = backoff.checked_div(2).unwrap_or(Duration::ZERO);
        backoff + backoff_jitter(Duration::ZERO, max_jitter.min(Duration::from_secs(5)))
    }
}

fn backoff_jitter(min_jitter: Duration, max_jitter: Duration) -> Duration {
    // Add jitter to avoid synchronized retries. Keep it bounded so we don't
    // unexpectedly amplify backoff by a large factor.
    let range = max_jitter.checked_sub(min_jitter).unwrap_or(Duration::ZERO);
    if range.is_zero() {
        return min_jitter;
    }

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let jitter_ns = seed % (range.as_nanos() + 1);
    let jitter = duration_from_nanos_u128(jitter_ns);
    min_jitter.checked_add(jitter).unwrap_or(max_jitter)
}

fn duration_from_nanos_u128(nanos: u128) -> Duration {
    const NS_PER_SEC: u128 = 1_000_000_000;
    let secs = nanos / NS_PER_SEC;
    let subsec = (nanos % NS_PER_SEC) as u32;
    if secs > u64::MAX as u128 {
        return Duration::MAX;
    }
    Duration::new(secs as u64, subsec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Serialize)]
    struct TestRequest {
        message: String,
    }

    #[derive(Deserialize, PartialEq, Debug)]
    struct TestResponse {
        result: String,
    }

    #[tokio::test]
    async fn test_successful_request() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "result": "success"
            })))
            .mount(&mock_server)
            .await;

        let config = ApiClientConfig {
            base_url: mock_server.uri(),
            max_requests_per_minute: 100,
            max_tokens_per_minute: 10000,
            ..Default::default()
        };

        let client = ApiClient::new(config).unwrap();
        let request = TestRequest {
            message: "test".to_string(),
        };

        let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
        assert_eq!(response.result, "success");
    }

    #[tokio::test]
    async fn test_retry_on_server_error() {
        let mock_server = MockServer::start().await;

        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let c = counter.clone();

        Mock::given(method("POST"))
            .and(path("/test"))
            .respond_with(move |_: &wiremock::Request| {
                let count = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 0 {
                    ResponseTemplate::new(503)
                } else {
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "result": "retry_success"
                    }))
                }
            })
            .mount(&mock_server)
            .await;

        let config = ApiClientConfig {
            base_url: mock_server.uri(),
            max_retries: 3,
            max_requests_per_minute: 100,
            max_tokens_per_minute: 10000,
            ..Default::default()
        };

        let client = ApiClient::new(config).unwrap();
        let request = TestRequest {
            message: "test".to_string(),
        };

        let start = Instant::now();
        let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(response.result, "retry_success");
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);
        // Should have waited at least 20ms due to backoff (2^1 * 10ms)
        assert!(elapsed >= Duration::from_millis(20));
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"result": "rate_limited"}))
                    .set_delay(Duration::from_millis(100)),
            )
            .mount(&mock_server)
            .await;

        let config = ApiClientConfig {
            base_url: mock_server.uri(),
            max_concurrent_requests: 1,
            max_requests_per_minute: 100,
            max_tokens_per_minute: 10000,
            ..Default::default()
        };

        let client = ApiClient::new(config).unwrap();
        let request = TestRequest {
            message: "test".to_string(),
        };

        let start = Instant::now();

        // Make two concurrent requests - second should wait for first
        let client_ref = &client;
        let request_ref = &request;
        let (result1, result2) = tokio::join!(
            client_ref.post_json::<_, TestResponse>("/test", request_ref, 10),
            client_ref.post_json::<_, TestResponse>("/test", request_ref, 10)
        );

        let elapsed = start.elapsed();

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // Should take at least 200ms (two sequential 100ms requests)
        assert!(elapsed >= Duration::from_millis(200));
    }

    #[tokio::test]
    async fn test_api_key_header() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/test"))
            .and(header("Authorization", "Bearer test_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "result": "authorized"
            })))
            .mount(&mock_server)
            .await;

        let config = ApiClientConfig {
            base_url: mock_server.uri(),
            api_key: Some(SecretString::from("test_key")),
            max_requests_per_minute: 100,
            max_tokens_per_minute: 10000,
            ..Default::default()
        };

        let client = ApiClient::new(config).unwrap();
        let request = TestRequest {
            message: "test".to_string(),
        };

        let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
        assert_eq!(response.result, "authorized");
    }

    #[tokio::test]
    async fn test_user_agent_header() {
        let mock_server = MockServer::start().await;

        let expected = format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));

        Mock::given(method("POST"))
            .and(path("/test"))
            .and(header("User-Agent", expected))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "result": "ua_ok"
            })))
            .mount(&mock_server)
            .await;

        let config = ApiClientConfig {
            base_url: mock_server.uri(),
            max_requests_per_minute: 100,
            max_tokens_per_minute: 10000,
            ..Default::default()
        };

        let client = ApiClient::new(config).unwrap();
        let request = TestRequest {
            message: "test".to_string(),
        };

        let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
        assert_eq!(response.result, "ua_ok");
    }

    #[tokio::test]
    async fn test_token_bucket_rate_limiting() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "result": "token_bucket"
            })))
            .mount(&mock_server)
            .await;

        // High rate to make tests fast but still demonstrate rate limiting
        let config = ApiClientConfig {
            base_url: mock_server.uri(),
            max_concurrent_requests: 10,
            max_requests_per_minute: 600, // 10 per second
            max_tokens_per_minute: 6000,  // 100 tokens per second
            ..Default::default()
        };

        let client = ApiClient::new(config).unwrap();
        let request = TestRequest {
            message: "test".to_string(),
        };

        // Make rapid requests to consume the burst capacity (10 seconds * 10/sec = 100).
        let start = Instant::now();
        let mut request_count = 0;
        while start.elapsed() < Duration::from_millis(300) {
            let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
            assert_eq!(response.result, "token_bucket");
            request_count += 1;
        }

        assert!(
            request_count >= 2,
            "Expected at least 2 requests, got {}",
            request_count
        );

        // Now verify rate limiting is working by timing subsequent requests.
        let before_limited = Instant::now();
        let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
        let response2: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
        let limited_elapsed = before_limited.elapsed();

        assert_eq!(response.result, "token_bucket");
        assert_eq!(response2.result, "token_bucket");

        // With 10 requests per second, spacing should be ~100ms between requests.
        assert!(
            limited_elapsed >= Duration::from_millis(90),
            "Expected rate limiting spacing, but got {:?}",
            limited_elapsed
        );
    }
}
