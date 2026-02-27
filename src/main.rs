use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use pulldown_cmark::{Event, Options, Parser as MdParser, Tag, TagEnd};
use pulldown_cmark_to_cmark::{Options as CmarkOptions, cmark_with_options};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{ErrorKind, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

const RLWRAP_GUARD_ENV: &str = "OLLAMA_MD_TRANSLATE_RLWRAP_ACTIVE";

#[derive(Parser, Debug, Clone)]
#[command(
    name = "ollama_md_translate",
    about = "Translate Markdown files via Ollama (TranslateGemma) while preserving code blocks and markup.",
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    /// Log level: error, warn, info, debug, trace
    #[arg(long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Translate markdown files or directories
    Translate(TranslateArgs),
    /// Interactive REPL mode: translate each input line immediately
    Repl(ReplArgs),
}

#[derive(Args, Debug, Clone)]
struct TranslateArgs {
    /// Input file or directory
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Source language code (ISO-639-3, e.g., "eng"). "auto" is also accepted.
    #[arg(long, default_value = "auto")]
    source_lang: String,

    /// Target language code (ISO-639-3, e.g., "deu")
    #[arg(long, default_value = "deu")]
    target_lang: String,

    /// Ollama model name (e.g., "translategemma:latest")
    #[arg(long, default_value = "translategemma:latest")]
    model: String,

    /// Ollama base URL (no trailing /api). Example: http://localhost:11434
    #[arg(long, default_value = "http://localhost:11434")]
    base_url: String,

    /// Keep model loaded in memory (Ollama keep_alive). Examples: "10m", "3600", "-1"
    #[arg(long, default_value = "10m")]
    keep_alive: String,

    /// Max concurrent translation requests
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    /// Translate YAML front matter at the top of the file (--- ... ---). Default: keep as-is.
    #[arg(long, default_value_t = false)]
    translate_frontmatter: bool,

    /// Write results in-place (atomic replace). If false, you must supply --out-dir.
    #[arg(long, default_value_t = false)]
    in_place: bool,

    /// Output directory (required unless --in-place)
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Cache directory for translated segments (default: .cache/ollama_md_translate)
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Dry run: do not write files, just log what would happen
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Args, Debug, Clone)]
struct ReplArgs {
    /// Source language code (ISO-639-3, e.g., "eng"). "auto" is also accepted.
    #[arg(long)]
    source_lang: String,

    /// Target language code (ISO-639-3, e.g., "deu")
    #[arg(long)]
    target_lang: String,

    /// Ollama model name (e.g., "translategemma:latest")
    #[arg(long, default_value = "translategemma:latest")]
    model: String,

    /// Ollama base URL (no trailing /api). Example: http://localhost:11434
    #[arg(long, default_value = "http://localhost:11434")]
    base_url: String,

    /// Keep model loaded in memory (Ollama keep_alive). Examples: "10m", "3600", "-1"
    #[arg(long, default_value = "10m")]
    keep_alive: String,

    /// Cache directory for translated segments (default: .cache/ollama_md_translate)
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct RuntimeConfig {
    source_lang: String,
    target_lang: String,
    model: String,
    base_url: String,
    keep_alive: String,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
    #[allow(dead_code)]
    model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    init_tracing(&cli.log_level)?;
    info!("log_level = {}", cli.log_level);

    if let Some(code) = maybe_run_repl_with_rlwrap(&cli)? {
        if code == 0 {
            return Ok(());
        }
        bail!("rlwrap child exited with status code {}", code);
    }

    match cli.command {
        Commands::Translate(args) => run_translate(args).await,
        Commands::Repl(args) => run_repl(args).await,
    }
}

async fn run_translate(args: TranslateArgs) -> Result<()> {
    validate_args(&args)?;
    let stop_requested = install_ctrlc_handler("translate");

    let runtime = RuntimeConfig {
        source_lang: normalize_lang_code(&args.source_lang, true)?,
        target_lang: normalize_lang_code(&args.target_lang, false)?,
        model: args.model.clone(),
        base_url: args.base_url.clone(),
        keep_alive: args.keep_alive.clone(),
    };

    let cache_dir = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".cache/ollama_md_translate"));

    if !args.dry_run {
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("creating cache dir: {}", cache_dir.display()))?;
    }

    info!("mode = translate");
    info!("input = {}", args.input.display());
    info!("model = {}", runtime.model);
    info!("base_url = {}", runtime.base_url);
    info!("source_lang = {}", runtime.source_lang);
    info!("target_lang = {}", runtime.target_lang);
    info!("in_place = {}", args.in_place);
    if let Some(out) = &args.out_dir {
        info!("out_dir = {}", out.display());
    }
    info!("cache_dir = {}", cache_dir.display());
    info!("concurrency = {}", args.concurrency);
    info!("dry_run = {}", args.dry_run);

    let client = Client::builder()
        .user_agent("ollama_md_translate/0.1.0")
        .build()
        .context("building reqwest client")?;

    let files = collect_markdown_files(&args.input)?;
    if files.is_empty() {
        warn!("No markdown files found at {}", args.input.display());
        return Ok(());
    }

    info!("found {} markdown file(s)", files.len());

    // Concurrency guard for HTTP calls.
    let sem = std::sync::Arc::new(Semaphore::new(args.concurrency));

    let mut failures = 0usize;

    for (idx, path) in files.iter().enumerate() {
        if stop_requested.load(Ordering::SeqCst) {
            warn!("Ctrl+C received; stopping after current file boundary");
            break;
        }

        info!(
            "[{}/{}] processing {}",
            idx + 1,
            files.len(),
            path.display()
        );

        match process_one_file(&args, &runtime, &client, &cache_dir, sem.clone(), path).await {
            Ok(()) => info!("ok: {}", path.display()),
            Err(e) => {
                failures += 1;
                error!("failed: {}: {:#}", path.display(), e);
            }
        }
    }

    if failures > 0 {
        bail!("completed with {} failure(s)", failures);
    }
    if stop_requested.load(Ordering::SeqCst) {
        info!("translation run interrupted by Ctrl+C");
        return Ok(());
    }

    info!("done");
    Ok(())
}

async fn run_repl(args: ReplArgs) -> Result<()> {
    validate_repl_args(&args)?;
    let use_internal_ctrlc = env::var_os(RLWRAP_GUARD_ENV).is_none();
    let stop_requested = if use_internal_ctrlc {
        Some(install_ctrlc_handler("repl"))
    } else {
        debug!("running under rlwrap; delegating Ctrl+C handling to rlwrap");
        None
    };

    let runtime = RuntimeConfig {
        source_lang: normalize_lang_code(&args.source_lang, true)?,
        target_lang: normalize_lang_code(&args.target_lang, false)?,
        model: args.model.clone(),
        base_url: args.base_url.clone(),
        keep_alive: args.keep_alive.clone(),
    };
    let cache_dir = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".cache/ollama_md_translate"));
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating cache dir: {}", cache_dir.display()))?;

    info!("mode = repl");
    info!("model = {}", runtime.model);
    info!("base_url = {}", runtime.base_url);
    info!("source_lang = {}", runtime.source_lang);
    info!("target_lang = {}", runtime.target_lang);
    info!("cache_dir = {}", cache_dir.display());
    info!("REPL started; reading lines from stdin until EOF");

    let client = Client::builder()
        .user_agent("ollama_md_translate/0.1.0")
        .build()
        .context("building reqwest client")?;
    let sem = std::sync::Arc::new(Semaphore::new(1));

    let stdin = std::io::stdin();
    let mut line_no = 0usize;
    let mut line = String::new();
    loop {
        if stop_requested
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
        {
            info!("REPL interrupted by Ctrl+C");
            break;
        }

        line.clear();
        let bytes = match stdin.read_line(&mut line) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == ErrorKind::Interrupted => {
                if stop_requested
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::SeqCst))
                {
                    info!("REPL interrupted by Ctrl+C");
                    break;
                }
                debug!("stdin read interrupted; retrying");
                continue;
            }
            Err(e) if e.raw_os_error() == Some(5) => {
                info!("REPL input stream closed (I/O error); exiting");
                break;
            }
            Err(e) => return Err(e).context("reading stdin line from REPL"),
        };
        if bytes == 0 {
            info!("REPL received EOF (Ctrl+D)");
            break;
        }
        line_no += 1;

        while line.ends_with('\n') || line.ends_with('\r') {
            line.pop();
        }

        debug!("repl line={} bytes={}", line_no, line.len());
        if line.trim().is_empty() {
            println!();
            std::io::stdout().flush().context("flushing stdout")?;
            continue;
        }
        let translated = translate_plaintext(
            &runtime,
            &client,
            &cache_dir,
            sem.clone(),
            line.as_str(),
            "repl_line",
        )
        .await
        .with_context(|| format!("translating stdin line {}", line_no))?;
        if std::io::stdout().is_terminal() {
            println!("\x1b[1m{}\x1b[0m", translated);
        } else {
            println!("{}", translated);
        }
        std::io::stdout().flush().context("flushing stdout")?;
    }

    info!("REPL ended (EOF)");
    Ok(())
}

fn maybe_run_repl_with_rlwrap(cli: &Cli) -> Result<Option<i32>> {
    if !matches!(cli.command, Commands::Repl(_)) {
        return Ok(None);
    }
    if env::var_os(RLWRAP_GUARD_ENV).is_some() {
        debug!("rlwrap guard env is present; not re-executing");
        return Ok(None);
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        debug!("stdin/stdout is not a terminal; skipping rlwrap integration");
        return Ok(None);
    }

    let exe = env::current_exe().context("resolving current executable path")?;
    let args: Vec<_> = env::args_os().skip(1).collect();
    info!("starting repl through rlwrap");

    match Command::new("rlwrap")
        .arg("--no-warnings")
        .arg("--pass-sigint-as-sigterm")
        .arg(exe)
        .args(args)
        .env(RLWRAP_GUARD_ENV, "1")
        .status()
    {
        Ok(status) => {
            let code = status.code().unwrap_or(1);
            info!("rlwrap child exited with status code {}", code);
            Ok(Some(code))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!("rlwrap not found in PATH; continuing without rlwrap");
            Ok(None)
        }
        Err(e) => Err(e).context("failed to launch rlwrap"),
    }
}

fn install_ctrlc_handler(scope: &str) -> std::sync::Arc<AtomicBool> {
    let stop_requested = std::sync::Arc::new(AtomicBool::new(false));
    let stop_requested_task = stop_requested.clone();
    let scope = scope.to_string();
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                stop_requested_task.store(true, Ordering::SeqCst);
                warn!(
                    "received Ctrl+C in {} mode; requesting graceful shutdown",
                    scope
                );
            }
            Err(e) => {
                error!("failed to listen for Ctrl+C in {} mode: {:#}", scope, e);
            }
        }
    });
    stop_requested
}

fn init_tracing(level: &str) -> Result<()> {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    Ok(())
}

fn validate_args(args: &TranslateArgs) -> Result<()> {
    if args.in_place && args.out_dir.is_some() {
        bail!("Use either --in-place or --out-dir, not both");
    }
    if !args.in_place && args.out_dir.is_none() {
        bail!("If not using --in-place, you must provide --out-dir");
    }
    validate_lang_code(&args.source_lang, "source language", true)?;
    validate_lang_code(&args.target_lang, "target language", false)?;
    Ok(())
}

fn validate_repl_args(args: &ReplArgs) -> Result<()> {
    validate_lang_code(&args.source_lang, "source language", true)?;
    validate_lang_code(&args.target_lang, "target language", false)?;
    Ok(())
}

fn validate_lang_code(code: &str, label: &str, allow_auto: bool) -> Result<()> {
    if normalize_lang_code(code, allow_auto).is_err() {
        bail!(
            "{} must be a 3-letter code (e.g., eng, deu){}: {}",
            label,
            if allow_auto { " or 'auto'" } else { "" },
            code
        );
    }
    Ok(())
}

fn normalize_lang_code(code: &str, allow_auto: bool) -> Result<String> {
    let normalized = code.trim().to_ascii_lowercase();
    if allow_auto && normalized == "auto" {
        return Ok(normalized);
    }
    if normalized.len() != 3 || !normalized.chars().all(|c| c.is_ascii_alphabetic()) {
        bail!("invalid 3-letter language code: {}", code);
    }

    let canonical = match normalized.as_str() {
        // Common ISO-639-2/B aliases -> ISO-639-3 forms.
        "gre" => "ell",
        "ger" => "deu",
        "fre" => "fra",
        "rum" => "ron",
        "slo" => "slk",
        "alb" => "sqi",
        "arm" => "hye",
        "baq" => "eus",
        "bur" => "mya",
        "chi" => "zho",
        "cze" => "ces",
        "dut" => "nld",
        "geo" => "kat",
        "ice" => "isl",
        "mac" => "mkd",
        "mao" => "mri",
        "may" => "msa",
        "per" => "fas",
        "tib" => "bod",
        "wel" => "cym",
        _ => normalized.as_str(),
    };
    Ok(canonical.to_string())
}

fn language_display_name(code: &str) -> &'static str {
    match code {
        "eng" => "English",
        "ell" => "Greek",
        "rus" => "Russian",
        "deu" => "German",
        "spa" => "Spanish",
        "fra" => "French",
        "ita" => "Italian",
        "por" => "Portuguese",
        "zho" => "Chinese",
        "jpn" => "Japanese",
        "kor" => "Korean",
        "ukr" => "Ukrainian",
        "pol" => "Polish",
        "arb" => "Arabic",
        "hin" => "Hindi",
        "auto" => "Auto-detect",
        _ => "Unknown",
    }
}

fn is_markdown_path(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(OsStr::to_str)
            .map(|s| s.to_lowercase())
            .as_deref(),
        Some("md") | Some("markdown") | Some("mdown") | Some("mkd") | Some("mkdn")
    )
}

fn collect_markdown_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if is_markdown_path(input) {
            return Ok(vec![input.to_path_buf()]);
        } else {
            return Ok(vec![]);
        }
    }

    if !input.is_dir() {
        bail!("input is neither file nor directory: {}", input.display());
    }

    let mut out = Vec::new();
    for ent in WalkDir::new(input).follow_links(false) {
        let ent = ent?;
        if ent.file_type().is_file() && is_markdown_path(ent.path()) {
            out.push(ent.path().to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

async fn process_one_file(
    args: &TranslateArgs,
    runtime: &RuntimeConfig,
    client: &Client,
    cache_dir: &Path,
    sem: std::sync::Arc<Semaphore>,
    path: &Path,
) -> Result<()> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let (frontmatter_opt, body) = split_frontmatter(&raw);

    let translated_frontmatter = if let Some(frontmatter) = frontmatter_opt.as_deref() {
        if args.translate_frontmatter {
            info!("translating front matter");
            Some(
                translate_plaintext(
                    runtime,
                    client,
                    cache_dir,
                    sem.clone(),
                    frontmatter,
                    "frontmatter",
                )
                .await?,
            )
        } else {
            debug!("keeping front matter as-is");
            Some(frontmatter.to_string())
        }
    } else {
        None
    };

    let translated_body =
        translate_markdown_body(runtime, client, cache_dir, sem.clone(), body).await?;

    let mut final_out = String::new();
    if let Some(fm) = translated_frontmatter {
        final_out.push_str(&fm);
    }
    final_out.push_str(&translated_body);

    let out_path = compute_output_path(args, path)?;
    if args.dry_run {
        info!("dry_run: would write {}", out_path.display());
        return Ok(());
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }

    atomic_write(&out_path, final_out.as_bytes())
        .with_context(|| format!("writing {}", out_path.display()))?;

    Ok(())
}

fn compute_output_path(args: &TranslateArgs, in_path: &Path) -> Result<PathBuf> {
    if args.in_place {
        return Ok(in_path.to_path_buf());
    }

    let out_dir = args
        .out_dir
        .as_ref()
        .ok_or_else(|| anyhow!("missing --out-dir"))?;

    if args.input.is_file() {
        // Single file: output is out_dir/<filename>
        let fname = in_path
            .file_name()
            .ok_or_else(|| anyhow!("bad filename: {}", in_path.display()))?;
        Ok(out_dir.join(fname))
    } else {
        // Directory: preserve relative path under input root
        let rel = in_path
            .strip_prefix(&args.input)
            .with_context(|| format!("computing relative path under {}", args.input.display()))?;
        Ok(out_dir.join(rel))
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp.ollama_md_translate");
    {
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("creating temp file {}", tmp.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("writing temp file {}", tmp.display()))?;
        f.sync_all()
            .with_context(|| format!("sync temp file {}", tmp.display()))?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("atomic rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Splits YAML front matter if present, returning (frontmatter_with_delimiters, body)
fn split_frontmatter(input: &str) -> (Option<String>, &str) {
    // YAML front matter pattern:
    // ---\n
    // ...\n
    // ---\n
    // <body>
    if !input.starts_with("---\n") {
        return (None, input);
    }

    // Find the next "\n---\n" after the opening.
    // Limit search to first ~200KB to avoid pathological cases.
    let hay = &input[..input.len().min(200_000)];
    if let Some(end_idx) = hay[4..].find("\n---\n") {
        // end_idx is relative to hay[4..]
        let fm_end = 4 + end_idx + "\n---\n".len();
        let (fm, rest) = input.split_at(fm_end);
        return (Some(fm.to_string()), rest);
    }

    (None, input)
}

async fn translate_markdown_body(
    runtime: &RuntimeConfig,
    client: &Client,
    cache_dir: &Path,
    sem: std::sync::Arc<Semaphore>,
    body: &str,
) -> Result<String> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts.insert(Options::ENABLE_SMART_PUNCTUATION);

    let parser = MdParser::new_ext(body, opts);

    let mut in_code_block_depth: usize = 0;
    let mut in_html_block_depth: usize = 0;

    // Keep the original event lifetime ('body) — translated text becomes OWNED.
    let mut out_events: Vec<Event<'_>> = Vec::new();

    let mut memo: HashMap<String, String> = HashMap::new();

    for ev in parser {
        match ev {
            Event::Start(tag) => {
                match &tag {
                    Tag::CodeBlock(_) => in_code_block_depth += 1,
                    Tag::HtmlBlock => in_html_block_depth += 1,
                    _ => {}
                }
                out_events.push(Event::Start(tag));
            }

            Event::End(tag_end) => {
                match &tag_end {
                    TagEnd::CodeBlock => {
                        if in_code_block_depth > 0 {
                            in_code_block_depth -= 1;
                        }
                    }
                    TagEnd::HtmlBlock => {
                        if in_html_block_depth > 0 {
                            in_html_block_depth -= 1;
                        }
                    }
                    _ => {}
                }
                out_events.push(Event::End(tag_end));
            }

            Event::Code(c) => {
                // inline code: do not translate
                out_events.push(Event::Code(c));
            }

            Event::Html(h) => {
                // raw inline HTML: do not translate
                out_events.push(Event::Html(h));
            }

            Event::Text(t) => {
                if in_code_block_depth > 0 || in_html_block_depth > 0 {
                    out_events.push(Event::Text(t));
                    continue;
                }

                let original = t.to_string();

                if original.trim().is_empty() {
                    out_events.push(Event::Text(t));
                    continue;
                }

                if let Some(hit) = memo.get(&original) {
                    out_events.push(Event::Text(hit.clone().into()));
                    continue;
                }

                let translated = translate_plaintext(
                    runtime,
                    client,
                    cache_dir,
                    sem.clone(),
                    &original,
                    "markdown_text",
                )
                .await?;

                memo.insert(original, translated.clone());
                // IMPORTANT: this becomes an OWNED CowStr internally, lifetime-safe.
                out_events.push(Event::Text(translated.into()));
            }

            other => {
                out_events.push(other);
            }
        }
    }

    // Render back to Markdown (CommonMark-ish). This may normalize whitespace.
    let mut out = String::new();
    let mut cmark_opts = CmarkOptions::default();
    cmark_opts.code_block_token_count = 3;

    // Convert Vec<Event> -> iterator of Cow<Event> so E: Borrow<Event> is satisfied.
    let cow_iter = out_events.into_iter().map(Cow::Owned);

    let _state =
        cmark_with_options(cow_iter, &mut out, cmark_opts).context("serializing markdown")?;

    Ok(out)
}

async fn translate_plaintext(
    runtime: &RuntimeConfig,
    client: &Client,
    cache_dir: &Path,
    sem: std::sync::Arc<Semaphore>,
    text: &str,
    kind: &str,
) -> Result<String> {
    let key = cache_key(
        &runtime.model,
        &runtime.source_lang,
        &runtime.target_lang,
        kind,
        text,
    );
    if let Some(hit) = cache_read(cache_dir, &key)? {
        debug!("cache hit: kind={} bytes={}", kind, text.len());
        return Ok(hit);
    }

    debug!("cache miss: kind={} bytes={}", kind, text.len());

    // Prevent flooding the server with too many concurrent requests.
    let _permit = sem.acquire().await.context("semaphore acquire")?;

    let prompt = build_translation_prompt(&runtime.source_lang, &runtime.target_lang, text);

    let url = format!("{}/api/generate", runtime.base_url.trim_end_matches('/'));

    debug!("POST {} ({} bytes)", url, prompt.len());

    let req = OllamaGenerateRequest {
        model: &runtime.model,
        prompt: &prompt,
        stream: false,
        system: None,
        keep_alive: Some(&runtime.keep_alive),
    };

    let resp = client
        .post(url)
        .json(&req)
        .send()
        .await
        .context("ollama request failed")?;

    let status = resp.status();
    let body = resp.text().await.context("reading ollama response body")?;

    if !status.is_success() {
        bail!("ollama returned {}: {}", status, body);
    }

    let parsed: OllamaGenerateResponse =
        serde_json::from_str(&body).context("parsing ollama JSON")?;

    let translated = postprocess_translation(&parsed.response);

    cache_write(cache_dir, &key, &translated)?;
    Ok(translated)
}

fn build_translation_prompt(source: &str, target: &str, text: &str) -> String {
    // Very strict: output only the translation.
    // We say "plain text extracted from Markdown" to discourage adding markup.
    let source_name = language_display_name(source);
    let target_name = language_display_name(target);
    format!(
        "You are a translation engine.\n\
         Translate from source language code {source} ({source_name}) to target language code {target} ({target_name}).\n\
         The input is plain text extracted from a Markdown document.\n\
         Rules:\n\
         - Output ONLY the translation (no preface, no quotes, no notes).\n\
         - Preserve line breaks exactly.\n\
         - Do not add Markdown syntax.\n\
         - Keep URLs and email addresses unchanged if they appear.\n\n\
        INPUT:\n{text}\n",
        source = source,
        source_name = source_name,
        target = target,
        target_name = target_name,
        text = text
    )
}

fn postprocess_translation(s: &str) -> String {
    // Ollama responses often include a trailing newline; keep it stable but avoid extra whitespace.
    // We do NOT trim internal whitespace or line breaks.
    let mut out = s.to_string();

    // Remove a single leading BOM if it appears (rare).
    if out.starts_with('\u{feff}') {
        out = out.trim_start_matches('\u{feff}').to_string();
    }

    // Remove a single trailing newline if it's the only trailing whitespace (common model behavior).
    // But keep intentional multi-line trailing content.
    while out.ends_with("\r\n") {
        out.pop();
        out.pop();
        out.push('\n');
    }

    // If model adds extra leading/trailing blank lines, trim only outermost blank lines.
    // This is conservative: it won't affect middle formatting.
    trim_outer_blank_lines(&out)
}

fn trim_outer_blank_lines(s: &str) -> String {
    let lines: Vec<&str> = s.split('\n').collect();
    if lines.is_empty() {
        return s.to_string();
    }

    let mut start = 0usize;
    let mut end = lines.len();

    while start < end && lines[start].trim().is_empty() {
        start += 1;
    }
    while end > start && lines[end - 1].trim().is_empty() {
        end -= 1;
    }

    let mut out = lines[start..end].join("\n");
    // Preserve a final newline if original had one and there is content.
    if s.ends_with('\n') && !out.is_empty() {
        out.push('\n');
    }
    out
}

fn cache_key(model: &str, source: &str, target: &str, kind: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update(b"\n");
    hasher.update(source.as_bytes());
    hasher.update(b"\n");
    hasher.update(target.as_bytes());
    hasher.update(b"\n");
    hasher.update(kind.as_bytes());
    hasher.update(b"\n");
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    hex_lower(&digest)
}

fn cache_read(cache_dir: &Path, key: &str) -> Result<Option<String>> {
    let path = cache_dir.join(format!("{}.txt", key));
    if !path.exists() {
        return Ok(None);
    }
    let s = fs::read_to_string(&path)
        .with_context(|| format!("reading cache file {}", path.display()))?;
    Ok(Some(s))
}

fn cache_write(cache_dir: &Path, key: &str, value: &str) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;
    let path = cache_dir.join(format!("{}.txt", key));
    fs::write(&path, value.as_bytes())
        .with_context(|| format!("writing cache file {}", path.display()))?;
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}
