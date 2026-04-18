// cursor-agent Auto-model logger
//
// Preloaded via `node --require`. We hook `tls.connect` on any connection to
// a cursor.sh / cursor.com host and observe the plaintext HTTP/1.1 stream
// (pre-encrypt on write, post-decrypt on read). For every request/response
// pair we extract:
//
//   - request: method, path, a safe subset of headers, body bytes
//   - response: status, headers, body bytes (gunzipped if Content-Encoding: gzip)
//   - scan: ASCII substrings that look like model ids
//     (composer-*, claude-*, gpt-*, gemini-*, grok-*, kimi-*)
//
// One JSON line per event to $CURSOR_AUTO_LOG. The analyzer (analyze.js)
// aggregates these into a per-turn / per-model-pick summary.
//
// Security: we NEVER write Authorization, Cookie, or any header matching a
// redaction pattern to disk.

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const tls = require('node:tls');
const net = require('node:net');
const zlib = require('node:zlib');

const LOG_PATH = process.env.CURSOR_AUTO_LOG
  || path.join(process.cwd(), 'cursor-auto.log');
const VERBOSE = process.env.CURSOR_AUTO_LOG_VERBOSE === '1';
const SCAN_CAP = Number(process.env.CURSOR_AUTO_LOG_SCAN_CAP || 4 * 1024 * 1024);

const API_HOST_RE = /(^|\.)cursor\.(sh|com)$/i;

const MODEL_ID_RE =
  /\b(?:composer-\d[a-z0-9.\-]*|claude-[a-z0-9.\-]{3,40}|gpt-[a-z0-9.\-]{1,40}|gemini-[a-z0-9.\-]{1,40}|grok-[a-z0-9.\-]{1,40}|kimi-[a-z0-9.\-]{1,40})\b/gi;

// Header names to redact entirely (value replaced with <redacted>). Lowercased.
const REDACT_HEADERS = new Set([
  'authorization',
  'cookie',
  'set-cookie',
  'x-access-token',
  'x-token-auth',
  'x-api-key',
  'proxy-authorization',
]);
// Additional body-level redactions (JWTs etc.) applied to any sampled strings.
const JWT_RE = /\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+/g;
const BEARER_RE = /Bearer\s+[A-Za-z0-9._\-~+/=]+/gi;

function redact(str) {
  if (typeof str !== 'string') return str;
  return str.replace(JWT_RE, '<redacted-jwt>').replace(BEARER_RE, 'Bearer <redacted>');
}

function log(obj) {
  try { fs.appendFileSync(LOG_PATH, JSON.stringify(obj) + '\n'); } catch (_) {}
  if (VERBOSE) {
    try { process.stderr.write('[auto-log] ' + JSON.stringify(obj) + '\n'); }
    catch (_) {}
  }
}

log({
  t: new Date().toISOString(),
  event: 'session_start',
  pid: process.pid,
  argv: process.argv,
  cwd: process.cwd(),
  host: os.hostname(),
  logPath: LOG_PATH,
});

// ---------------------------------------------------------------------------
// scanning helpers

function printableStrings(buf, minLen = 4, cap = SCAN_CAP) {
  const n = Math.min(buf.length, cap);
  const out = [];
  let start = -1;
  for (let i = 0; i < n; i++) {
    const c = buf[i];
    const printable = c >= 0x20 && c < 0x7f;
    if (printable) {
      if (start < 0) start = i;
    } else {
      if (start >= 0 && i - start >= minLen) {
        out.push(Buffer.from(buf.subarray(start, i)).toString('ascii'));
      }
      start = -1;
    }
  }
  if (start >= 0 && n - start >= minLen) {
    out.push(Buffer.from(buf.subarray(start, n)).toString('ascii'));
  }
  return out;
}

function scanModelHits(buf) {
  if (!buf || buf.length === 0) return { modelHits: [], sampleStrings: [] };
  const strings = printableStrings(buf);
  const joined = strings.join('\n');
  const hits = new Set();
  let m;
  MODEL_ID_RE.lastIndex = 0;
  while ((m = MODEL_ID_RE.exec(joined)) !== null) hits.add(m[0].toLowerCase());
  // Redact JWTs / bearer tokens from any sampled strings we write to disk.
  const redactedSample = strings
    .filter(s => s.length > 12)
    .slice(0, 20)
    .map(redact);
  return { modelHits: [...hits], sampleStrings: redactedSample };
}

// ---------------------------------------------------------------------------
// HTTP/1.1 framing parser.
//
// undici speaks HTTP/1.1 over TLS. For each connection we accumulate the
// bytes, then parse out request and response frames. Bodies can be
// chunked-encoded or content-length delimited, and sometimes gzip-encoded.
// We only need to extract: method, path, headers, body.

function parseHttpStream(buf) {
  // Returns { messages: [{kind:'req'|'resp', firstLine, headers, body}] }
  const messages = [];
  let off = 0;
  while (off < buf.length) {
    // Find end of headers: "\r\n\r\n"
    const hdrEnd = indexOf(buf, Buffer.from('\r\n\r\n'), off);
    if (hdrEnd < 0) break;
    const header = buf.slice(off, hdrEnd).toString('utf8', 0, Math.min(hdrEnd - off, 32 * 1024));
    const lines = header.split(/\r\n/);
    const firstLine = lines.shift() || '';
    const headers = {};
    for (const l of lines) {
      const idx = l.indexOf(':');
      if (idx > 0) headers[l.slice(0, idx).trim().toLowerCase()] = l.slice(idx + 1).trim();
    }
    const kind = /^HTTP\//.test(firstLine) ? 'resp' : 'req';

    let bodyStart = hdrEnd + 4;
    let bodyEnd = buf.length; // best-effort if no length indicators
    const cl = headers['content-length'];
    const te = (headers['transfer-encoding'] || '').toLowerCase();

    let body = null;
    if (te.includes('chunked')) {
      const parsed = parseChunked(buf, bodyStart);
      body = parsed.body;
      bodyEnd = parsed.end;
    } else if (cl != null) {
      const n = parseInt(cl, 10);
      if (Number.isFinite(n) && n >= 0) {
        bodyEnd = Math.min(buf.length, bodyStart + n);
        body = buf.slice(bodyStart, bodyEnd);
      }
    } else if (kind === 'req') {
      // Request with no body indicator: assume empty (Connect unary uses
      // content-length).
      bodyEnd = bodyStart;
      body = Buffer.alloc(0);
    } else {
      // Response with no CL/TE: read till close. We'll just take whatever is
      // left (but this is uncommon for Connect).
      body = buf.slice(bodyStart, bodyEnd);
    }
    messages.push({ kind, firstLine, headers, body });
    off = bodyEnd;
  }
  return { messages };
}

function parseChunked(buf, start) {
  const chunks = [];
  let p = start;
  while (p < buf.length) {
    const eol = indexOf(buf, Buffer.from('\r\n'), p);
    if (eol < 0) break;
    const sizeStr = buf.slice(p, eol).toString('ascii').split(';')[0].trim();
    const size = parseInt(sizeStr, 16);
    if (!Number.isFinite(size)) break;
    if (size === 0) {
      // trailers + final CRLF — skip until blank line
      p = eol + 2;
      const trailerEnd = indexOf(buf, Buffer.from('\r\n\r\n'), p - 2);
      if (trailerEnd >= 0) p = trailerEnd + 4;
      break;
    }
    const chunkStart = eol + 2;
    const chunkEnd = chunkStart + size;
    if (chunkEnd > buf.length) break;
    chunks.push(buf.slice(chunkStart, chunkEnd));
    p = chunkEnd + 2;
  }
  return { body: Buffer.concat(chunks), end: p };
}

function indexOf(buf, needle, from = 0) {
  // Buffer.indexOf exists but sometimes slow for large streams; native is fine.
  return buf.indexOf(needle, from);
}

function maybeDecompress(body, encoding) {
  if (!body || body.length === 0) return body;
  const enc = String(encoding || '').toLowerCase();
  try {
    if (enc === 'gzip') return zlib.gunzipSync(body);
    if (enc === 'deflate') return zlib.inflateSync(body);
    if (enc === 'br') return zlib.brotliDecompressSync(body);
  } catch (_) {
    // Leave body as-is; the scanner still works on raw bytes.
  }
  return body;
}

function filterHeaders(h) {
  const out = {};
  for (const k of Object.keys(h || {})) {
    if (REDACT_HEADERS.has(k)) { out[k] = '<redacted>'; continue; }
    out[k] = redact(h[k]);
  }
  return out;
}

// Track per-connection finalizers so we can flush them on process exit,
// before the kernel/runtime drops the sockets.
const openFinalizers = new Set();

// ---------------------------------------------------------------------------
// tls.connect hook

function installTlsHook() {
  const origConnect = tls.connect;
  if (origConnect.__cursorAutoLogHooked) return;

  function wrappedConnect(...args) {
    const socket = origConnect.apply(tls, args);
    let host = '', servername = '', rawHost = '';
    try {
      const opts = typeof args[0] === 'object' ? args[0]
                 : typeof args[1] === 'object' ? args[1]
                 : {};
      servername = (opts && opts.servername) || '';
      rawHost = (opts && opts.host) || '';
      host = servername || rawHost || '';
    } catch (_) {}
    // Match either servername (SNI, reliable) or host (may be an IP). Since
    // the CLI sometimes resolves hostnames to IPs in JS and passes the IP as
    // `host` while keeping servername, accept connections where either is a
    // cursor host. Also opt into "all" for debugging.
    const interesting = API_HOST_RE.test(host) || API_HOST_RE.test(servername) || API_HOST_RE.test(rawHost);
    if (!interesting) {
      if (process.env.CURSOR_AUTO_LOG_ALL_HOSTS === '1') {
        log({
          t: new Date().toISOString(),
          event: 'tls_connect_skipped',
          host, servername, rawHost,
          pid: process.pid,
        });
      }
      return socket;
    }

    const connId = `${Date.now()}-${Math.floor(Math.random() * 1e6).toString(36)}`;
    const startedAt = Date.now();
    log({
      t: new Date().toISOString(),
      event: 'tls_connect',
      conn: connId,
      host, servername, rawHost,
      pid: process.pid,
    });
    const writes = [];
    let writeBytes = 0;
    const reads = [];
    let readBytes = 0;

    const origWrite = socket.write.bind(socket);
    socket.write = function patchedWrite(chunk, ...rest) {
      try {
        if (chunk && writeBytes < SCAN_CAP) {
          const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
          writes.push(buf);
          writeBytes += buf.length;
          // Real-time request line log so long-lived streams are visible
          // before the connection closes.
          const preview = buf.toString('utf8', 0, Math.min(buf.length, 512));
          const m = preview.match(/^(POST|GET|PUT|DELETE) (\S+) HTTP\/1\.1/);
          if (m) {
            log({
              t: new Date().toISOString(),
              event: 'http_req_line',
              conn: connId,
              host,
              method: m[1],
              path: m[2],
              pid: process.pid,
            });
          }
        }
      } catch (_) {}
      return origWrite(chunk, ...rest);
    };

    // Hook the Readable.push() used by TLSSocket to inject decrypted plaintext
    // into the stream. This catches data regardless of whether the consumer
    // reads via 'data' events, pipe(), or direct .read() calls.
    const origPush = socket.push.bind(socket);
    socket.push = function patchedPush(chunk, encoding) {
      try {
        if (chunk && readBytes < SCAN_CAP) {
          const buf = Buffer.isBuffer(chunk) ? chunk
                    : (typeof chunk === 'string' ? Buffer.from(chunk, encoding || 'utf8') : null);
          if (buf) {
            reads.push(buf);
            readBytes += buf.length;
          }
        }
      } catch (_) {}
      return origPush(chunk, encoding);
    };

    const finalize = () => {
      try {
        const reqBuf = Buffer.concat(writes, Math.min(writeBytes, SCAN_CAP));
        const respBuf = Buffer.concat(reads, Math.min(readBytes, SCAN_CAP));
        log({
          t: new Date().toISOString(),
          event: 'tls_finalize',
          conn: connId,
          host,
          pid: process.pid,
          reqBufLen: reqBuf.length,
          respBufLen: respBuf.length,
        });
        const reqMsgs = parseHttpStream(reqBuf).messages.filter(m => m.kind === 'req');
        const respMsgs = parseHttpStream(respBuf).messages.filter(m => m.kind === 'resp');
        // For connections where we captured bytes but parsing produced no
        // complete messages (e.g. the response stream is still mid-flight),
        // emit a raw preview so the user has something to scan.
        if (reqMsgs.length === 0 && respMsgs.length === 0 && (reqBuf.length > 0 || respBuf.length > 0)) {
          log({
            t: new Date().toISOString(),
            event: 'tls_raw_preview',
            conn: connId,
            host,
            pid: process.pid,
            reqPreview: reqBuf.toString('utf8', 0, Math.min(reqBuf.length, 2048)).replace(JWT_RE, '<redacted-jwt>').replace(BEARER_RE, 'Bearer <redacted>'),
            respPreviewHead: respBuf.toString('utf8', 0, Math.min(respBuf.length, 2048)),
            respScan: scanModelHits(respBuf),
          });
        }
        // Pair them positionally (HTTP/1.1 keep-alive is sequential).
        const n = Math.max(reqMsgs.length, respMsgs.length);
        for (let i = 0; i < n; i++) {
          const req = reqMsgs[i];
          const resp = respMsgs[i];
          const reqBody = req ? req.body : null;
          const respBody = resp ? maybeDecompress(resp.body, resp.headers['content-encoding']) : null;
          const reqScan = reqBody ? scanModelHits(reqBody) : null;
          const respScan = respBody ? scanModelHits(respBody) : null;
          const turnId = `${connId}:${i}`;
          log({
            t: new Date().toISOString(),
            event: 'http',
            conn: connId,
            turn: turnId,
            host,
            pid: process.pid,
            reqLine: req ? req.firstLine : null,
            reqPath: req ? (req.firstLine.split(/\s+/)[1] || '') : null,
            reqMethod: req ? (req.firstLine.split(/\s+/)[0] || '') : null,
            reqHeaders: req ? filterHeaders(req.headers) : null,
            reqBodyLen: reqBody ? reqBody.length : 0,
            reqScan,
            respLine: resp ? resp.firstLine : null,
            respStatus: resp ? parseInt((resp.firstLine.split(/\s+/)[1] || '0'), 10) : null,
            respHeaders: resp ? filterHeaders(resp.headers) : null,
            respBodyLen: respBody ? respBody.length : 0,
            respEncoding: resp ? resp.headers['content-encoding'] || null : null,
            respScan,
            durMs: Date.now() - startedAt,
          });
        }
      } catch (err) {
        log({
          t: new Date().toISOString(),
          event: 'parse_error',
          conn: connId,
          error: String(err && err.message || err),
        });
      }
    };

    socket.once('close', finalize);
    // Flush whatever we have if the process exits before the socket closes.
    openFinalizers.add(finalize);
    socket.once('close', () => openFinalizers.delete(finalize));
    return socket;
  }
  wrappedConnect.__cursorAutoLogHooked = true;
  tls.connect = wrappedConnect;
  log({ t: new Date().toISOString(), event: 'tls_hook_installed' });
}

installTlsHook();

// Also hook net.connect to see Unix domain socket traffic (worker.sock).
// This helps when the CLI talks to a local daemon that proxies cursor.sh.
function installNetHook() {
  const origConnect = net.connect;
  if (origConnect.__cursorAutoLogHooked) return;

  function wrappedConnect(...args) {
    const socket = origConnect.apply(net, args);
    let target = '';
    try {
      const opts = typeof args[0] === 'object' ? args[0] : null;
      if (opts && opts.path) target = `unix:${opts.path}`;
      else if (opts && opts.port) target = `${opts.host || 'localhost'}:${opts.port}`;
      else if (typeof args[0] === 'string' || typeof args[0] === 'number') {
        target = `${args[0]}${args[1] ? ':' + args[1] : ''}`;
      }
    } catch (_) {}
    if (!/^unix:/.test(target)) return socket; // only care about IPC here

    const connId = `n${Date.now()}-${Math.floor(Math.random() * 1e6).toString(36)}`;
    const startedAt = Date.now();
    log({
      t: new Date().toISOString(),
      event: 'net_connect',
      conn: connId,
      target,
      pid: process.pid,
    });

    let wrote = 0, read = 0;
    const origWrite = socket.write.bind(socket);
    socket.write = function(chunk, ...rest) {
      try {
        if (chunk) {
          const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
          wrote += buf.length;
          const preview = buf.toString('utf8', 0, Math.min(buf.length, 300));
          const m = preview.match(/^(POST|GET|PUT|DELETE) (\S+) HTTP/);
          if (m) {
            log({
              t: new Date().toISOString(),
              event: 'net_req_line',
              conn: connId,
              target,
              method: m[1],
              path: m[2],
              pid: process.pid,
            });
          }
        }
      } catch (_) {}
      return origWrite(chunk, ...rest);
    };
    socket.on('data', (chunk) => {
      try { if (chunk) read += chunk.length; } catch (_) {}
    });
    socket.once('close', () => {
      log({
        t: new Date().toISOString(),
        event: 'net_close',
        conn: connId,
        target,
        wrote,
        read,
        durMs: Date.now() - startedAt,
        pid: process.pid,
      });
    });
    return socket;
  }
  wrappedConnect.__cursorAutoLogHooked = true;
  net.connect = wrappedConnect;
  // net.createConnection is an alias; keep them in sync
  net.createConnection = wrappedConnect;
  log({ t: new Date().toISOString(), event: 'net_hook_installed', pid: process.pid });
}

installNetHook();

process.on('exit', (code) => {
  try {
    for (const f of openFinalizers) {
      try { f(); } catch (_) {}
    }
  } catch (_) {}
  try {
    fs.appendFileSync(LOG_PATH,
      JSON.stringify({ t: new Date().toISOString(), event: 'session_end', code, pid: process.pid, openConns: openFinalizers.size }) + '\n');
  } catch (_) {}
});
