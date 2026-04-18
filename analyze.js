#!/usr/bin/env node
// Summarise a cursor-auto.log produced by logger.js.
//
// Usage:
//   node analyze.js [path/to/cursor-auto.log]
//
// What we look for:
//   1. GetDefaultModelForCli    → what the backend picks for `--model auto`.
//   2. GetUsableModels          → the catalogue of models the user can select.
//   3. Telemetry (TrackEvents/SubmitLogs) that reports the *actual* model used
//      for the turn (if present).
//   4. Any chat RPC we do see on the wire.
//
// Notes:
//   The actual chat streaming RPC (StreamUnifiedChatWithTools) is issued from
//   a transport layer that bypasses Node's TLS module, so its wire bytes do
//   not show up in this log. The auto-model *decision* is visible in
//   GetDefaultModelForCli.

'use strict';

const fs = require('node:fs');
const path = require('node:path');

const LOG = process.argv[2] || path.join(process.cwd(), 'cursor-auto.log');
if (!fs.existsSync(LOG)) { console.error(`no log file at ${LOG}`); process.exit(1); }

const lines = fs.readFileSync(LOG, 'utf8').split('\n').filter(Boolean);
const events = [];
for (const line of lines) { try { events.push(JSON.parse(line)); } catch {} }

const http = events.filter(e => e.event === 'http');

function modelsIn(scan) {
  if (!scan || typeof scan !== 'object') return [];
  const a = Array.isArray(scan.modelHits) ? scan.modelHits : [];
  return a;
}

function samplesOf(scan) {
  if (!scan || typeof scan !== 'object') return [];
  return Array.isArray(scan.sampleStrings) ? scan.sampleStrings : [];
}

console.log(`# cursor-agent auto-model summary`);
console.log(`# source: ${LOG}`);
console.log(`# total events: ${events.length}   http: ${http.length}`);
console.log('');

// 1. Auto default pick (GetDefaultModelForCli)
const defaultPicks = http.filter(e => String(e.reqPath || '').endsWith('/GetDefaultModelForCli'));
console.log('## auto-mode default pick');
console.log(`   endpoint: /aiserver.v1.AiService/GetDefaultModelForCli`);
if (defaultPicks.length === 0) {
  console.log('   (none captured in this log)');
} else {
  for (const e of defaultPicks) {
    const models = modelsIn(e.respScan);
    console.log(`   ${e.t}   default=${models.join(',') || '?'}   status=${e.respStatus}   durMs=${e.durMs}`);
  }
}
console.log('');

// 2. Usable models catalogue
const usable = http.filter(e => String(e.reqPath || '').endsWith('/GetUsableModels'));
console.log('## usable models catalogue (sent from GetUsableModels)');
if (usable.length === 0) {
  console.log('   (none captured in this log)');
} else {
  const latest = usable[usable.length - 1];
  const models = modelsIn(latest.respScan);
  console.log(`   ${latest.t}   ${models.length} models:`);
  for (const m of models) console.log(`     - ${m}`);
}
console.log('');

// 3. Telemetry that records the actual model used
const telemetry = http.filter(e => /TrackEvents|SubmitLogs/.test(String(e.reqPath || '')));
console.log('## telemetry events (may reveal the actual model used per turn)');
if (telemetry.length === 0) {
  console.log('   (none)');
} else {
  const modelHints = new Set();
  for (const e of telemetry) {
    for (const s of samplesOf(e.reqScan)) {
      // Look for model-id-ish tokens inside the sampled payloads.
      const matches = s.match(/"(composer[\w\-]*|gpt-[\w\-]+|claude[\w\-.]+|gemini[\w\-.]+|grok[\w\-.]+|kimi[\w\-.]+)"/g) || [];
      for (const m of matches) modelHints.add(m.replace(/"/g, ''));
      const reqM = s.match(/"(requested|model|actual_model|model_used|ai_model|modelName)":\s*\{?\s*"?(?:stringValue"?:\s*)?"([^"]+)"/g) || [];
      for (const m of reqM) modelHints.add(m);
    }
  }
  if (modelHints.size === 0) {
    console.log('   captured but no model name leaked in telemetry bodies');
  } else {
    for (const h of modelHints) console.log(`   - ${h}`);
  }
}
console.log('');

// 4. All captured RPCs (for completeness)
console.log('## all captured RPC calls');
for (const e of http) {
  const p = (e.reqPath || '?').split('/').slice(-1)[0];
  const enc = e.respEncoding || '-';
  const rs = modelsIn(e.respScan);
  console.log(`   ${e.t}  ${String(p).padEnd(40)}  ${String(e.respStatus || '?').padStart(3)}  ${enc.padEnd(4)}  ${String(e.durMs || '').padStart(6)}ms  resp_models=${rs.slice(0, 6).join(',') || '-'}`);
}
console.log('');

// 5. Any chat-style endpoint (just in case)
const chatLike = http.filter(e => /StreamUnifiedChat|StreamChat|StreamComposer|StreamConversation/.test(String(e.reqPath || '')));
if (chatLike.length > 0) {
  console.log('## chat RPCs seen on the wire');
  for (const e of chatLike) {
    const reqM = modelsIn(e.reqScan);
    const respM = modelsIn(e.respScan);
    console.log(`   ${e.t}  ${e.reqPath}   req_models=${reqM.join(',')||'-'}   resp_models=${respM.join(',')||'-'}`);
  }
} else {
  console.log('## chat RPCs seen on the wire');
  console.log('   (none — the chat streaming RPC uses a transport that bypasses');
  console.log('    Node\'s tls.connect, so its wire bytes are not captured here.');
  console.log('    The auto-model decision is still visible via GetDefaultModelForCli.)');
}
