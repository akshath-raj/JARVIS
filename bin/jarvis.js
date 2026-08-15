#!/usr/bin/env node
'use strict';

/*
 * `jarvis` — the Node launcher for the JARVIS voice assistant.
 *
 * JARVIS itself is a Python app (voice pipeline + agent brain). This thin CLI
 * bundles that Python source inside an npm package and, on first run, builds an
 * isolated Python virtualenv, installs the requirements, and sets up a user-owned
 * config file — so a user only needs `npm install -g` (or `npx`) and one command.
 *
 * User-owned state lives under ~/.jarvis/ so it survives package reinstalls:
 *   ~/.jarvis/.env             your API keys + settings (copied from .env.example)
 *   ~/.jarvis/runtime/venv     the Python environment
 * The Python source is read from the installed package directory.
 */

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

// ---------------------------------------------------------------- paths -----
const PKG_ROOT = path.resolve(__dirname, '..');
const HOME = os.homedir();
const JARVIS_HOME = path.join(HOME, '.jarvis');
const RUNTIME = path.join(JARVIS_HOME, 'runtime');
const VENV = path.join(RUNTIME, 'venv');
const ENV_FILE = path.join(JARVIS_HOME, '.env');
const REQ = path.join(PKG_ROOT, 'requirements.txt');
const ENV_EXAMPLE = path.join(PKG_ROOT, '.env.example');
const DEPS_STAMP = path.join(RUNTIME, '.deps-stamp');

const IS_MAC = process.platform === 'darwin';
const venvBin = IS_MAC || process.platform !== 'win32'
  ? path.join(VENV, 'bin')
  : path.join(VENV, 'Scripts');
const venvPython = path.join(venvBin, IS_MAC ? 'python' : (process.platform === 'win32' ? 'python.exe' : 'python'));

// ---------------------------------------------------------------- output ----
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const c = (code, s) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : s);
const cyan = (s) => c('36', s);
const green = (s) => c('32', s);
const yellow = (s) => c('33', s);
const red = (s) => c('31', s);
const dim = (s) => c('2', s);
const bold = (s) => c('1', s);

const log = (s = '') => console.log(s);
const step = (s) => log(cyan('▸ ') + s);
const ok = (s) => log(green('✓ ') + s);
const warn = (s) => log(yellow('! ') + s);
const fail = (s) => log(red('✗ ') + s);

function banner() {
  log();
  log(cyan('   ┌─────────────────────────────────────────┐'));
  log(cyan('   │   ') + bold('J A R V I S') + cyan('   voice assistant        │'));
  log(cyan('   └─────────────────────────────────────────┘'));
  log();
}

function die(msg) {
  fail(msg);
  process.exit(1);
}

// ------------------------------------------------------------- utilities ----
function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (res.error) throw res.error;
  return res.status === 0;
}

function capture(cmd, args) {
  const res = spawnSync(cmd, args, { encoding: 'utf8' });
  if (res.status !== 0 || res.error) return null;
  return (res.stdout || '').trim();
}

function commandExists(cmd) {
  const which = process.platform === 'win32' ? 'where' : 'which';
  const res = spawnSync(which, [cmd], { encoding: 'utf8' });
  return res.status === 0;
}

// Find a usable python3 (>= 3.11) on PATH.
function findPython() {
  const candidates = ['python3.13', 'python3.12', 'python3.11', 'python3', 'python'];
  for (const cmd of candidates) {
    const out = capture(cmd, ['-c', 'import sys;print("%d.%d" % sys.version_info[:2])']);
    if (!out) continue;
    const [maj, min] = out.split('.').map(Number);
    if (maj === 3 && min >= 11) return { cmd, version: out };
  }
  return null;
}

function fileHash(p) {
  try {
    return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  } catch {
    return '';
  }
}

// --------------------------------------------------------------- setup ------
function ensureDirs() {
  fs.mkdirSync(RUNTIME, { recursive: true });
}

function ensureVenv() {
  if (fs.existsSync(venvPython)) return true;
  const py = findPython();
  if (!py) {
    fail('No suitable Python found (need Python 3.11+).');
    log(dim('  Install it with:  ') + bold('brew install python'));
    return false;
  }
  step(`Creating Python environment (${py.cmd} ${py.version}) …`);
  if (!run(py.cmd, ['-m', 'venv', VENV])) {
    fail('Failed to create the virtualenv.');
    return false;
  }
  ok('Python environment created at ' + dim(VENV));
  return true;
}

function installDeps({ force = false } = {}) {
  const want = fileHash(REQ);
  const have = fs.existsSync(DEPS_STAMP) ? fs.readFileSync(DEPS_STAMP, 'utf8').trim() : '';
  if (!force && want && want === have && fs.existsSync(venvPython)) {
    return true; // already up to date
  }
  step('Installing Python dependencies (first run can take a few minutes) …');
  run(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip', '--quiet']);
  if (!run(venvPython, ['-m', 'pip', 'install', '-r', REQ])) {
    fail('Dependency install failed. See the pip output above.');
    return false;
  }
  fs.writeFileSync(DEPS_STAMP, want);
  ok('Dependencies installed.');
  return true;
}

function ensureEnvFile() {
  if (fs.existsSync(ENV_FILE)) return true;
  if (!fs.existsSync(ENV_EXAMPLE)) {
    warn('.env.example not found in the package; skipping config template.');
    return false;
  }
  fs.copyFileSync(ENV_EXAMPLE, ENV_FILE);
  ok('Created your config at ' + dim(ENV_FILE));
  return true;
}

// Parse ~/.jarvis/.env into a plain object (naive KEY=VALUE reader).
function readEnv() {
  const out = {};
  if (!fs.existsSync(ENV_FILE)) return out;
  for (const line of fs.readFileSync(ENV_FILE, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !line.trim().startsWith('#')) out[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
  return out;
}

// Which keys are needed depends on JARVIS_MODE (cloud default) + brain provider.
function missingKeys(env) {
  const missing = [];
  const mode = env.JARVIS_MODE || '1';
  const provider = (env.JARVIS_AGENT_PROVIDER || 'cerebras').toLowerCase();
  if (mode === '1') {
    if (!env.DEEPGRAM_API_KEY) missing.push(['DEEPGRAM_API_KEY', 'speech-to-text + text-to-speech']);
  }
  const brainKey = provider === 'cerebras'
    ? (env.CEREBRAS_API_KEY || env.OPENAI_API_KEY)
    : env.OPENAI_API_KEY;
  if (!brainKey) {
    missing.push([provider === 'cerebras' ? 'CEREBRAS_API_KEY' : 'OPENAI_API_KEY', 'the agent brain']);
  }
  return missing;
}

// --------------------------------------------------------------- checks -----
function checkOllama() {
  if (!commandExists('ollama')) {
    warn('Ollama not found — document search & long-term memory need it.');
    log(dim('  Install:  ') + bold('brew install ollama') + dim('  then  ') + bold('brew services start ollama'));
    return;
  }
  // Is the embedding model present?
  const tags = capture('ollama', ['list']) || '';
  if (!/nomic-embed-text/.test(tags)) {
    step('Pulling the local embedding model (nomic-embed-text) …');
    run('ollama', ['pull', 'nomic-embed-text']);
  }
}

// --------------------------------------------------------------- commands ---
function cmdSetup() {
  banner();
  if (!IS_MAC) {
    warn('JARVIS is built for macOS. Setup will run, but most features (music, browser, screen, focus) rely on macOS APIs and will not work here.');
  }
  ensureDirs();
  if (!ensureVenv()) process.exit(1);
  if (!installDeps()) process.exit(1);
  ensureEnvFile();
  checkOllama();
  log();
  const env = readEnv();
  const missing = missingKeys(env);
  if (missing.length) {
    warn('Before starting, add your API key(s) to ' + bold('~/.jarvis/.env') + ':');
    for (const [k, what] of missing) log('    ' + bold(k) + dim('  — ' + what));
    log();
    log('  Edit it with:  ' + bold('jarvis config'));
    log('  Get keys:      Cerebras ' + dim('cloud.cerebras.ai') + ' · Deepgram ' + dim('console.deepgram.com'));
  } else {
    ok('Config looks complete.');
  }
  log();
  ok('Setup done. Start JARVIS with:  ' + bold('jarvis'));
  log();
}

function ensureReady() {
  ensureDirs();
  if (!ensureVenv()) process.exit(1);
  if (!installDeps()) process.exit(1);
  ensureEnvFile();
}

function cmdStart(extraArgs) {
  ensureReady();
  const env = readEnv();
  const missing = missingKeys(env);
  if (missing.length) {
    banner();
    warn('Missing API key(s) in ' + bold('~/.jarvis/.env') + ':');
    for (const [k, what] of missing) log('    ' + bold(k) + dim('  — ' + what));
    log();
    log('  Add them with:  ' + bold('jarvis config') + '   then run ' + bold('jarvis') + ' again.');
    log(dim('  (Running fully on-device? Set JARVIS_MODE=0 and it needs no cloud keys.)'));
    log();
    process.exit(1);
  }

  // Best-effort: make sure Ollama is running (embeddings/memory).
  if (commandExists('ollama')) {
    const up = spawnSync('curl', ['-sf', '--max-time', '1', 'http://localhost:11434/api/tags'], { stdio: 'ignore' });
    if (up.status !== 0) {
      spawnSync('sh', ['-c', 'ollama serve >/dev/null 2>&1 &']);
    }
  }

  banner();
  ok('Launching the voice console. Say ' + bold('"Hey Jarvis"') + '.  ' + dim('Ctrl-C to stop.'));
  log();

  const child = spawn(venvPython, ['-m', 'jarvis.agent', 'console', ...extraArgs], {
    cwd: PKG_ROOT,
    stdio: 'inherit',
    env: { ...process.env, JARVIS_ENV_FILE: ENV_FILE, PYTHONPATH: PKG_ROOT },
  });
  child.on('exit', (code) => process.exit(code == null ? 0 : code));
}

function cmdConfig() {
  ensureDirs();
  ensureEnvFile();
  const editor = process.env.EDITOR;
  if (editor) {
    run(editor, [ENV_FILE]);
  } else if (IS_MAC) {
    run('open', ['-e', ENV_FILE]); // TextEdit
  } else {
    log('Edit this file to add your keys:  ' + bold(ENV_FILE));
  }
  log(dim('Config file: ') + ENV_FILE);
}

function cmdDoctor() {
  banner();
  log(bold('Environment check'));
  log();
  const py = findPython();
  py ? ok(`Python ${py.version} (${py.cmd})`) : fail('Python 3.11+ not found — brew install python');
  commandExists('node') ? ok('Node ' + (capture('node', ['--version']) || '')) : warn('node not found');
  IS_MAC ? ok('Platform: macOS') : warn(`Platform: ${process.platform} (JARVIS targets macOS)`);
  commandExists('ollama') ? ok('Ollama installed') : warn('Ollama missing — brew install ollama');
  commandExists('ffmpeg') ? ok('ffmpeg installed') : warn('ffmpeg missing — brew install ffmpeg (needed for audio)');
  fs.existsSync(venvPython) ? ok('Python env ready (' + dim(VENV) + ')') : warn('Python env not built yet — run: jarvis setup');
  fs.existsSync(ENV_FILE) ? ok('Config present (' + dim(ENV_FILE) + ')') : warn('No config yet — run: jarvis setup');

  if (fs.existsSync(ENV_FILE)) {
    const env = readEnv();
    const missing = missingKeys(env);
    if (missing.length === 0) ok('API keys present for the current mode');
    else {
      warn('Missing keys: ' + missing.map(([k]) => k).join(', '));
      log(dim('    add them with: jarvis config'));
    }
  }
  log();
}

function cmdUpdate() {
  banner();
  ensureDirs();
  if (!ensureVenv()) process.exit(1);
  installDeps({ force: true });
  checkOllama();
  ok('Up to date.');
}

function cmdPostinstall() {
  // Runs during `npm install`. Keep it light — no heavy pip here.
  if (!IS_MAC) {
    log(yellow('\nNote: JARVIS targets macOS. It installed, but device features need a Mac.\n'));
  }
  log();
  log(green('JARVIS installed.') + '  Finish setup with:');
  log('    ' + bold('jarvis setup') + dim('   # build the Python env + create your config'));
  log('    ' + bold('jarvis config') + dim('  # paste your API keys'));
  log('    ' + bold('jarvis') + dim('         # start the voice assistant'));
  log();
}

function cmdHelp() {
  banner();
  log(bold('Usage:') + '  jarvis <command>');
  log();
  log('  ' + bold('jarvis') + '            Start the voice assistant (runs setup first if needed)');
  log('  ' + bold('jarvis setup') + '      Build the Python env, install deps, create your config');
  log('  ' + bold('jarvis config') + '     Open ~/.jarvis/.env to add your API keys');
  log('  ' + bold('jarvis doctor') + '     Check prerequisites and configuration');
  log('  ' + bold('jarvis update') + '     Reinstall Python dependencies (after upgrading)');
  log('  ' + bold('jarvis help') + '       Show this help');
  log();
  log(dim('State lives in ~/.jarvis/ (config + Python env), so it survives reinstalls.'));
  log(dim('Full docs: https://github.com/akshath-raj/JARVIS#readme'));
  log();
}

// ----------------------------------------------------------------- main -----
function main() {
  const [, , cmd, ...rest] = process.argv;
  switch (cmd) {
    case undefined:
    case 'start':
      return cmdStart(rest);
    case 'setup':
    case 'install':
      return cmdSetup();
    case 'config':
      return cmdConfig();
    case 'doctor':
      return cmdDoctor();
    case 'update':
    case 'upgrade':
      return cmdUpdate();
    case 'postinstall':
      return cmdPostinstall();
    case 'version':
    case '--version':
    case '-v': {
      const p = require(path.join(PKG_ROOT, 'package.json'));
      return log('jarvis ' + p.version);
    }
    case 'help':
    case '--help':
    case '-h':
      return cmdHelp();
    default:
      fail('Unknown command: ' + cmd);
      cmdHelp();
      process.exit(1);
  }
}

main();
