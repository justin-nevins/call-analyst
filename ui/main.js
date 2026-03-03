const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 500,
    backgroundColor: '#0b1120',
    titleBarStyle: 'default',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPython() {
  const projectDir = path.resolve(__dirname, '..');
  const venvPython = path.join(projectDir, 'venv', 'bin', 'python');

  pythonProcess = spawn(venvPython, ['call_analyst.py'], {
    cwd: projectDir,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[python] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[python] ${data.toString().trim()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null;
  });
}

function stopPython() {
  if (pythonProcess) {
    pythonProcess.kill('SIGINT');
    setTimeout(() => {
      if (pythonProcess) pythonProcess.kill('SIGKILL');
    }, 3000);
  }
}

app.on('ready', () => {
  startPython();
  // Give Python a moment to start the WebSocket server
  setTimeout(createWindow, 1500);
});

app.on('window-all-closed', () => {
  stopPython();
  app.quit();
});

app.on('before-quit', stopPython);
