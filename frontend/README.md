# Telco Churn Frontend (Vite + React)

This React app sends customer details to the FastAPI `/predict` endpoint and displays churn probability and a simple risk band.

## Setup

```bash
# "brew install node" if Node.js that includes npm is not installed
cd frontend
npm install
```

## Run locally

```bash
npm run dev
```

Visit the dev server URL printed in the terminal (default `http://localhost:5173`).

## Build for production

```bash
npm run build
```

## Configure API URL

Set `VITE_API_BASE_URL` in a `.env` file at the project root (same directory as `package.json`). Example:

```
VITE_API_BASE_URL=https://your-container-app-url
```

## Deploy to Azure Static Web Apps (high-level)

1. Create a Static Web App resource.
2. Build command: `npm run build`. Output folder: `dist`.
3. Set environment variable `VITE_API_BASE_URL` in the Static Web App configuration to point to your FastAPI Container App.
