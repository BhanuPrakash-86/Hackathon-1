import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Initialize Gemini Client
let geminiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI | null {
  if (!geminiClient && process.env.GEMINI_API_KEY) {
    geminiClient = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return geminiClient;
}

// Health check API
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'EngineGuard AI Diagnostics Server', timestamp: new Date().toISOString() });
});

// Helper to generate physics-based aero-PHM diagnostic
function generateRuleBasedDiagnosis(params: {
  engineId: number | string;
  currentCycle: number;
  maxCycles: number;
  rul: number;
  healthIndex: number;
  status: 'Nominal' | 'Warning' | 'Critical';
  criticalSensors: any;
  faultMode?: string;
  dataset?: string;
}) {
  const { engineId, currentCycle, maxCycles, rul, healthIndex, status, faultMode, dataset, criticalSensors } = params;
  const t30 = criticalSensors?.T30_HPC_Temp || '1590.2';
  const t50 = criticalSensors?.T50_EGT || '1405.6';
  const egtMargin = criticalSensors?.EGT_Margin_C ?? (healthIndex > 50 ? 45 : 12);
  const vib = criticalSensors?.Vibration_G ?? 1.2;

  return {
    summary: `Turbofan Unit #${engineId} (${dataset || 'C-MAPSS'}) at flight cycle ${currentCycle}/${maxCycles} indicates an estimated Remaining Useful Life (RUL) of ${rul} flight cycles with an overall Health Index of ${healthIndex}%.`,
    primaryDegradation: faultMode || 'High Pressure Compressor (HPC) blade tip clearance erosion & EGT margin loss.',
    severity: status,
    findings: [
      `Total temperature at HPC outlet (T30: ${t30}°R) and LPT outlet (T50/EGT: ${t50}°R) exhibit positive thermal creep indicative of aerodynamic efficiency loss.`,
      `EGT Margin remaining calculated at ${egtMargin}°C, showing accelerated thermal erosion.`,
      `Spool dynamic vibration amplitude registered at ${vib}g with static pressure Ps30 tracking degraded compression ratio.`
    ],
    recommendedActions: [
      status === 'Critical' 
        ? 'Immediate AOG (Aircraft On Ground) maintenance hold. Disallow further flight dispatches and schedule engine shop visit.'
        : status === 'Warning'
        ? 'Schedule boroscopic inspection of HPC stages 5–8 stator vanes within 15 flight cycles.'
        : 'Perform routine pre-flight visual and fluid leak check.',
      'Perform compressor detergent wash procedure to restore aerodynamic efficiency.',
      'Calibrate T50 exhaust gas thermocouple harness and review FADEC fault logs.',
      'Stage replacement Line Replaceable Unit (LRU) compressor module in local line inventory.'
    ],
    dispatchRecommendation: status === 'Critical' 
      ? 'GROUND IMMEDIATELY (AOG - NO-GO)' 
      : status === 'Warning' 
      ? 'RESTRICTED DISPATCH (ETOPs Revoked - Max 2 Cycles)' 
      : 'GO FOR FLIGHT DISPATCH',
    maintenanceWindowHours: Math.max(2, Math.round(rul * 1.8)),
    estimatedCostImpact: status === 'Critical' 
      ? '$185,000 (Overhaul / Unscheduled Module Swap)' 
      : status === 'Warning' 
      ? '$28,000 (Preventative Maintenance & Borescope)' 
      : '$2,500 (Routine Line Check)'
  };
}

// Diagnostics endpoint using Gemini with multi-model retry & robust fallback
app.post('/api/diagnostics', async (req, res) => {
  const { engineId, currentCycle, maxCycles, rul, healthIndex, status, criticalSensors, faultMode, dataset } = req.body;

  const fallbackReport = generateRuleBasedDiagnosis({
    engineId,
    currentCycle,
    maxCycles,
    rul,
    healthIndex,
    status: status || 'Nominal',
    criticalSensors,
    faultMode,
    dataset,
  });

  const ai = getGeminiClient();
  if (!ai) {
    return res.json({
      success: true,
      source: 'aero-phm-engine',
      analysis: fallbackReport,
    });
  }

  const prompt = `You are an expert Chief Aerospace Propulsion Engineer & Flight Safety Specialist analyzing telemetry data from a commercial turbofan engine (NASA C-MAPSS dataset simulation).

Telemetry Data:
- Engine Unit ID: #${engineId}
- Simulation Dataset: ${dataset}
- Current Flight Cycle: ${currentCycle} of ${maxCycles}
- Estimated Remaining Useful Life (RUL): ${rul} cycles
- Overall Health Index: ${healthIndex}%
- Engine Status: ${status}
- Known/Suspected Fault Mode: ${faultMode || 'Compressor Degradation'}
- Critical Sensor Readings & Anomalies:
${JSON.stringify(criticalSensors, null, 2)}

Provide an authoritative, detailed technical diagnostic report formatted strictly as a JSON object with the following keys:
- summary: string (2-3 sentences summarizing the condition and risk)
- primaryDegradation: string (Specific thermodynamic or mechanical degradation mechanism, e.g., HPC erosion, Fan tip rubbing, Combustor hotspotting, HPT blade creep)
- severity: "Nominal" | "Warning" | "Critical"
- findings: string[] (3-4 specific engineering observations citing sensor anomalies like T30, T50/EGT, Ps30, phi, etc.)
- recommendedActions: string[] (3-5 concrete aviation maintenance actions conforming to FAA/EASA standard procedures, e.g., borescope, compressor wash, fuel nozzle spray check, module replacement)
- dispatchRecommendation: string ("GO FOR FLIGHT DISPATCH" | "RESTRICTED DISPATCH (ETOPs Revoked)" | "GROUND IMMEDIATELY (AOG - NO-GO)")
- maintenanceWindowHours: number (estimated flight hours remaining before mandatory servicing)
- estimatedCostImpact: string (estimated dollar cost difference between preventative vs unplanned catastrophic failure overhaul)`;

  // Try candidate models with retry
  const modelsToTry = ['gemini-3.7-flash', 'gemini-3.1-flash-lite'];
  
  for (const modelName of modelsToTry) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            responseMimeType: 'application/json',
          },
        });

        const text = response.text || '{}';
        const parsed = JSON.parse(text);

        if (parsed.summary && parsed.findings) {
          return res.json({
            success: true,
            source: `gemini-ai (${modelName})`,
            analysis: parsed,
          });
        }
      } catch (err: any) {
        console.warn(`Gemini attempt failed (model: ${modelName}, attempt: ${attempt + 1}):`, err?.message || err);
        // Short backoff before next attempt
        await new Promise((resolve) => setTimeout(resolve, 600));
      }
    }
  }

  // Gracefully provide expert aero-PHM diagnostic if upstream API encounters demand spikes (503)
  return res.json({
    success: true,
    source: 'aero-phm-engine (backup)',
    analysis: fallbackReport,
  });
});

// Real-time Turbofan RUL Prediction Endpoint (Compliant with ML API schema)
const handlePredictRequest = async (req: express.Request, res: express.Response) => {
  try {
    const {
      op_setting_1, op_setting_2, op_setting_3,
      sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9,
      sensor_11, sensor_12, sensor_13, sensor_14, sensor_15,
      sensor_17, sensor_20, sensor_21
    } = req.body;

    // Check if external FastAPI/Flask model service is reachable on port 8000
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
        signal: AbortSignal.timeout(1200)
      });
      if (response.ok) {
        const data = await response.json();
        return res.json(data);
      }
    } catch {
      // Local port 8000 ML service not active; calculate physics-based RUL regression
    }

    const t30Val = Number(sensor_3) || 1585;
    const t50Val = Number(sensor_4) || 1400;
    const ps30Val = Number(sensor_11) || 47.4;

    const t30Deg = Math.max(0, Math.min(1, (t30Val - 1575) / (1625 - 1575)));
    const t50Deg = Math.max(0, Math.min(1, (t50Val - 1390) / (1440 - 1390)));
    const ps30Deg = Math.max(0, Math.min(1, (47.8 - ps30Val) / (47.8 - 46.8)));

    const compositeDeg = 0.4 * t30Deg + 0.35 * t50Deg + 0.25 * ps30Deg;
    const predicted_rul = Math.max(1, Math.round(195 * (1 - Math.pow(compositeDeg, 1.35))));
    const alert_threshold = 30;
    const alert = predicted_rul <= alert_threshold;

    return res.json({
      predicted_rul,
      alert,
      alert_threshold
    });
  } catch (error: any) {
    return res.status(500).json({ error: error.message || 'Prediction failed' });
  }
};

app.post('/predict', handlePredictRequest);
app.post('/api/predict', handlePredictRequest);

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`EngineGuard AI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
