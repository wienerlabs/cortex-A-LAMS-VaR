/**
 * Retraining Executor
 *
 * Executes model retraining by:
 * 1. Collecting fresh training data
 * 2. Running the appropriate training script
 * 3. Exporting to ONNX format
 * 4. Running calibration
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { ModelMetrics } from './modelRegistry.js';

const execAsync = promisify(exec);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============= TYPES =============

export interface RetrainingResult {
  success: boolean;
  modelPath: string;
  metadataPath: string;
  calibrationPath?: string;
  metrics: ModelMetrics;
  version: string;
  trainedAt: Date;
  trainingDurationMs: number;
  error?: string;
}

export interface TrainingScriptResult {
  success: boolean;
  modelPath: string;
  metadataPath: string;
  metrics: ModelMetrics;
  stdout: string;
  stderr: string;
}

export interface ExecutorConfig {
  pythonPath: string;
  scriptsDir: string;
  modelsDir: string;
  dataDir: string;
  timeoutMs: number;
}

// ============= CONSTANTS =============

const AGENT_ROOT = path.resolve(__dirname, '../../../../');
const ELIZA_ROOT = path.resolve(__dirname, '../../../');

const DEFAULT_CONFIG: ExecutorConfig = {
  pythonPath: 'python3',
  scriptsDir: path.join(AGENT_ROOT, 'scripts'),
  modelsDir: path.join(ELIZA_ROOT, 'models'),
  dataDir: path.join(AGENT_ROOT, 'data'),
  timeoutMs: 30 * 60 * 1000,  // 30 minutes
};

// Training script mapping
const TRAINING_SCRIPTS: Record<string, string> = {
  'perps': 'train_perps_model.py',
  'spot': 'train_spot_model.py',
  'lp': 'train_lp_model.py',
  'lending': 'train_lending_model.py',
  'arbitrage': 'train_arbitrage_model.py',
};

// Data file mapping
const DATA_FILES: Record<string, string> = {
  'perps': 'perps/funding_rates.csv',
  'spot': 'spot/spot_training_data.csv',
  'lp': 'lp/lp_historical.csv',
  'lending': 'lending/lending_historical.csv',
  'arbitrage': 'arbitrage/cross_dex_data.csv',
};

// ============= RETRAINING EXECUTOR CLASS =============

class RetrainingExecutor {
  private config: ExecutorConfig;

  constructor(config: Partial<ExecutorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[RetrainingExecutor] Initialized', {
      scriptsDir: this.config.scriptsDir,
      modelsDir: this.config.modelsDir,
    });
  }

  /**
   * Execute full retraining pipeline for a model
   */
  async retrain(modelName: string): Promise<RetrainingResult> {
    const startTime = Date.now();
    logger.info('[RetrainingExecutor] Starting retraining', { modelName });

    try {
      // 1. Verify training script exists
      const scriptPath = this.getTrainingScriptPath(modelName);
      if (!fs.existsSync(scriptPath)) {
        throw new Error(`Training script not found: ${scriptPath}`);
      }

      // 2. Collect/verify training data
      const dataPath = await this.collectData(modelName);
      
      // 3. Run training script
      const trainResult = await this.runTrainingScript(modelName, dataPath);
      
      if (!trainResult.success) {
        throw new Error(`Training failed: ${trainResult.stderr}`);
      }

      // 4. Export to ONNX (if not already done by script)
      const onnxPath = await this.ensureONNXExport(modelName, trainResult.modelPath);

      // 5. Run calibration
      const calibrationPath = await this.runCalibration(modelName, onnxPath);

      // 6. Generate version
      const version = this.generateVersion();

      const result: RetrainingResult = {
        success: true,
        modelPath: onnxPath,
        metadataPath: trainResult.metadataPath,
        calibrationPath,
        metrics: trainResult.metrics,
        version,
        trainedAt: new Date(),
        trainingDurationMs: Date.now() - startTime,
      };

      logger.info('[RetrainingExecutor] Retraining completed', {
        modelName,
        version,
        durationMs: result.trainingDurationMs,
        metrics: result.metrics,
      });

      return result;

    } catch (error) {
      logger.error('[RetrainingExecutor] Retraining failed', {
        modelName,
        error: String(error),
      });

      return {
        success: false,
        modelPath: '',
        metadataPath: '',
        metrics: { precision: 0, recall: 0, f1Score: 0, rocAuc: 0 },
        version: '',
        trainedAt: new Date(),
        trainingDurationMs: Date.now() - startTime,
        error: String(error),
      };
    }
  }

  /**
   * Get training script path for a model
   */
  private getTrainingScriptPath(modelName: string): string {
    const scriptName = TRAINING_SCRIPTS[modelName];
    if (!scriptName) {
      throw new Error(`No training script configured for model: ${modelName}`);
    }
    return path.join(this.config.scriptsDir, scriptName);
  }

  /**
   * Collect or verify training data exists
   */
  private async collectData(modelName: string): Promise<string> {
    const dataFile = DATA_FILES[modelName];
    if (!dataFile) {
      throw new Error(`No data file configured for model: ${modelName}`);
    }

    const dataPath = path.join(this.config.dataDir, dataFile);

    // Check if data exists and is recent
    if (fs.existsSync(dataPath)) {
      const stats = fs.statSync(dataPath);
      const ageHours = (Date.now() - stats.mtimeMs) / (1000 * 60 * 60);

      if (ageHours < 24) {
        logger.info('[RetrainingExecutor] Using existing data', { dataPath, ageHours });
        return dataPath;
      }
    }

    // Try to collect fresh data
    logger.info('[RetrainingExecutor] Collecting fresh data', { modelName });

    const collectorScript = `collect_${modelName}_data.py`;
    const collectorPath = path.join(this.config.scriptsDir, collectorScript);

    if (fs.existsSync(collectorPath)) {
      try {
        await execAsync(
          `${this.config.pythonPath} ${collectorPath} --output ${dataPath}`,
          { timeout: this.config.timeoutMs / 2 }
        );
        logger.info('[RetrainingExecutor] Data collection completed', { dataPath });
      } catch (error) {
        logger.warn('[RetrainingExecutor] Data collection failed, using existing', { error: String(error) });
      }
    }

    if (!fs.existsSync(dataPath)) {
      throw new Error(`Training data not found: ${dataPath}`);
    }

    return dataPath;
  }

  /**
   * Run the training script
   */
  private async runTrainingScript(modelName: string, dataPath: string): Promise<TrainingScriptResult> {
    const scriptPath = this.getTrainingScriptPath(modelName);
    const outputDir = path.join(this.config.modelsDir, modelName);

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const command = `${this.config.pythonPath} ${scriptPath} --data ${dataPath} --output ${outputDir}`;

    logger.info('[RetrainingExecutor] Running training script', { command });

    try {
      const { stdout, stderr } = await execAsync(command, {
        timeout: this.config.timeoutMs,
        cwd: AGENT_ROOT,
      });

      // Parse metrics from stdout (expect JSON on last line)
      const metrics = this.parseMetricsFromOutput(stdout);

      // Find generated model files
      const modelPath = this.findLatestModel(outputDir);
      const metadataPath = this.findLatestMetadata(outputDir);

      return {
        success: true,
        modelPath,
        metadataPath,
        metrics,
        stdout,
        stderr,
      };

    } catch (error: any) {
      return {
        success: false,
        modelPath: '',
        metadataPath: '',
        metrics: { precision: 0, recall: 0, f1Score: 0, rocAuc: 0 },
        stdout: error.stdout || '',
        stderr: error.stderr || String(error),
      };
    }
  }

  /**
   * Parse metrics from training script output
   */
  private parseMetricsFromOutput(stdout: string): ModelMetrics {
    const lines = stdout.trim().split('\n');

    // Try to find JSON metrics in output
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith('{') && line.endsWith('}')) {
        try {
          const parsed = JSON.parse(line);
          return {
            precision: parsed.precision || parsed.test_precision || 0,
            recall: parsed.recall || parsed.test_recall || 0,
            f1Score: parsed.f1_score || parsed.f1 || 0,
            rocAuc: parsed.roc_auc || parsed.auc || 0,
            sharpe: parsed.sharpe_ratio || parsed.sharpe,
          };
        } catch {
          continue;
        }
      }
    }

    // Default metrics if parsing fails
    logger.warn('[RetrainingExecutor] Could not parse metrics from output');
    return { precision: 0, recall: 0, f1Score: 0, rocAuc: 0 };
  }

  /**
   * Find the latest model file in directory
   */
  private findLatestModel(dir: string): string {
    const files = fs.readdirSync(dir)
      .filter(f => f.endsWith('.onnx') || f.endsWith('.pkl') || f.endsWith('.joblib'))
      .map(f => ({
        name: f,
        path: path.join(dir, f),
        mtime: fs.statSync(path.join(dir, f)).mtimeMs,
      }))
      .sort((a, b) => b.mtime - a.mtime);

    if (files.length === 0) {
      throw new Error(`No model files found in ${dir}`);
    }

    return files[0].path;
  }

  /**
   * Find the latest metadata file in directory
   */
  private findLatestMetadata(dir: string): string {
    const metadataDir = path.join(dir, 'metadata');
    const searchDirs = [metadataDir, dir];

    for (const searchDir of searchDirs) {
      if (!fs.existsSync(searchDir)) continue;

      const files = fs.readdirSync(searchDir)
        .filter(f => f.endsWith('_metadata.json') || f.endsWith('_config.json'))
        .map(f => ({
          name: f,
          path: path.join(searchDir, f),
          mtime: fs.statSync(path.join(searchDir, f)).mtimeMs,
        }))
        .sort((a, b) => b.mtime - a.mtime);

      if (files.length > 0) {
        return files[0].path;
      }
    }

    // Create minimal metadata if none exists
    const metadataPath = path.join(dir, 'training_metadata.json');
    fs.writeFileSync(metadataPath, JSON.stringify({
      trained_at: new Date().toISOString(),
      model_name: path.basename(dir),
    }));
    return metadataPath;
  }

  /**
   * Ensure model is exported to ONNX format
   */
  private async ensureONNXExport(_modelName: string, modelPath: string): Promise<string> {
    if (modelPath.endsWith('.onnx')) {
      return modelPath;
    }

    // Run ONNX export script
    const exportScript = path.join(this.config.scriptsDir, 'export_onnx.py');
    const onnxPath = modelPath.replace(/\.(pkl|joblib)$/, '.onnx');

    if (fs.existsSync(exportScript)) {
      try {
        await execAsync(
          `${this.config.pythonPath} ${exportScript} --input ${modelPath} --output ${onnxPath}`,
          { timeout: 60000 }
        );
        logger.info('[RetrainingExecutor] ONNX export completed', { onnxPath });
        return onnxPath;
      } catch (error) {
        logger.warn('[RetrainingExecutor] ONNX export failed', { error: String(error) });
      }
    }

    return modelPath;
  }

  /**
   * Run calibration on the model
   */
  private async runCalibration(_modelName: string, modelPath: string): Promise<string | undefined> {
    const calibrationScript = path.join(this.config.scriptsDir, 'calibrate_model.py');
    const calibrationPath = modelPath.replace('.onnx', '_calibration.json');

    if (!fs.existsSync(calibrationScript)) {
      logger.debug('[RetrainingExecutor] No calibration script found');
      return undefined;
    }

    try {
      await execAsync(
        `${this.config.pythonPath} ${calibrationScript} --model ${modelPath} --output ${calibrationPath}`,
        { timeout: 60000 }
      );
      logger.info('[RetrainingExecutor] Calibration completed', { calibrationPath });
      return calibrationPath;
    } catch (error) {
      logger.warn('[RetrainingExecutor] Calibration failed', { error: String(error) });
      return undefined;
    }
  }

  /**
   * Generate a version string
   */
  private generateVersion(): string {
    const now = new Date();
    const major = 1;
    const minor = Math.floor((now.getTime() - new Date('2024-01-01').getTime()) / (30 * 24 * 60 * 60 * 1000));
    const patch = now.getDate();
    return `${major}.${minor}.${patch}`;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ExecutorConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('[RetrainingExecutor] Config updated', { config: this.config });
  }
}

// ============= SINGLETON =============

let executorInstance: RetrainingExecutor | null = null;

export function getRetrainingExecutor(config?: Partial<ExecutorConfig>): RetrainingExecutor {
  if (!executorInstance) {
    executorInstance = new RetrainingExecutor(config);
  }
  return executorInstance;
}

export const retrainingExecutor = getRetrainingExecutor();

export { RetrainingExecutor };

