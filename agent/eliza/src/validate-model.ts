/**
 * ONNX Model Validation Script
 * Loads pool_features.csv and validates predictions against actual labels
 */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { lpRebalancerModel } from './inference/model.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CSV_PATH = path.join(__dirname, '../../data/lp_rebalancer/features/pool_features.csv');

interface ValidationResult {
  totalSamples: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  threshold: number;
  confusionMatrix: {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
  };
}

function parseCSV(content: string): Record<string, any>[] {
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',');
  
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const row: Record<string, any> = {};
    headers.forEach((h, i) => {
      const val = values[i];
      row[h] = val === '' ? 0 : isNaN(Number(val)) ? val : Number(val);
    });
    return row;
  });
}

async function validateModel(): Promise<ValidationResult> {
  console.log('Loading model...');
  await lpRebalancerModel.initialize();
  
  console.log('Loading CSV...');
  const csvContent = fs.readFileSync(CSV_PATH, 'utf-8');
  const rows = parseCSV(csvContent);
  
  // Use last 20% as test set (time-series split)
  const testStartIdx = Math.floor(rows.length * 0.8);
  const testRows = rows.slice(testStartIdx);
  
  console.log(`Total rows: ${rows.length}, Test rows: ${testRows.length}`);
  
  const featureOrder = lpRebalancerModel.getFeatureOrder();
  const threshold = lpRebalancerModel.getConfig()?.inference.threshold ?? 0.9;
  
  let tp = 0, tn = 0, fp = 0, fn = 0;
  let processed = 0;
  
  for (const row of testRows) {
    const features: Record<string, number> = {};
    for (const f of featureOrder) {
      features[f] = row[f] ?? 0;
    }
    
    const result = await lpRebalancerModel.predict(features);
    const predicted = result.probability >= threshold ? 1 : 0;
    const actual = row.label;
    
    if (predicted === 1 && actual === 1) tp++;
    else if (predicted === 0 && actual === 0) tn++;
    else if (predicted === 1 && actual === 0) fp++;
    else if (predicted === 0 && actual === 1) fn++;
    
    processed++;
    if (processed % 200 === 0) {
      console.log(`Processed ${processed}/${testRows.length}...`);
    }
  }
  
  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = 2 * (precision * recall) / (precision + recall) || 0;
  
  return {
    totalSamples: testRows.length,
    accuracy,
    precision,
    recall,
    f1,
    threshold,
    confusionMatrix: { tp, tn, fp, fn }
  };
}

async function main() {
  console.log('=== ONNX Model Validation ===\n');
  
  const result = await validateModel();
  
  console.log('\n=== Results ===');
  console.log(`Samples: ${result.totalSamples}`);
  console.log(`Threshold: ${result.threshold}`);
  console.log(`Accuracy: ${(result.accuracy * 100).toFixed(2)}%`);
  console.log(`Precision: ${(result.precision * 100).toFixed(2)}%`);
  console.log(`Recall: ${(result.recall * 100).toFixed(2)}%`);
  console.log(`F1 Score: ${(result.f1 * 100).toFixed(2)}%`);
  console.log('\nConfusion Matrix:');
  console.log(`  TP: ${result.confusionMatrix.tp} | FP: ${result.confusionMatrix.fp}`);
  console.log(`  FN: ${result.confusionMatrix.fn} | TN: ${result.confusionMatrix.tn}`);
}

main().catch(console.error);

