import { USE_MOCK } from '../config/env';
import { post } from './apiClient';
import type { CreateIterationRequest, IterationCreateResponse } from '../types';

function delay<T>(value: T, ms = 240): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

export async function createIteration(req: CreateIterationRequest): Promise<IterationCreateResponse> {
  if (USE_MOCK) {
    return delay({
      iterationId: req.iterationId,
      status: 'created',
      dataDir: `data/iterations/${req.iterationId}`,
      outputDir: `outputs/iterations/${req.iterationId}`,
      allSamplesCsv: `data/iterations/${req.iterationId}/all_samples.csv`,
      splitSummary: {
        allInput: 134920,
        seenForSplit: 119800,
        trainBase: 105600,
        trainHard: 10240,
        trainStage1: req.trainPlan === 'hard_in_stage1' ? 115840 : 105600,
        trainStage2: 115840,
        trainStage3Initial: 115840,
        val: req.valRealTotal + req.valFakeTotal,
        testUnseen: 0,
        testAll: req.valRealTotal + req.valFakeTotal,
        reviewedPool: 4880,
        outputDir: `data/iterations/${req.iterationId}`,
        files: {
          trainBaseCsv: `data/iterations/${req.iterationId}/train_base.csv`,
          trainHardCsv: `data/iterations/${req.iterationId}/train_hard.csv`,
          trainStage1Csv: `data/iterations/${req.iterationId}/train_stage1.csv`,
          trainStage2Csv: `data/iterations/${req.iterationId}/train_stage2.csv`,
          trainStage3Csv: `data/iterations/${req.iterationId}/train_stage3.csv`,
          valCsv: `data/iterations/${req.iterationId}/val.csv`,
          testUnseenCsv: `data/iterations/${req.iterationId}/test_unseen.csv`,
          testAllCsv: `data/iterations/${req.iterationId}/test_all.csv`,
          reviewedPoolCsv: `data/iterations/${req.iterationId}/reviewed_pool.csv`,
        },
        warnings: ['当前没有 split_hint=unseen 的数据，本轮不会生成泛化测试集。'],
      },
      configPath: `outputs/iterations/${req.iterationId}/iteration_config.json`,
    });
  }
  return post<CreateIterationRequest, IterationCreateResponse>('/api/iterations', req);
}
