import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Radio,
  Row,
  Space,
  Steps,
  Typography,
  message,
} from 'antd';
import { CheckCircleOutlined, RocketOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DatasetSummaryCard from '../components/DatasetSummaryCard';
import ManifestBuilder from '../components/ManifestBuilder/ManifestBuilder';
import { mergeDatasetIndex, previewDatasetSplit } from '../services/dataset';
import { createIteration } from '../services/iteration';
import type { CreateIterationRequest, DatasetSplitSummary, MergeIndexResponse, TrainPlan } from '../types';

type IterationCreateFormValues = {
  iterationId: string;
  description?: string;
  seed: number;
  nproc: number;
  baseCsv: string;
  incrementManifest?: string;
  valRealTotal: number;
  valFakeTotal: number;
  trainPlan: TrainPlan;
  runStage1: boolean;
  runStage2: boolean;
  runReplay: boolean;
  runStage3: boolean;
  runEval: boolean;
};

type ManifestMode = 'existing' | 'manual';

const defaultValues: IterationCreateFormValues = {
  iterationId: 'iter_20260511_001',
  description: '新增合成图像数据准备任务',
  seed: 3407,
  nproc: 8,
  baseCsv: 'safepp_pytorch/manifests/base_index.csv',
  incrementManifest: 'safepp_pytorch/manifests/increment_manifest.yaml',
  valRealTotal: 3000,
  valFakeTotal: 3000,
  trainPlan: 'hard_in_stage2',
  runStage1: true,
  runStage2: true,
  runReplay: true,
  runStage3: true,
  runEval: true,
};

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '请求失败';
}

export default function NewIterationPage() {
  const [form] = Form.useForm<IterationCreateFormValues>();
  const navigate = useNavigate();
  const [current, setCurrent] = useState(0);
  const [manifestMode, setManifestMode] = useState<ManifestMode>('existing');
  const [mergeResult, setMergeResult] = useState<MergeIndexResponse>();
  const [splitSummary, setSplitSummary] = useState<DatasetSplitSummary>();
  const [error, setError] = useState<string>();
  const [checking, setChecking] = useState(false);
  const [creating, setCreating] = useState(false);
  const currentIterationId = Form.useWatch('iterationId', form) || defaultValues.iterationId;

  const runPreflight = async () => {
    setError(undefined);
    setChecking(true);
    try {
      const values = await form.validateFields();
      const merged = await mergeDatasetIndex({
        iterationId: values.iterationId,
        baseCsv: values.baseCsv,
        incrementManifest: values.incrementManifest,
      });
      const split = await previewDatasetSplit({
        iterationId: values.iterationId,
        inputCsv: merged.allSamplesCsv,
        trainPlan: values.trainPlan,
        valRealTotal: values.valRealTotal,
        valFakeTotal: values.valFakeTotal,
        seed: values.seed,
      });
      setMergeResult(merged);
      setSplitSummary(split);
      setCurrent(4);
      message.success('启动前检查完成');
    } catch (err) {
      const msg = getErrorMessage(err);
      setError(msg);
      message.error(msg);
    } finally {
      setChecking(false);
    }
  };

  const submitCreate = async () => {
    setError(undefined);
    setCreating(true);
    try {
      const values = await form.validateFields();
      const req: CreateIterationRequest = {
        iterationId: values.iterationId,
        description: values.description,
        baseCsv: values.baseCsv,
        incrementManifest: values.incrementManifest,
        trainPlan: values.trainPlan,
        seed: values.seed,
        nproc: values.nproc,
        valRealTotal: values.valRealTotal,
        valFakeTotal: values.valFakeTotal,
        runStage1: values.runStage1,
        runStage2: values.runStage2,
        runReplay: values.runReplay,
        runStage3: values.runStage3,
        runEval: values.runEval,
      };
      const resp = await createIteration(req);
      localStorage.setItem('safepp:lastTaskId', resp.iterationId);
      message.success('迭代任务已创建，训练阶段暂未启动');
      navigate(`/tasks/${resp.iterationId}`);
    } catch (err) {
      const msg = getErrorMessage(err);
      setError(msg);
      message.error(msg);
    } finally {
      setCreating(false);
    }
  };

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>新建迭代任务</Typography.Title>
          <Typography.Paragraph type="secondary">
            本页只完成数据准备、切分预检查和任务配置保存，不启动 Stage1 / Stage2 / Stage3。
          </Typography.Paragraph>
        </div>
      </div>

      {error ? <Alert type="error" showIcon message="接口调用失败" description={error} /> : null}

      <Card className="panel-card">
        <Steps
          current={current}
          items={[
            { title: '基础配置' },
            { title: '数据配置' },
            { title: '训练方案' },
            { title: '启动前检查' },
            { title: '创建任务' },
          ]}
          className="create-steps"
        />

        <Form form={form} layout="vertical" initialValues={defaultValues} className="stepped-form">
          <div className="step-section">
            <Typography.Title level={4}>Step 1 基础配置</Typography.Title>
            <Row gutter={16}>
              <Col xs={24} md={8}>
                <Form.Item name="iterationId" label="iterationId" rules={[{ required: true }]}>
                  <Input onFocus={() => setCurrent(0)} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item name="seed" label="seed" rules={[{ required: true }]}>
                  <InputNumber min={0} precision={0} className="full-width" onFocus={() => setCurrent(0)} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item name="nproc" label="nproc" rules={[{ required: true }]}>
                  <InputNumber min={1} max={128} precision={0} className="full-width" onFocus={() => setCurrent(0)} />
                </Form.Item>
              </Col>
              <Col xs={24}>
                <Form.Item name="description" label="description">
                  <Input.TextArea rows={2} onFocus={() => setCurrent(0)} />
                </Form.Item>
              </Col>
            </Row>
          </div>

          <div className="step-section">
            <Typography.Title level={4}>Step 2 数据配置</Typography.Title>
            <Form.Item label="Manifest 来源">
              <Radio.Group
                value={manifestMode}
                onChange={(event) => {
                  setManifestMode(event.target.value as ManifestMode);
                  setCurrent(1);
                }}
              >
                <Radio.Button value="existing">已有文件路径</Radio.Button>
                <Radio.Button value="manual">手动创建</Radio.Button>
              </Radio.Group>
            </Form.Item>
            <Row gutter={16}>
              <Col xs={24} md={12}>
                <Form.Item name="baseCsv" label="baseCsv" rules={[{ required: true }]}>
                  <Input onFocus={() => setCurrent(1)} />
                </Form.Item>
              </Col>
              {manifestMode === 'existing' ? (
                <Col xs={24} md={12}>
                  <Form.Item name="incrementManifest" label="incrementManifest">
                    <Input onFocus={() => setCurrent(1)} />
                  </Form.Item>
                </Col>
              ) : null}
              <Col xs={24} md={8}>
                <Form.Item name="valRealTotal" label="valRealTotal" rules={[{ required: true }]}>
                  <InputNumber min={1} precision={0} className="full-width" onFocus={() => setCurrent(1)} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item name="valFakeTotal" label="valFakeTotal" rules={[{ required: true }]}>
                  <InputNumber min={1} precision={0} className="full-width" onFocus={() => setCurrent(1)} />
                </Form.Item>
              </Col>
            </Row>
            {manifestMode === 'manual' ? (
              <ManifestBuilder
                iterationId={currentIterationId}
                value={form.getFieldValue('incrementManifest') as string | undefined}
                onManifestSaved={(manifestPath) => {
                  form.setFieldValue('incrementManifest', manifestPath);
                  setCurrent(1);
                }}
              />
            ) : null}
          </div>

          <div className="step-section">
            <Typography.Title level={4}>Step 3 训练方案</Typography.Title>
            <Form.Item name="trainPlan" label="trainPlan" rules={[{ required: true }]}>
              <Radio.Group className="plan-group" onChange={() => setCurrent(2)}>
                <Radio.Button value="hard_in_stage1">
                  hard_in_stage1：Stage1 = Base + Hard，Stage2 = Stage1，Stage3 = Stage2 + Replay
                </Radio.Button>
                <Radio.Button value="hard_in_stage2">
                  hard_in_stage2：Stage1 = Base，Stage2 = Base + Hard，Stage3 = Stage2 + Replay
                </Radio.Button>
              </Radio.Group>
            </Form.Item>
            <Row gutter={16}>
              {[
                ['runStage1', 'runStage1'],
                ['runStage2', 'runStage2'],
                ['runReplay', 'runReplay'],
                ['runStage3', 'runStage3'],
                ['runEval', 'runEval'],
              ].map(([name, label]) => (
                <Col xs={12} md={4} key={name}>
                  <Form.Item name={name} valuePropName="checked">
                    <Checkbox onChange={() => setCurrent(2)}>{label}</Checkbox>
                  </Form.Item>
                </Col>
              ))}
            </Row>
          </div>

          <div className="step-section">
            <Typography.Title level={4}>Step 4 启动前检查</Typography.Title>
            <Button type="primary" icon={<CheckCircleOutlined />} loading={checking} onClick={runPreflight}>
              启动前检查
            </Button>
            {mergeResult ? (
              <Descriptions column={1} size="small" className="preflight-desc">
                <Descriptions.Item label="allSamplesCsv">{mergeResult.allSamplesCsv}</Descriptions.Item>
                <Descriptions.Item label="totalRows">{mergeResult.totalRows}</Descriptions.Item>
                <Descriptions.Item label="duplicateRemoved">{mergeResult.duplicateRemoved}</Descriptions.Item>
              </Descriptions>
            ) : null}
            {splitSummary ? (
              <Space direction="vertical" size={12} className="page-stack preflight-summary">
                {splitSummary.testUnseen === 0 ? (
                  <Alert
                    type="warning"
                    showIcon
                    message="当前没有 split_hint=unseen 的数据，本轮不会生成泛化测试集。"
                  />
                ) : null}
                {splitSummary.warnings.length > 0 ? (
                  <Alert type="warning" showIcon message={splitSummary.warnings.join('；')} />
                ) : null}
                <DatasetSummaryCard summary={splitSummary} />
              </Space>
            ) : null}
          </div>

          <div className="step-section">
            <Typography.Title level={4}>Step 5 创建任务</Typography.Title>
            <Button
              type="primary"
              size="large"
              icon={<RocketOutlined />}
              loading={creating}
              disabled={!splitSummary}
              onClick={submitCreate}
            >
              创建任务
            </Button>
          </div>
        </Form>
      </Card>
    </Space>
  );
}
