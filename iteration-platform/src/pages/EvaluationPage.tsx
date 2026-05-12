import {
  Alert,
  Card,
  Col,
  Descriptions,
  Empty,
  Input,
  Row,
  Select,
  Space,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getEvalMetrics, getEvaluationSummary, getPredictions } from '../services/evaluation';
import type {
  EvalDatasetName,
  EvalMetrics,
  EvaluationSummary,
  PredictionPage,
  PredictionQuery,
  PredictionRecord,
} from '../types/evaluation';

const datasets: EvalDatasetName[] = ['val', 'test_unseen', 'test_all', 'hard', 'replay'];

function percent(value?: number) {
  return value === undefined ? '-' : `${(value * 100).toFixed(2)}%`;
}

function numeric(value?: number) {
  return value === undefined ? '-' : value.toFixed(4);
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : '请求失败';
}

const metricColumns: ColumnsType<EvalMetrics> = [
  { title: 'dataset', dataIndex: 'dataset', fixed: 'left', width: 130 },
  {
    title: 'exists',
    dataIndex: 'exists',
    width: 100,
    render: (value: boolean) => <Tag color={value ? 'green' : 'default'}>{value ? '已生成' : '未生成'}</Tag>,
  },
  { title: 'numSamples', dataIndex: 'numSamples', width: 120 },
  { title: 'acc', dataIndex: 'acc', render: percent },
  { title: 'ap', dataIndex: 'ap', render: percent },
  { title: 'auroc', dataIndex: 'auroc', render: percent },
  { title: 'recall@p95', dataIndex: 'recall_p95', render: percent },
  { title: 'recall@p98', dataIndex: 'recall_p98', render: percent },
  { title: 'recall@p99', dataIndex: 'recall_p99', render: percent },
  { title: 'real_fpr', dataIndex: 'real_fpr', render: percent },
  { title: 'fake_fnr', dataIndex: 'fake_fnr', render: percent },
  {
    title: 'metricsPath',
    dataIndex: 'metricsPath',
    width: 260,
    ellipsis: true,
    render: (value?: string) => value || '-',
  },
];

const predictionColumns: ColumnsType<PredictionRecord> = [
  { title: 'path', dataIndex: 'path', ellipsis: true, width: 320 },
  {
    title: 'label',
    dataIndex: 'label',
    width: 90,
    render: (value: 0 | 1) => <Tag color={value === 1 ? 'volcano' : 'green'}>{value}</Tag>,
  },
  { title: 'probability', dataIndex: 'probability', width: 120, render: numeric },
  {
    title: 'prediction',
    dataIndex: 'prediction',
    width: 110,
    render: (value: 0 | 1) => <Tag color={value === 1 ? 'blue' : 'cyan'}>{value}</Tag>,
  },
  { title: 'source', dataIndex: 'source', width: 130 },
  { title: 'generator', dataIndex: 'generator', width: 130 },
  { title: 'split_hint', dataIndex: 'split_hint', width: 130 },
  { title: 'hard_type', dataIndex: 'hard_type', width: 120 },
  {
    title: 'is_error',
    dataIndex: 'is_error',
    width: 110,
    render: (value: boolean) => <Tag color={value ? 'red' : 'green'}>{value ? 'true' : 'false'}</Tag>,
  },
  {
    title: 'error_type',
    dataIndex: 'error_type',
    width: 150,
    render: (value?: PredictionRecord['error_type']) => {
      const color = value === 'false_positive' ? 'orange' : value === 'false_negative' ? 'red' : 'green';
      return <Tag color={color}>{value || '-'}</Tag>;
    },
  },
];

export default function EvaluationPage() {
  const { taskId = 'iter_20260511_001' } = useParams();
  const [summary, setSummary] = useState<EvaluationSummary>();
  const [selectedDataset, setSelectedDataset] = useState<EvalDatasetName>('val');
  const [selectedMetric, setSelectedMetric] = useState<EvalMetrics>();
  const [predictionPage, setPredictionPage] = useState<PredictionPage>();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [errorType, setErrorType] = useState<PredictionQuery['errorType']>('all');
  const [source, setSource] = useState<string>();
  const [generator, setGenerator] = useState<string>();
  const [splitHint, setSplitHint] = useState<string>();
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    setLoadingSummary(true);
    getEvaluationSummary(taskId)
      .then((data) => {
        setSummary(data);
        const firstExisting = data.datasets.find((item) => item.exists)?.dataset;
        if (firstExisting) setSelectedDataset(firstExisting);
      })
      .catch((err) => {
        const msg = getErrorMessage(err);
        setError(msg);
        message.error(msg);
      })
      .finally(() => setLoadingSummary(false));
  }, [taskId]);

  useEffect(() => {
    getEvalMetrics(taskId, selectedDataset)
      .then(setSelectedMetric)
      .catch((err) => setError(getErrorMessage(err)));
  }, [selectedDataset, taskId]);

  useEffect(() => {
    setLoadingPredictions(true);
    getPredictions(taskId, {
      dataset: selectedDataset,
      page,
      pageSize,
      errorType,
      source,
      generator,
      splitHint,
    })
      .then(setPredictionPage)
      .catch((err) => {
        const msg = getErrorMessage(err);
        setError(msg);
        message.error(msg);
      })
      .finally(() => setLoadingPredictions(false));
  }, [errorType, generator, page, pageSize, selectedDataset, source, splitHint, taskId]);

  const datasetExists = useMemo(() => {
    const metric = selectedMetric || summary?.datasets.find((item) => item.dataset === selectedDataset);
    return metric?.exists ?? false;
  }, [selectedDataset, selectedMetric, summary]);

  if (loadingSummary && !summary) {
    return (
      <div className="center-box">
        <Spin tip="加载评估结果" />
      </div>
    );
  }

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>评估结果</Typography.Title>
          <Typography.Paragraph type="secondary">
            {taskId} 的评估总览、测试集指标和 predictions 明细。
          </Typography.Paragraph>
        </div>
      </div>

      {error ? <Alert type="error" showIcon message="接口调用失败" description={error} /> : null}

      <Card title="评估状态" className="panel-card">
        <Descriptions column={1} size="small">
          <Descriptions.Item label="iterationId">{summary?.iterationId || taskId}</Descriptions.Item>
          <Descriptions.Item label="status">
            <Tag color={summary?.status === 'success' ? 'green' : summary?.status === 'missing' ? 'default' : 'blue'}>
              {summary?.status || 'unknown'}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="outputDir">{summary?.outputDir || '-'}</Descriptions.Item>
        </Descriptions>
        {summary?.status === 'missing' ? (
          <Alert
            type="warning"
            showIcon
            className="inner-alert"
            message="当前任务还没有评估结果，请先完成训练和评估。"
          />
        ) : null}
        {summary?.warnings.map((warning) => (
          <Alert key={warning} type="warning" showIcon className="inner-alert" message={warning} />
        ))}
      </Card>

      <Card title="指标总览" className="panel-card">
        <Table
          rowKey="dataset"
          columns={metricColumns}
          dataSource={summary?.datasets || []}
          pagination={false}
          scroll={{ x: 1280 }}
        />
      </Card>

      <Card title="Predictions 明细" className="panel-card">
        <Space direction="vertical" size={14} className="page-stack">
          <Tabs
            activeKey={selectedDataset}
            onChange={(key) => {
              setSelectedDataset(key as EvalDatasetName);
              setPage(1);
            }}
            items={datasets.map((dataset) => ({ key: dataset, label: dataset }))}
          />
          <Row gutter={[12, 12]}>
            <Col xs={24} md={6}>
              <Select
                value={errorType}
                className="full-width"
                onChange={(value: PredictionQuery['errorType']) => {
                  setErrorType(value);
                  setPage(1);
                }}
                options={[
                  { value: 'all', label: '全部样本' },
                  { value: 'false_positive', label: 'false_positive' },
                  { value: 'false_negative', label: 'false_negative' },
                ]}
              />
            </Col>
            <Col xs={24} md={6}>
              <Input
                allowClear
                placeholder="source"
                value={source}
                onChange={(event) => {
                  setSource(event.target.value || undefined);
                  setPage(1);
                }}
              />
            </Col>
            <Col xs={24} md={6}>
              <Input
                allowClear
                placeholder="generator"
                value={generator}
                onChange={(event) => {
                  setGenerator(event.target.value || undefined);
                  setPage(1);
                }}
              />
            </Col>
            <Col xs={24} md={6}>
              <Input
                allowClear
                placeholder="split_hint"
                value={splitHint}
                onChange={(event) => {
                  setSplitHint(event.target.value || undefined);
                  setPage(1);
                }}
              />
            </Col>
          </Row>

          {!datasetExists ? (
            <Empty description="该测试集未生成评估结果" />
          ) : (
            <Table
              rowKey="id"
              columns={predictionColumns}
              dataSource={predictionPage?.records || []}
              loading={loadingPredictions}
              pagination={{
                current: predictionPage?.page || page,
                pageSize: predictionPage?.pageSize || pageSize,
                total: predictionPage?.total || 0,
                showSizeChanger: true,
              }}
              onChange={(pagination) => {
                setPage(pagination.current || 1);
                setPageSize(pagination.pageSize || 20);
              }}
              locale={{ emptyText: <Empty description="暂无预测明细" /> }}
              scroll={{ x: 1500 }}
            />
          )}
        </Space>
      </Card>
    </Space>
  );
}
