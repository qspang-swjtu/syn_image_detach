import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Form,
  Input,
  Radio,
  Row,
  Space,
  Statistic,
  Typography,
  message,
} from 'antd';
import { FileSearchOutlined, MergeCellsOutlined, ScanOutlined } from '@ant-design/icons';
import { useEffect, useState } from 'react';
import { USE_MOCK } from '../config/env';
import DatasetFlow from '../components/DatasetFlow';
import ManifestBuilder from '../components/ManifestBuilder/ManifestBuilder';
import ManifestPreviewTable from '../components/ManifestPreviewTable';
import {
  getBaseDatasetInfo,
  getManifestPreview,
  mergeDatasetIndex,
  scanIncrementManifest,
} from '../services/dataset';
import type { DatasetInfo, ManifestPreview, MergeIndexResponse, ScanIncrementResponse } from '../types';

type DatasetFormValues = {
  iterationId: string;
  baseCsv: string;
  incrementManifest?: string;
};

type ManifestMode = 'existing' | 'manual';

const defaultValues: DatasetFormValues = {
  iterationId: 'iter_20260511_001',
  baseCsv: 'safepp_pytorch/manifests/base_index.csv',
  incrementManifest: 'safepp_pytorch/manifests/increment_manifest.yaml',
};

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '请求失败';
}

function hasUnseen(bySplitHint: Record<string, number>) {
  return Object.entries(bySplitHint).some(([key, value]) => key.toLowerCase().includes('unseen') && value > 0);
}

export default function DatasetPage() {
  const [form] = Form.useForm<DatasetFormValues>();
  const [baseInfo, setBaseInfo] = useState<DatasetInfo>();
  const [manifestPreview, setManifestPreview] = useState<ManifestPreview>();
  const [scanResult, setScanResult] = useState<ScanIncrementResponse>();
  const [mergeResult, setMergeResult] = useState<MergeIndexResponse>();
  const [error, setError] = useState<string>();
  const [loadingKey, setLoadingKey] = useState<string>();
  const [manifestMode, setManifestMode] = useState<ManifestMode>('existing');
  const currentIterationId = Form.useWatch('iterationId', form) || defaultValues.iterationId;

  const runWithLoading = async (key: string, action: () => Promise<void>) => {
    setError(undefined);
    setLoadingKey(key);
    try {
      await action();
    } catch (err) {
      const msg = getErrorMessage(err);
      setError(msg);
      message.error(msg);
    } finally {
      setLoadingKey(undefined);
    }
  };

  useEffect(() => {
    if (!USE_MOCK) return;
    getBaseDatasetInfo(defaultValues.baseCsv).then(setBaseInfo);
    getManifestPreview(defaultValues.incrementManifest || '').then(setManifestPreview);
  }, []);

  const readBaseInfo = () => {
    const { baseCsv } = form.getFieldsValue();
    if (!baseCsv) {
      message.warning('请先输入 base_csv 路径');
      return;
    }
    runWithLoading('base-info', async () => {
      setBaseInfo(await getBaseDatasetInfo(baseCsv));
    });
  };

  const previewManifest = () => {
    const { incrementManifest } = form.getFieldsValue();
    if (!incrementManifest) {
      message.warning('请先输入或保存 increment_manifest');
      return;
    }
    runWithLoading('manifest-preview', async () => {
      setManifestPreview(await getManifestPreview(incrementManifest));
    });
  };

  const scanManifest = () => {
    const { iterationId, incrementManifest } = form.getFieldsValue();
    if (!iterationId || !incrementManifest) {
      message.warning('请先输入 iterationId 并提供 increment_manifest');
      return;
    }
    runWithLoading('scan', async () => {
      setScanResult(await scanIncrementManifest(iterationId, incrementManifest));
      message.success('新增 manifest 扫描完成');
    });
  };

  const mergeIndex = () => {
    const { iterationId, baseCsv, incrementManifest } = form.getFieldsValue();
    if (!iterationId || !baseCsv) {
      message.warning('请先输入 iterationId 和 base_csv');
      return;
    }
    runWithLoading('merge', async () => {
      setMergeResult(await mergeDatasetIndex({ iterationId, baseCsv, incrementManifest }));
      message.success('本轮 all_samples.csv 已生成');
    });
  };

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>数据集管理</Typography.Title>
          <Typography.Paragraph type="secondary">
            管理基础 CSV、增量 manifest 和本轮 all_samples.csv。可以使用已有 manifest，也可以手动创建。
          </Typography.Paragraph>
        </div>
      </div>

      {error ? <Alert type="error" showIcon message="接口调用失败" description={error} /> : null}

      <Card title="数据准备输入" className="panel-card">
        <Form form={form} layout="vertical" initialValues={defaultValues}>
          <Form.Item label="Manifest 来源">
            <Radio.Group value={manifestMode} onChange={(event) => setManifestMode(event.target.value as ManifestMode)}>
              <Radio.Button value="existing">已有文件路径</Radio.Button>
              <Radio.Button value="manual">手动创建</Radio.Button>
            </Radio.Group>
          </Form.Item>
          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item name="iterationId" label="iteration_id">
                <Input />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="baseCsv" label="base_csv">
                <Input />
              </Form.Item>
            </Col>
            {manifestMode === 'existing' ? (
              <Col xs={24} md={8}>
                <Form.Item name="incrementManifest" label="increment_manifest">
                  <Input />
                </Form.Item>
              </Col>
            ) : null}
          </Row>
          <Space wrap>
            <Button icon={<FileSearchOutlined />} loading={loadingKey === 'base-info'} onClick={readBaseInfo}>
              读取基础数据集信息
            </Button>
            {manifestMode === 'existing' ? (
              <Button loading={loadingKey === 'manifest-preview'} onClick={previewManifest}>
                预览 Manifest
              </Button>
            ) : null}
            <Button icon={<ScanOutlined />} loading={loadingKey === 'scan'} onClick={scanManifest}>
              扫描新增 Manifest
            </Button>
            <Button type="primary" icon={<MergeCellsOutlined />} loading={loadingKey === 'merge'} onClick={mergeIndex}>
              合并本轮索引
            </Button>
          </Space>
        </Form>

        {manifestMode === 'manual' ? (
          <div className="manifest-builder-wrap">
            <ManifestBuilder
              iterationId={currentIterationId}
              value={form.getFieldValue('incrementManifest') as string | undefined}
              onManifestSaved={(manifestPath) => {
                form.setFieldValue('incrementManifest', manifestPath);
                setManifestPreview(undefined);
              }}
            />
            <Space wrap className="card-actions">
              <Button loading={loadingKey === 'manifest-preview'} onClick={previewManifest}>
                预览已保存 Manifest
              </Button>
            </Space>
          </div>
        ) : null}
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={10}>
          <Card title="基础数据集信息" className="panel-card">
            {baseInfo ? (
              <Descriptions column={1} size="small">
                <Descriptions.Item label="csvPath">{baseInfo.csvPath}</Descriptions.Item>
                <Descriptions.Item label="总样本数">{baseInfo.totalRows}</Descriptions.Item>
                <Descriptions.Item label="real 数量">{baseInfo.realCount}</Descriptions.Item>
                <Descriptions.Item label="fake 数量">{baseInfo.fakeCount}</Descriptions.Item>
                <Descriptions.Item label="seen 数量">{baseInfo.seenCount}</Descriptions.Item>
                <Descriptions.Item label="hard 数量">{baseInfo.hardCount}</Descriptions.Item>
                <Descriptions.Item label="unseen 数量">{baseInfo.unseenCount}</Descriptions.Item>
                <Descriptions.Item label="reviewed_pool 数量">{baseInfo.reviewedPoolCount}</Descriptions.Item>
                <Descriptions.Item label="最后更新时间">{baseInfo.lastModified || '-'}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Typography.Text type="secondary">输入 base_csv 后点击读取。</Typography.Text>
            )}
          </Card>
        </Col>
        <Col xs={24} xl={14}>
          <Card title="Manifest 预览" className="panel-card">
            {manifestPreview ? (
              <Space direction="vertical" size={12} className="page-stack">
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="manifestPath">{manifestPreview.manifestPath}</Descriptions.Item>
                  <Descriptions.Item label="estimatedRows">{manifestPreview.estimatedRows ?? '-'}</Descriptions.Item>
                </Descriptions>
                {manifestPreview.warnings.length > 0 ? (
                  <Alert type="warning" showIcon message={manifestPreview.warnings.join('；')} />
                ) : null}
                <ManifestPreviewTable data={manifestPreview.sources} />
              </Space>
            ) : (
              <Typography.Text type="secondary">输入或保存 increment_manifest 后点击预览。</Typography.Text>
            )}
          </Card>
        </Col>
      </Row>

      {scanResult ? (
        <Card title="新增 Manifest 扫描结果" className="panel-card">
          <Space direction="vertical" size={14} className="page-stack">
            <Descriptions column={1} size="small" className="path-descriptions">
              <Descriptions.Item label="incrementCsv">
                <Typography.Text copyable className="path-value">
                  {scanResult.incrementCsv}
                </Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="summaryYaml">
                <Typography.Text copyable className="path-value">
                  {scanResult.summaryYaml}
                </Typography.Text>
              </Descriptions.Item>
            </Descriptions>
            <Row gutter={[12, 12]}>
              <Col xs={24} md={8}>
                <div className="summary-tile">
                  <Statistic title="rows" value={scanResult.rows} />
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="summary-tile">
                  <Typography.Text type="secondary">byLabel</Typography.Text>
                  <Typography.Paragraph className="json-value">{JSON.stringify(scanResult.byLabel)}</Typography.Paragraph>
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="summary-tile">
                  <Typography.Text type="secondary">bySplitHint</Typography.Text>
                  <Typography.Paragraph className="json-value">
                    {JSON.stringify(scanResult.bySplitHint)}
                  </Typography.Paragraph>
                </div>
              </Col>
            </Row>
          </Space>
        </Card>
      ) : null}

      {mergeResult ? (
        <Card title="合并后索引" className="panel-card">
          {!hasUnseen(mergeResult.bySplitHint) ? (
            <Alert
              type="warning"
              showIcon
              message="当前没有 split_hint=unseen 的数据，本轮不会生成泛化测试集。"
              className="inner-alert"
            />
          ) : null}
          <Space direction="vertical" size={14} className="page-stack">
            <Descriptions column={1} size="small" className="path-descriptions">
              <Descriptions.Item label="allSamplesCsv">
                <Typography.Text copyable className="path-value">
                  {mergeResult.allSamplesCsv}
                </Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="summaryYaml">
                <Typography.Text copyable className="path-value">
                  {mergeResult.summaryYaml}
                </Typography.Text>
              </Descriptions.Item>
            </Descriptions>
            <Row gutter={[12, 12]}>
              <Col xs={12} md={4}>
                <div className="summary-tile">
                  <Statistic title="totalRows" value={mergeResult.totalRows} />
                </div>
              </Col>
              <Col xs={12} md={4}>
                <div className="summary-tile">
                  <Statistic title="duplicateRemoved" value={mergeResult.duplicateRemoved} />
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="summary-tile">
                  <Typography.Text type="secondary">byLabel</Typography.Text>
                  <Typography.Paragraph className="json-value">{JSON.stringify(mergeResult.byLabel)}</Typography.Paragraph>
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="summary-tile">
                  <Typography.Text type="secondary">bySplitHint</Typography.Text>
                  <Typography.Paragraph className="json-value">
                    {JSON.stringify(mergeResult.bySplitHint)}
                  </Typography.Paragraph>
                </div>
              </Col>
            </Row>
          </Space>
        </Card>
      ) : null}

      <Card title="数据切分可视化" className="panel-card">
        <DatasetFlow />
      </Card>

      <Card title="字段规范说明" className="panel-card">
        <Collapse
          items={[
            {
              key: 'required-fields',
              label: 'CSV 必要字段',
              children: (
                <div className="field-spec">
                  <Typography.Paragraph>
                    必要字段：path、label、source、generator、split_hint、sample_weight、is_hard_negative。
                  </Typography.Paragraph>
                  <ul>
                    <li>label=0 表示 real</li>
                    <li>label=1 表示 fake</li>
                    <li>split_hint=hard 表示已收集 hard 样本</li>
                    <li>split_hint=unseen 表示泛化测试集</li>
                    <li>is_hard_negative=1 表示困难样本，可提高采样权重</li>
                  </ul>
                </div>
              ),
            },
          ]}
        />
      </Card>
    </Space>
  );
}
